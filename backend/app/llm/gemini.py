"""Adapter: Google Gemini via its `generateContent` REST API.

One thin async client shared by every stage that wants a hosted LLM. It speaks
HTTP directly with `httpx` rather than pulling in a vendor SDK, matching how the
Ollama adapter is written, and returns plain text - callers own the prompting
and the parsing.

`is_configured` is the switch the composition root reads: with no API key the
client refuses to make a call, and each stage falls back to its offline path.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Minimal async client for the Gemini text generation endpoint."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self.timeout = (
            timeout if timeout is not None else settings.gemini_timeout_seconds
        )
        self.temperature = (
            temperature if temperature is not None else settings.gemini_temperature
        )
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.gemini_max_output_tokens
        )
        self._client = client or httpx.AsyncClient(timeout=self.timeout)

    @property
    def is_configured(self) -> bool:
        """Whether an API key is present. False means every call would fail."""
        return bool(self.api_key)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Return the model's text reply.

        Raises `LLMError` for every failure mode - missing key, transport
        error, HTTP error, blocked or empty candidate - so callers have one
        exception type to catch.
        """
        if not self.is_configured:
            raise LLMError(
                "GEMINI_API_KEY is not set; add it to backend/.env to enable Gemini."
            )

        try:
            response = await self._client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json=self._payload(prompt, system, json_mode, temperature),
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Gemini returned {exc.response.status_code}: {self._error_detail(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"Gemini returned a malformed response: {exc}") from exc

        return self._extract_text(body)

    def _payload(
        self,
        prompt: str,
        system: str | None,
        json_mode: bool,
        temperature: float | None,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "maxOutputTokens": self.max_output_tokens,
        }
        # Constrained decoding: the model is told to emit JSON at the transport
        # level, which removes the "here is your JSON:" preamble problem.
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        """Pull the reply text out of a `generateContent` response.

        A response with no candidates is normal rather than exceptional - it is
        how a safety block or a token-limit stop is reported - so it becomes an
        `LLMError` and the caller degrades.
        """
        candidates = body.get("candidates") or []
        if not candidates:
            reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no usable output ({reason}).")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()

        if not text:
            raise LLMError(
                f"Gemini returned an empty reply (finishReason="
                f"{candidate.get('finishReason', 'unknown')})."
            )
        return text

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Best-effort extraction of Google's error message, for logs."""
        try:
            return (response.json().get("error") or {}).get("message", response.text[:200])
        except ValueError:
            return response.text[:200]

    async def aclose(self) -> None:
        await self._client.aclose()
