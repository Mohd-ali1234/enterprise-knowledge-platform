"""Stage: Query Rewriting.

Four collaborating pieces:

* `GeminiQueryRewriter`    - asks a hosted LLM for clearer variants.
* `OllamaQueryRewriter`    - the same, against a local Ollama server.
* `HeuristicQueryRewriter` - dependency-free fallback, always available.
* `ResilientQueryRewriter` - tries the primary, degrades to the fallback.

The pipeline only ever talks to the `QueryRewriter` protocol, so swapping one
LLM for another means adding one adapter here and one line in the composition
root.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.exceptions import LLMError, QueryRewriteError
from app.core.logging import get_logger
from app.domain.query import RewriteResult, RewriteStrategy, RewrittenQuery
from app.llm.gemini import GeminiClient
from app.ports.query_rewriter import QueryRewriter

logger = get_logger(__name__)

REWRITE_VARIANT_COUNT = 3

_WORD = re.compile(r"\w+")

# Conversational filler that adds no retrieval signal.
_FILLER_TERMS = frozenset(
    {
        "hey", "hi", "hello", "dude", "bro", "please", "pls", "thanks",
        "tell", "me", "about", "like", "that", "just", "kinda", "really",
        "wanna", "gimme", "ok", "okay",
    }
)


# ── LLM-backed rewriters ─────────────────────────────────────────


class _LLMRewriteEnvelope(BaseModel):
    """Schema the model is asked to return; validates its JSON reply."""

    original_query: str = ""
    rewritten_queries: list[RewrittenQuery] = Field(
        min_length=REWRITE_VARIANT_COUNT,
        max_length=REWRITE_VARIANT_COUNT,
    )


REWRITE_SYSTEM_PROMPT = f"""You are a query rewriting assistant for an enterprise RAG system.
Your task: rewrite ambiguous or underspecified user queries into {REWRITE_VARIANT_COUNT} clearer, more specific variants.

RULES:
1. NEVER answer the question. Only rewrite it.
2. Preserve the original user intent exactly.
3. Produce exactly {REWRITE_VARIANT_COUNT} rewritten queries.
4. Each rewrite must use a different strategy:
   - EXPAND: Add missing context/scope (e.g. "policy" -> "company expense reimbursement policy")
   - DISAMBIGUATE: Resolve ambiguous terms (e.g. "Apple" -> "Apple Inc. technology company")
   - SPECIFY: Add constraints (timeframe, department, document type)
5. Return ONLY valid JSON matching the schema below.

SCHEMA:
{{
  "original_query": "string",
  "rewritten_queries": [
    {{"query": "string", "strategy": "EXPAND|DISAMBIGUATE|SPECIFY", "confidence": 0.0-1.0}}
  ]
}}"""


def parse_rewrite_envelope(body: str) -> list[RewrittenQuery]:
    """Validate a model's JSON reply into exactly `REWRITE_VARIANT_COUNT` variants.

    Shared by every LLM-backed rewriter: the prompt and the schema are the
    same whichever provider produced the text.
    """
    try:
        envelope = _LLMRewriteEnvelope.model_validate(json.loads(body))
    except json.JSONDecodeError as exc:
        raise QueryRewriteError(f"Invalid JSON from model: {exc}") from exc
    except ValidationError as exc:
        raise QueryRewriteError(
            f"Model did not return {REWRITE_VARIANT_COUNT} valid rewrites: {exc}"
        ) from exc

    return envelope.rewritten_queries


class GeminiQueryRewriter:
    """Rewrites queries with Gemini.

    Thin by design: the prompt, the schema and the parsing are shared with the
    other LLM rewriters, and everything provider-specific lives in
    `GeminiClient`.
    """

    name = "gemini"

    SYSTEM_PROMPT = REWRITE_SYSTEM_PROMPT

    def __init__(self, client: GeminiClient | None = None):
        self._client = client or GeminiClient()

    @property
    def is_configured(self) -> bool:
        return self._client.is_configured

    async def rewrite(self, query: str) -> RewriteResult:
        try:
            body = await self._client.generate(
                prompt=f"Original query: {query}\n\nJSON response:",
                system=self.SYSTEM_PROMPT,
                json_mode=True,
            )
        except LLMError as exc:
            raise QueryRewriteError(str(exc)) from exc

        return RewriteResult(queries=parse_rewrite_envelope(body), rewriter=self.name)

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaQueryRewriter:
    """Rewrites queries with a local model served by Ollama."""

    name = "ollama"

    SYSTEM_PROMPT = REWRITE_SYSTEM_PROMPT

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.ollama_timeout_seconds
        self.temperature = (
            temperature if temperature is not None else settings.ollama_temperature
        )
        self._client = client or httpx.AsyncClient(timeout=self.timeout)

    async def rewrite(self, query: str) -> RewriteResult:
        payload = {
            "model": self.model,
            "prompt": f"{self.SYSTEM_PROMPT}\n\nOriginal query: {query}\n\nJSON response:",
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        try:
            response = await self._client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            body = response.json().get("response", "").strip()
        except httpx.HTTPError as exc:
            raise QueryRewriteError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise QueryRewriteError(f"Ollama returned a malformed response: {exc}") from exc

        return RewriteResult(queries=parse_rewrite_envelope(body), rewriter=self.name)

    async def aclose(self) -> None:
        await self._client.aclose()


# ── Heuristic rewriter ───────────────────────────────────────────


class HeuristicQueryRewriter:
    """Rule-based rewriter that needs no model and never fails.

    Strips conversational filler and emits one variant per strategy. Domain
    vocabulary is injected rather than hard-coded, so the platform stays
    domain-neutral: pass `synonyms={"pto": "paid time off leave policy"}` to
    teach it a corpus's terminology.
    """

    name = "heuristic"

    def __init__(self, synonyms: Mapping[str, str] | None = None):
        self._synonyms = {key.lower(): value for key, value in (synonyms or {}).items()}

    async def rewrite(self, query: str) -> RewriteResult:
        core = self._core_terms(query)
        expansion = self._expansion(query)
        subject = f"{core} {expansion}".strip()

        variants = [
            RewrittenQuery(
                query=f"What do the indexed documents say about {subject}?",
                strategy=RewriteStrategy.EXPAND,
                confidence=0.72,
            ),
            RewrittenQuery(
                query=f"Which document or policy section covers {subject}?",
                strategy=RewriteStrategy.DISAMBIGUATE,
                confidence=0.68,
            ),
            RewrittenQuery(
                query=f"What are the specific rules, limits, and conditions for {core}?",
                strategy=RewriteStrategy.SPECIFY,
                confidence=0.66,
            ),
        ]
        return RewriteResult(queries=variants, rewriter=self.name)

    def _core_terms(self, query: str) -> str:
        """Drop filler words, keeping the query's substantive terms."""
        terms = [token for token in _WORD.findall(query.lower()) if token not in _FILLER_TERMS]
        return " ".join(terms) or query.strip()

    def _expansion(self, query: str) -> str:
        """Append configured synonyms for any recognised term, de-duplicated."""
        tokens = set(_WORD.findall(query.lower()))
        expanded = " ".join(
            self._synonyms[token] for token in tokens if token in self._synonyms
        )
        return " ".join(dict.fromkeys(expanded.split()))


# ── Resilient composition ────────────────────────────────────────


class ResilientQueryRewriter:
    """Tries a primary rewriter and falls back when it is unavailable.

    Query rewriting is an enhancement, never a hard dependency: if the LLM is
    down the request still succeeds, with a warning attached to the response.
    """

    name = "resilient"

    def __init__(self, primary: QueryRewriter, fallback: QueryRewriter):
        self._primary = primary
        self._fallback = fallback

    async def rewrite(self, query: str) -> RewriteResult:
        try:
            return await self._primary.rewrite(query)
        except QueryRewriteError as exc:
            logger.warning("Primary rewriter '%s' failed: %s", self._primary.name, exc)
            result = await self._fallback.rewrite(query)
            result.warning = f"{exc}. Used local fallback rewrite."
            return result

    async def aclose(self) -> None:
        """Release resources held by either rewriter."""
        for rewriter in (self._primary, self._fallback):
            closer = getattr(rewriter, "aclose", None)
            if closer is not None:
                await closer()
