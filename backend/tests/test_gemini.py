"""Tests for the Gemini adapters.

The HTTP client is always substituted, so nothing here reaches the network or
needs an API key. Live checks are marked `integration` and require a real
`GEMINI_API_KEY`.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from app.core.config import settings
from app.core.exceptions import AnswerGenerationError, LLMError, QueryRewriteError
from app.domain.query import AnswerContext, RewriteStrategy
from app.llm.gemini import GeminiClient
from app.query.generation import (
    NO_CONTEXT_ANSWER,
    ExtractiveAnswerGenerator,
    GeminiAnswerGenerator,
    ResilientAnswerGenerator,
)
from app.query.rewriting import (
    REWRITE_VARIANT_COUNT,
    GeminiQueryRewriter,
    HeuristicQueryRewriter,
    ResilientQueryRewriter,
)
from tests.conftest import make_match


def gemini_reply(text: str) -> dict:
    """A well-formed `generateContent` response carrying `text`."""
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def rewrite_json() -> str:
    return json.dumps(
        {
            "original_query": "policy",
            "rewritten_queries": [
                {"query": "company expense policy", "strategy": "EXPAND", "confidence": 0.95},
                {"query": "which policy document", "strategy": "DISAMBIGUATE", "confidence": 0.9},
                {"query": "2024 expense policy", "strategy": "SPECIFY", "confidence": 0.88},
            ],
        }
    )


def stub_client(payload: dict, status: int = 200) -> GeminiClient:
    """A configured client whose transport returns `payload`."""
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status = Mock()
    if status != 200:
        request = httpx.Request("POST", "https://example.invalid")
        real = httpx.Response(status, json=payload, request=request)
        response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("error", request=request, response=real)
        )

    transport = Mock()
    transport.post = AsyncMock(return_value=response)
    return GeminiClient(api_key="test-key", client=transport)


def failing_client(exc: Exception) -> GeminiClient:
    transport = Mock()
    transport.post = AsyncMock(side_effect=exc)
    return GeminiClient(api_key="test-key", client=transport)


@pytest.fixture
def unconfigured(monkeypatch) -> None:
    """Simulate a deployment with no API key.

    Keeps these tests independent of whether the developer running them has
    `GEMINI_API_KEY` set in their own `.env`.
    """
    monkeypatch.setattr(settings, "gemini_api_key", None)


@pytest.fixture
def context() -> AnswerContext:
    return AnswerContext(
        text="[1] Employees may expense meals up to $75 per day.",
        chunks=[make_match("doc-1_0", "Employees may expense meals up to $75 per day.", 0.9)],
    )


class TestGeminiClient:
    async def test_returns_the_candidate_text(self):
        assert await stub_client(gemini_reply("hello")).generate("hi") == "hello"

    async def test_joins_multipart_replies(self):
        payload = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}

        assert await stub_client(payload).generate("hi") == "ab"

    async def test_is_not_configured_without_a_key(self, unconfigured):
        # `api_key=None` means "take it from settings", so the absence of a key
        # has to be simulated there rather than through the argument.
        assert GeminiClient().is_configured is False
        assert GeminiClient(api_key="k").is_configured is True

    async def test_refuses_to_call_without_a_key(self, unconfigured):
        with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
            await GeminiClient().generate("hi")

    async def test_sends_the_key_as_a_header_not_a_query_param(self):
        """The key must not land in a URL, where proxies and logs would keep it."""
        client = stub_client(gemini_reply("ok"))

        await client.generate("hi")

        _, kwargs = client._client.post.call_args
        assert kwargs["headers"]["x-goog-api-key"] == "test-key"

    async def test_json_mode_constrains_the_response_type(self):
        client = stub_client(gemini_reply("{}"))

        await client.generate("hi", json_mode=True)

        _, kwargs = client._client.post.call_args
        assert kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"

    async def test_system_instruction_is_sent_separately_from_the_prompt(self):
        client = stub_client(gemini_reply("ok"))

        await client.generate("the prompt", system="the rules")

        _, kwargs = client._client.post.call_args
        assert kwargs["json"]["systemInstruction"]["parts"][0]["text"] == "the rules"
        assert kwargs["json"]["contents"][0]["parts"][0]["text"] == "the prompt"

    async def test_a_blocked_prompt_becomes_an_llm_error(self):
        payload = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}

        with pytest.raises(LLMError, match="SAFETY"):
            await stub_client(payload).generate("hi")

    async def test_an_empty_candidate_becomes_an_llm_error(self):
        payload = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}

        with pytest.raises(LLMError, match="MAX_TOKENS"):
            await stub_client(payload).generate("hi")

    async def test_surfaces_the_api_error_message(self):
        payload = {"error": {"message": "API key not valid"}}

        with pytest.raises(LLMError, match="API key not valid"):
            await stub_client(payload, status=400).generate("hi")

    async def test_wraps_transport_failures(self):
        with pytest.raises(LLMError, match="Gemini request failed"):
            await failing_client(httpx.ConnectError("refused")).generate("hi")


class TestGeminiQueryRewriter:
    async def test_returns_one_variant_per_strategy(self):
        rewriter = GeminiQueryRewriter(stub_client(gemini_reply(rewrite_json())))

        result = await rewriter.rewrite("policy")

        assert len(result.queries) == REWRITE_VARIANT_COUNT
        assert result.rewriter == "gemini"
        assert {variant.strategy for variant in result.queries} == set(RewriteStrategy)

    async def test_rejects_invalid_json(self):
        rewriter = GeminiQueryRewriter(stub_client(gemini_reply("not json {{")))

        with pytest.raises(QueryRewriteError, match="Invalid JSON"):
            await rewriter.rewrite("policy")

    async def test_rejects_the_wrong_number_of_variants(self):
        short = json.dumps(
            {"rewritten_queries": [{"query": "q", "strategy": "EXPAND", "confidence": 0.9}]}
        )
        rewriter = GeminiQueryRewriter(stub_client(gemini_reply(short)))

        with pytest.raises(QueryRewriteError, match="valid rewrites"):
            await rewriter.rewrite("policy")

    async def test_translates_llm_errors_into_stage_errors(self):
        """`ResilientQueryRewriter` only catches `QueryRewriteError`."""
        rewriter = GeminiQueryRewriter(failing_client(httpx.ConnectError("refused")))

        with pytest.raises(QueryRewriteError):
            await rewriter.rewrite("policy")

    async def test_resilient_falls_back_to_the_heuristic(self):
        resilient = ResilientQueryRewriter(
            primary=GeminiQueryRewriter(failing_client(httpx.ConnectError("refused"))),
            fallback=HeuristicQueryRewriter(),
        )

        result = await resilient.rewrite("hey dude tell me about leave")

        assert result.rewriter == "heuristic"
        assert "Used local fallback rewrite" in result.warning


def answer_json(**overrides) -> str:
    payload = {
        "answer": "Meals are capped at $75 [1].",
        "key_points": ["Meals are capped at $75 per day [1]."],
        "related_questions": ["What is the receipt threshold?"],
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestGeminiAnswerGenerator:
    async def test_returns_the_models_answer(self, context):
        generator = GeminiAnswerGenerator(stub_client(gemini_reply(answer_json())))

        answer = await generator.generate("expense policy?", context)

        assert answer.text == "Meals are capped at $75 [1]."
        assert answer.generator == "gemini"

    async def test_returns_key_points_and_related_questions(self, context):
        generator = GeminiAnswerGenerator(stub_client(gemini_reply(answer_json())))

        answer = await generator.generate("expense policy?", context)

        assert answer.key_points == ["Meals are capped at $75 per day [1]."]
        assert answer.related_questions == ["What is the receipt threshold?"]

    async def test_asks_for_json_so_the_extras_can_be_parsed(self, context):
        client = stub_client(gemini_reply(answer_json()))

        await GeminiAnswerGenerator(client).generate("q", context)

        _, kwargs = client._client.post.call_args
        assert kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"

    async def test_extras_are_capped(self, context):
        """A model that ignores the limits must not flood the UI."""
        generator = GeminiAnswerGenerator(
            stub_client(
                gemini_reply(
                    answer_json(
                        key_points=[f"point {i}" for i in range(20)],
                        related_questions=[f"q{i}?" for i in range(20)],
                    )
                )
            )
        )

        answer = await generator.generate("q", context)

        assert len(answer.key_points) == 5
        assert len(answer.related_questions) == 4

    async def test_a_non_json_reply_still_answers(self, context):
        """A formatting slip should cost the extras, not the answer."""
        generator = GeminiAnswerGenerator(stub_client(gemini_reply("Just prose, no JSON.")))

        answer = await generator.generate("q", context)

        assert answer.text == "Just prose, no JSON."
        assert answer.key_points == []

    async def test_malformed_extras_are_ignored_not_fatal(self, context):
        generator = GeminiAnswerGenerator(
            stub_client(gemini_reply(answer_json(key_points="not a list", related_questions=None)))
        )

        answer = await generator.generate("q", context)

        assert answer.text
        assert answer.key_points == []
        assert answer.related_questions == []

    async def test_an_empty_answer_field_is_an_error(self, context):
        generator = GeminiAnswerGenerator(stub_client(gemini_reply(answer_json(answer="  "))))

        with pytest.raises(AnswerGenerationError, match="no answer text"):
            await generator.generate("q", context)

    async def test_grounds_the_prompt_in_the_numbered_context(self, context):
        client = stub_client(gemini_reply("ok"))

        await GeminiAnswerGenerator(client).generate("expense policy?", context)

        _, kwargs = client._client.post.call_args
        assert context.text in kwargs["json"]["contents"][0]["parts"][0]["text"]

    async def test_skips_the_api_call_when_there_is_no_context(self):
        client = stub_client(gemini_reply("should not be used"))

        answer = await GeminiAnswerGenerator(client).generate("anything", AnswerContext())

        assert answer.text == NO_CONTEXT_ANSWER
        client._client.post.assert_not_awaited()

    async def test_translates_llm_errors_into_stage_errors(self, context):
        generator = GeminiAnswerGenerator(failing_client(httpx.ConnectError("refused")))

        with pytest.raises(AnswerGenerationError):
            await generator.generate("expense policy?", context)


class TestResilientAnswerGenerator:
    async def test_uses_the_primary_when_it_succeeds(self, context):
        resilient = ResilientAnswerGenerator(
            primary=GeminiAnswerGenerator(stub_client(gemini_reply("synthesised [1]"))),
            fallback=ExtractiveAnswerGenerator(),
        )

        answer = await resilient.generate("expense policy?", context)

        assert answer.generator == "gemini"

    async def test_degrades_to_extraction_when_the_primary_fails(self, context):
        resilient = ResilientAnswerGenerator(
            primary=GeminiAnswerGenerator(failing_client(httpx.ConnectError("refused"))),
            fallback=ExtractiveAnswerGenerator(),
        )

        answer = await resilient.generate("expense policy?", context)

        # The response names the generator that really answered, so a silent
        # downgrade is still visible to the caller.
        assert answer.generator == "extractive"
        assert "$75" in answer.text

    async def test_closing_is_safe_for_generators_without_resources(self):
        resilient = ResilientAnswerGenerator(
            ExtractiveAnswerGenerator(), ExtractiveAnswerGenerator()
        )

        await resilient.aclose()  # must not raise


class TestGeminiIntegration:
    """Live tests; run with `pytest -m integration` and a real GEMINI_API_KEY."""

    @pytest.mark.integration
    async def test_live_rewrite_of_an_ambiguous_query(self):
        if not settings.gemini_api_key:
            pytest.skip("GEMINI_API_KEY is not set")

        rewriter = GeminiQueryRewriter()
        try:
            result = await rewriter.rewrite("policy")
        finally:
            await rewriter.aclose()

        assert len(result.queries) == REWRITE_VARIANT_COUNT

    @pytest.mark.integration
    async def test_live_answer_is_grounded_in_the_context(self, context):
        if not settings.gemini_api_key:
            pytest.skip("GEMINI_API_KEY is not set")

        generator = GeminiAnswerGenerator()
        try:
            answer = await generator.generate("What is the meal limit?", context)
        finally:
            await generator.aclose()

        assert "75" in answer.text
