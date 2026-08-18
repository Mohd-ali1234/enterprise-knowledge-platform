"""Tests for the query rewriting stage."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from app.core.exceptions import QueryRewriteError
from app.domain.query import RewriteStrategy
from app.query.rewriting import (
    REWRITE_VARIANT_COUNT,
    HeuristicQueryRewriter,
    OllamaQueryRewriter,
    ResilientQueryRewriter,
)


def ollama_reply(variants: list[dict] | None = None) -> dict:
    return {
        "response": json.dumps(
            {
                "original_query": "policy",
                "rewritten_queries": variants
                or [
                    {
                        "query": "What is the company expense reimbursement policy?",
                        "strategy": "EXPAND",
                        "confidence": 0.95,
                    },
                    {
                        "query": "Which policy document covers expense reimbursement?",
                        "strategy": "DISAMBIGUATE",
                        "confidence": 0.90,
                    },
                    {
                        "query": "What is the 2024 expense reimbursement policy?",
                        "strategy": "SPECIFY",
                        "confidence": 0.88,
                    },
                ],
            }
        )
    }


def mock_post(rewriter: OllamaQueryRewriter, payload: dict):
    """Patch the rewriter's HTTP client to return `payload`."""
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return patch.object(rewriter, "_client", **{"post": AsyncMock(return_value=response)})


class TestOllamaQueryRewriter:
    @pytest.fixture
    def rewriter(self) -> OllamaQueryRewriter:
        return OllamaQueryRewriter(model="test-model", base_url="http://localhost:11434")

    async def test_returns_three_distinct_strategies(self, rewriter):
        with mock_post(rewriter, ollama_reply()):
            result = await rewriter.rewrite("policy")

        assert len(result.queries) == REWRITE_VARIANT_COUNT
        assert result.rewriter == "ollama"
        assert result.warning is None
        assert {variant.strategy for variant in result.queries} == set(RewriteStrategy)

    async def test_best_variant_is_the_first(self, rewriter):
        with mock_post(rewriter, ollama_reply()):
            result = await rewriter.rewrite("policy")

        assert result.best is result.queries[0]

    async def test_rejects_invalid_json(self, rewriter):
        with mock_post(rewriter, {"response": "not valid json {{"}):
            with pytest.raises(QueryRewriteError, match="Invalid JSON"):
                await rewriter.rewrite("policy")

    async def test_rejects_the_wrong_number_of_variants(self, rewriter):
        short = [{"query": "q1", "strategy": "EXPAND", "confidence": 0.9}]

        with mock_post(rewriter, ollama_reply(short)):
            with pytest.raises(QueryRewriteError, match="valid rewrites"):
                await rewriter.rewrite("policy")

    async def test_rejects_an_out_of_range_confidence(self, rewriter):
        invalid = [
            {"query": f"q{index}", "strategy": "EXPAND", "confidence": 4.0}
            for index in range(REWRITE_VARIANT_COUNT)
        ]

        with mock_post(rewriter, ollama_reply(invalid)):
            with pytest.raises(QueryRewriteError):
                await rewriter.rewrite("policy")

    async def test_wraps_transport_failures(self, rewriter):
        with patch.object(
            rewriter,
            "_client",
            **{"post": AsyncMock(side_effect=httpx.ConnectError("refused"))},
        ):
            with pytest.raises(QueryRewriteError, match="Ollama request failed"):
                await rewriter.rewrite("policy")


class TestHeuristicQueryRewriter:
    async def test_always_produces_one_variant_per_strategy(self):
        result = await HeuristicQueryRewriter().rewrite("hey dude tell me about my pay")

        assert len(result.queries) == REWRITE_VARIANT_COUNT
        assert {variant.strategy for variant in result.queries} == set(RewriteStrategy)
        assert result.rewriter == "heuristic"

    async def test_strips_conversational_filler(self):
        result = await HeuristicQueryRewriter().rewrite("hey bro please tell me about payroll")

        assert "payroll" in result.best.query.lower()
        for filler in ("hey", "bro", "please"):
            assert filler not in result.best.query.lower()

    async def test_applies_injected_domain_synonyms(self):
        rewriter = HeuristicQueryRewriter(synonyms={"pto": "paid time off leave policy"})

        result = await rewriter.rewrite("what is our PTO")

        assert "paid time off" in result.best.query.lower()

    async def test_domain_vocabulary_is_not_assumed(self):
        """The platform is domain-neutral: no built-in corpus vocabulary."""
        result = await HeuristicQueryRewriter().rewrite("what is our PTO")

        assert "paid time off" not in result.best.query.lower()

    async def test_empty_query_still_yields_variants(self):
        result = await HeuristicQueryRewriter().rewrite("")

        assert len(result.queries) == REWRITE_VARIANT_COUNT


class TestResilientQueryRewriter:
    async def test_uses_the_primary_when_it_succeeds(self):
        primary = OllamaQueryRewriter()
        resilient = ResilientQueryRewriter(primary, HeuristicQueryRewriter())

        with mock_post(primary, ollama_reply()):
            result = await resilient.rewrite("policy")

        assert result.rewriter == "ollama"
        assert result.warning is None

    async def test_falls_back_when_the_primary_is_down(self):
        primary = OllamaQueryRewriter()
        resilient = ResilientQueryRewriter(primary, HeuristicQueryRewriter())

        with patch.object(
            primary,
            "_client",
            **{"post": AsyncMock(side_effect=httpx.ConnectError("All connection attempts failed"))},
        ):
            result = await resilient.rewrite("hey dude tell me about my pay")

        assert result.rewriter == "heuristic"
        assert len(result.queries) == REWRITE_VARIANT_COUNT
        assert "Used local fallback rewrite" in result.warning

    async def test_closing_is_safe_for_rewriters_without_resources(self):
        resilient = ResilientQueryRewriter(HeuristicQueryRewriter(), HeuristicQueryRewriter())

        await resilient.aclose()  # must not raise


class TestOllamaIntegration:
    """Live tests; run with `pytest -m integration` and Ollama running."""

    @pytest.mark.integration
    async def test_live_rewrite_of_an_ambiguous_query(self):
        rewriter = OllamaQueryRewriter()
        try:
            result = await rewriter.rewrite("policy")
        finally:
            await rewriter.aclose()

        assert len(result.queries) == REWRITE_VARIANT_COUNT
        assert {variant.strategy for variant in result.queries} == set(RewriteStrategy)
