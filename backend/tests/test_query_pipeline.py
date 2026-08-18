"""Tests for the query stages and the pipeline that sequences them."""

from __future__ import annotations

import pytest

from app.domain.query import AnswerContext, AnswerRoute, QueryIntent, QueryUnderstanding
from app.domain.retrieval import ScoreKind
from app.query.context import ContextBuilder
from app.query.generation import (
    NO_CONTEXT_ANSWER,
    AnswerGeneratorRegistry,
    ExtractiveAnswerGenerator,
)
from app.query.pipeline import QueryPipeline
from app.query.reranking import LexicalReranker
from app.query.rewriting import HeuristicQueryRewriter
from app.query.routing import AnswerRouter
from app.query.understanding import QueryUnderstandingService
from tests.conftest import StubRetriever, make_match


def build_pipeline(matches, **overrides) -> QueryPipeline:
    defaults = dict(
        retriever=StubRetriever(matches),
        rewriter=HeuristicQueryRewriter(),
        reranker=LexicalReranker(),
        generators=AnswerGeneratorRegistry(default=ExtractiveAnswerGenerator()),
    )
    return QueryPipeline(**{**defaults, **overrides})


# ── Understanding ────────────────────────────────────────────────


class TestQueryUnderstanding:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Summarize the leave policy", QueryIntent.SUMMARIZATION),
            ("Compare the 2023 and 2024 policies", QueryIntent.ANALYSIS),
            ("What is the leave policy?", QueryIntent.QUESTION_ANSWERING),
            ("leave policy", QueryIntent.LOOKUP),
        ],
    )
    def test_classifies_intent(self, query, expected):
        assert QueryUnderstandingService().understand(query).intent is expected

    def test_complexity_grows_with_breadth(self):
        service = QueryUnderstandingService()

        simple = service.understand("leave policy")
        complex_query = service.understand(
            "Compare all leave policies across every department and explain the impact"
        )

        assert complex_query.complexity_score > simple.complexity_score
        assert 0.0 <= complex_query.complexity_score <= 1.0

    def test_reports_signals(self):
        understanding = QueryUnderstandingService().understand("What does Acme Corp pay?")

        assert understanding.signals["has_question_word"] is True
        assert understanding.signals["token_count"] == 5
        assert "Acme Corp" in understanding.entities


# ── Reranking ────────────────────────────────────────────────────


class TestLexicalReranker:
    def test_promotes_the_chunk_that_shares_query_terms(self):
        matches = [
            make_match("a", "Unrelated content about facilities.", 0.5),
            make_match("b", "The payroll schedule and salary details.", 0.5),
        ]

        reranked = LexicalReranker().rerank("payroll salary", matches, limit=2)

        assert reranked[0].id == "b"
        assert ScoreKind.RERANK.value in reranked[0].scores

    def test_truncates_to_the_limit(self):
        matches = [make_match(str(i), "text", 0.5) for i in range(5)]
        assert len(LexicalReranker().rerank("text", matches, limit=2)) == 2

    def test_empty_input_yields_no_results(self):
        assert LexicalReranker().rerank("anything", [], limit=5) == []

    def test_does_not_mutate_its_input(self):
        matches = [make_match("a", "payroll salary", 0.5)]

        LexicalReranker().rerank("payroll", matches, limit=1)

        assert matches[0].scores == {ScoreKind.SEMANTIC.value: 0.5}


# ── Context building ─────────────────────────────────────────────


class TestContextBuilder:
    def test_numbers_citations_in_rank_order(self):
        context = ContextBuilder().build(
            [make_match("doc-1_0", "First.", 0.9, section_title="One"),
             make_match("doc-1_1", "Second.", 0.8, section_title="Two")]
        )

        assert [citation.citation_id for citation in context.citations] == [1, 2]
        assert context.citations[0].chunk_id == "doc-1_0"
        assert context.citations[0].section_title == "One"
        assert context.text.startswith("[1] First.")

    def test_drops_duplicate_chunk_text(self):
        context = ContextBuilder().build(
            [make_match("a", "Same text.", 0.9), make_match("b", "Same   text.", 0.8)]
        )

        assert len(context.chunks) == 1

    def test_stops_at_the_character_budget(self):
        builder = ContextBuilder(token_budget=10, chars_per_token=4)  # 40 chars
        matches = [make_match(str(i), "x" * 30, 0.9 - i / 10) for i in range(4)]

        context = builder.build(matches)

        assert len(context.chunks) == 1

    def test_always_admits_the_top_chunk(self):
        builder = ContextBuilder(token_budget=1, chars_per_token=1)

        context = builder.build([make_match("a", "a very long chunk of text", 0.9)])

        assert len(context.chunks) == 1

    def test_no_matches_gives_an_empty_context(self):
        context = ContextBuilder().build([])

        assert context.is_empty
        assert context.citations == []


# ── Routing ──────────────────────────────────────────────────────


class TestAnswerRouter:
    @staticmethod
    def understanding(intent=QueryIntent.LOOKUP, complexity=0.2) -> QueryUnderstanding:
        return QueryUnderstanding(intent=intent, complexity_score=complexity)

    def test_empty_context_routes_to_synthesis(self):
        decision = AnswerRouter().choose(self.understanding(), AnswerContext())

        assert decision.route is AnswerRoute.LOCAL_LLM
        assert decision.confidence == 0.0

    def test_high_confidence_factual_query_answers_directly(self):
        context = ContextBuilder().build([make_match("a", "text", 0.95)])

        decision = AnswerRouter().choose(self.understanding(), context)

        assert decision.route is AnswerRoute.DIRECT

    def test_complex_query_escalates(self):
        context = ContextBuilder().build([make_match("a", "text", 0.4)])

        decision = AnswerRouter().choose(
            self.understanding(intent=QueryIntent.ANALYSIS, complexity=0.85), context
        )

        assert decision.route is AnswerRoute.ONLINE_LLM

    def test_confidence_is_clamped_to_one(self):
        context = ContextBuilder().build([make_match("a", "text", 4.2)])

        assert AnswerRouter().choose(self.understanding(), context).confidence == 1.0


# ── Generation ───────────────────────────────────────────────────


class TestAnswerGeneration:
    async def test_quotes_context_with_citation_markers(self):
        context = ContextBuilder().build([make_match("a", "Leave accrues monthly.", 0.9)])

        answer = await ExtractiveAnswerGenerator().generate("leave?", context)

        assert answer.text == "Leave accrues monthly. [1]"
        assert answer.generator == "extractive"

    async def test_reports_when_nothing_was_retrieved(self):
        answer = await ExtractiveAnswerGenerator().generate("anything", AnswerContext())

        assert answer.text == NO_CONTEXT_ANSWER

    def test_registry_falls_back_to_the_default_generator(self):
        default = ExtractiveAnswerGenerator()
        registry = AnswerGeneratorRegistry(default=default)

        assert registry.resolve(AnswerRoute.ONLINE_LLM) is default

    def test_registry_prefers_a_registered_generator(self):
        default = ExtractiveAnswerGenerator()
        special = ExtractiveAnswerGenerator()
        registry = AnswerGeneratorRegistry(default=default)
        registry.register(AnswerRoute.ONLINE_LLM, special)

        assert registry.resolve(AnswerRoute.ONLINE_LLM) is special
        assert registry.resolve(AnswerRoute.DIRECT) is default


# ── Orchestrator ─────────────────────────────────────────────────


class TestQueryPipeline:
    async def test_returns_answer_citations_and_route(self, policy_matches):
        pipeline = build_pipeline(policy_matches)

        result = await pipeline.run("How does leave policy affect payroll?", top_k=2, rerank_top_k=2)

        assert result.answer.text
        assert result.route.route in set(AnswerRoute)
        assert len(result.context.citations) == 2
        assert result.context.citations[0].document_id == "doc-1"
        assert result.matches

    async def test_handles_empty_retrieval(self):
        pipeline = build_pipeline([])

        result = await pipeline.run("Unknown topic")

        assert result.answer.text == NO_CONTEXT_ANSWER
        assert result.route.confidence == 0.0
        assert result.context.citations == []

    async def test_skips_rewriting_by_default(self, policy_matches):
        retriever = StubRetriever(policy_matches)
        pipeline = build_pipeline(policy_matches, retriever=retriever)

        result = await pipeline.run("my pay")

        assert result.rewritten_queries == []
        assert result.retrieval_query == "my pay"
        assert retriever.last_query == "my pay"

    async def test_retrieves_with_the_rewritten_query(self, policy_matches):
        retriever = StubRetriever(policy_matches)
        pipeline = build_pipeline(policy_matches, retriever=retriever)

        result = await pipeline.run("hey dude tell me about my pay", rewrite=True)

        assert len(result.rewritten_queries) == 3
        assert result.retrieval_query != result.query
        assert retriever.last_query == result.retrieval_query

    async def test_reranks_against_the_original_query(self, policy_matches):
        """A rewrite may widen recall, but ranking must follow the user's words."""

        class RecordingReranker:
            seen: list[str] = []

            def rerank(self, query, matches, limit):  # noqa: ANN001
                self.seen.append(query)
                return list(matches)[:limit]

        reranker = RecordingReranker()
        pipeline = build_pipeline(policy_matches, reranker=reranker)

        await pipeline.run("hey dude tell me about my pay", rewrite=True)

        assert reranker.seen == ["hey dude tell me about my pay"]
