"""Tests for filtering, fusion, and the dense/sparse/hybrid retrievers."""

from __future__ import annotations

import pytest

from app.domain.documents import DocumentChunk, DocumentMetadata, KnowledgeGraph, ProcessedDocument
from app.domain.retrieval import MatchSource, RetrievalMode, ScoreKind
from app.indexing.indexer import DocumentIndexer
from app.retrieval.dense import DenseRetriever
from app.retrieval.filters import matches, normalize_filters, to_chroma_where
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever
from tests.conftest import make_match

from datetime import datetime, timezone


def build_document(document_id: str, texts: list[str]) -> ProcessedDocument:
    return ProcessedDocument(
        metadata=DocumentMetadata(
            document_id=document_id,
            filename=f"{document_id}.pdf",
            file_type="pdf",
            file_size=1024,
            pages=1,
            char_count=sum(len(text) for text in texts),
            word_count=sum(len(text.split()) for text in texts),
            uploaded_at=datetime.now(timezone.utc),
        ),
        knowledge_graph=KnowledgeGraph(),
        chunks=[
            DocumentChunk(
                chunk_id=index,
                document_id=document_id,
                section_id=0,
                section_title="Policy",
                text=text,
                start=0,
                end=len(text),
            )
            for index, text in enumerate(texts)
        ],
    )


# ── Filters ──────────────────────────────────────────────────────


class TestFilters:
    def test_placeholder_values_are_not_constraints(self):
        placeholders = {"additionalProp1": {}, "empty_list": [], "missing": None}

        assert normalize_filters(placeholders) == {}
        assert to_chroma_where(placeholders) is None
        assert matches({"document_id": "doc-1"}, placeholders) is True

    def test_exact_match_is_enforced(self):
        assert matches({"document_id": "doc-1"}, {"document_id": "doc-1"}) is True
        assert matches({"document_id": "doc-2"}, {"document_id": "doc-1"}) is False

    def test_list_value_means_any_of(self):
        assert matches({"document_id": "doc-2"}, {"document_id": ["doc-1", "doc-2"]}) is True
        assert matches({"document_id": "doc-9"}, {"document_id": ["doc-1", "doc-2"]}) is False

    def test_chroma_clause_shape(self):
        assert to_chroma_where({"document_id": "doc-1"}) == {"document_id": "doc-1"}
        assert to_chroma_where({"document_id": ["a", "b"]}) == {"document_id": {"$in": ["a", "b"]}}

        combined = to_chroma_where({"document_id": "doc-1", "section_id": 2})
        assert combined == {"$and": [{"document_id": "doc-1"}, {"section_id": 2}]}


# ── Fusion ───────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    def test_chunks_found_by_both_strategies_rank_first(self):
        dense = [make_match("a", "alpha", 0.9), make_match("b", "beta", 0.5)]
        sparse = [make_match("b", "beta", 3.0), make_match("c", "gamma", 1.0)]
        sparse[0].source = MatchSource.KEYWORD
        sparse[1].source = MatchSource.KEYWORD

        fused = reciprocal_rank_fusion([dense, sparse], limit=3, k=60)

        assert fused[0].id == "b"
        assert fused[0].source is MatchSource.HYBRID
        assert ScoreKind.FUSION.value in fused[0].scores

    def test_respects_the_limit(self):
        dense = [make_match(str(index), "text", 0.5) for index in range(10)]
        assert len(reciprocal_rank_fusion([dense], limit=3)) == 3

    def test_empty_input_yields_no_results(self):
        assert reciprocal_rank_fusion([[], []], limit=5) == []

    def test_preserves_scores_from_every_strategy(self):
        dense = [make_match("a", "alpha", 0.9)]
        sparse = [make_match("a", "alpha", 4.0)]
        sparse[0].source = MatchSource.KEYWORD
        sparse[0].scores = {ScoreKind.KEYWORD.value: 4.0}

        fused = reciprocal_rank_fusion([dense, sparse], limit=1)

        assert fused[0].scores[ScoreKind.SEMANTIC.value] == 0.9
        assert fused[0].scores[ScoreKind.KEYWORD.value] == 4.0


# ── Retrievers ───────────────────────────────────────────────────


class TestRetrievers:
    @pytest.fixture
    def indexed_store(self, embedder, vector_store):
        indexer = DocumentIndexer(embedder, vector_store)
        indexer.index(
            build_document(
                "doc-1",
                [
                    "Employees accrue paid leave every month under the leave policy.",
                    "Travel expenses are reimbursed within thirty days.",
                ],
            )
        )
        indexer.index(build_document("doc-2", ["Payroll runs on the last working day."]))
        return vector_store

    def test_dense_retriever_returns_matches(self, embedder, indexed_store):
        results = DenseRetriever(embedder, indexed_store).search("leave policy", limit=2)

        assert results
        assert all(match.source is MatchSource.SEMANTIC for match in results)

    def test_sparse_retriever_matches_on_keywords(self, indexed_store):
        results = SparseRetriever(indexed_store).search("payroll", limit=5)

        assert [match.id for match in results] == ["doc-2_0"]
        assert results[0].source is MatchSource.KEYWORD

    def test_sparse_retriever_reindexes_when_the_corpus_grows(self, embedder, indexed_store):
        sparse = SparseRetriever(indexed_store)
        assert sparse.search("onboarding", limit=5) == []

        DocumentIndexer(embedder, indexed_store).index(
            build_document("doc-3", ["The onboarding checklist covers first-week tasks."])
        )

        # No explicit refresh: the retriever notices the corpus changed.
        assert [match.id for match in sparse.search("onboarding", limit=5)] == ["doc-3_0"]

    def test_filters_narrow_results_to_one_document(self, embedder, indexed_store):
        retriever = self._hybrid(embedder, indexed_store)

        results = retriever.retrieve(
            "policy", limit=5, filters={"document_id": "doc-2"}
        )

        assert results
        assert {match.document_id for match in results} == {"doc-2"}

    def test_modes_select_the_intended_strategy(self, embedder, indexed_store):
        retriever = self._hybrid(embedder, indexed_store)

        semantic = retriever.retrieve("leave", limit=3, mode=RetrievalMode.SEMANTIC)
        keyword = retriever.retrieve("leave", limit=3, mode=RetrievalMode.KEYWORD)
        hybrid = retriever.retrieve("leave", limit=3, mode=RetrievalMode.HYBRID)

        assert all(match.source is MatchSource.SEMANTIC for match in semantic)
        assert all(match.source is MatchSource.KEYWORD for match in keyword)
        assert all(ScoreKind.FUSION.value in match.scores for match in hybrid)

    def test_blank_query_returns_nothing(self, embedder, indexed_store):
        assert self._hybrid(embedder, indexed_store).retrieve("   ", limit=5) == []

    @staticmethod
    def _hybrid(embedder, store) -> HybridRetriever:
        return HybridRetriever(
            dense=DenseRetriever(embedder, store),
            sparse=SparseRetriever(store),
        )


class TestDocumentIndexer:
    def test_writes_one_record_per_chunk_with_flat_metadata(self, embedder, vector_store):
        document = build_document("doc-1", ["first chunk text", "second chunk text"])
        document.chunks[0].entities = ["Finance", "Priya"]

        written = DocumentIndexer(embedder, vector_store).index(document)

        assert written == 2
        assert vector_store.count() == 2
        record = vector_store.records["doc-1_0"]
        assert record.metadata["entities"] == "Finance, Priya"
        assert record.metadata["document_id"] == "doc-1"

    def test_reindexing_replaces_rather_than_duplicates(self, embedder, vector_store):
        document = build_document("doc-1", ["only chunk"])
        indexer = DocumentIndexer(embedder, vector_store)

        indexer.index(document)
        indexer.index(document)

        assert vector_store.count() == 1

    def test_notifies_the_listener_after_indexing(self, embedder, vector_store):
        calls: list[int] = []
        indexer = DocumentIndexer(embedder, vector_store, on_index=lambda: calls.append(1))

        indexer.index(build_document("doc-1", ["chunk"]))

        assert calls == [1]

    def test_document_without_chunks_indexes_nothing(self, embedder, vector_store):
        assert DocumentIndexer(embedder, vector_store).index(build_document("doc-1", [])) == 0
