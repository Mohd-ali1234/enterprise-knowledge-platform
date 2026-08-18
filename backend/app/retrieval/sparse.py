"""Sparse (keyword) retrieval over the indexed corpus.

BM25 is used when `rank_bm25` is installed, and a term-overlap scorer otherwise.
The index is rebuilt automatically whenever the vector store's size changes, so
newly indexed documents become keyword-searchable without a restart.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from app.core.logging import get_logger
from app.domain.retrieval import ChunkMatch, MatchSource, ScoreKind, StoredChunk
from app.ports.vector_store import VectorStore
from app.retrieval.filters import matches as metadata_matches
from app.retrieval.tokenization import tokenize

logger = get_logger(__name__)


class LexicalScorer(Protocol):
    """Scores a tokenised query against a tokenised corpus."""

    def get_scores(self, tokenized_query: Sequence[str]) -> Sequence[float]:
        ...


class TermOverlapScorer:
    """Fallback scorer: term overlap plus term density.

    Used when `rank_bm25` is unavailable. Ranks by how many query terms a
    document contains, tie-broken by how concentrated they are.
    """

    def __init__(self, corpus: list[list[str]]):
        self._corpus = corpus

    def get_scores(self, tokenized_query: Sequence[str]) -> list[float]:
        query_terms = set(tokenized_query)
        scores: list[float] = []

        for tokens in self._corpus:
            if not tokens:
                scores.append(0.0)
                continue

            hits = sum(1 for token in tokens if token in query_terms)
            overlap = len(query_terms & set(tokens))
            scores.append(float(overlap + hits / len(tokens)))

        return scores


def _build_scorer(corpus: list[list[str]]) -> LexicalScorer:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed; using term-overlap keyword scoring")
        return TermOverlapScorer(corpus)

    return BM25Okapi(corpus)


class SparseRetriever:
    """Keyword search over every chunk held in the vector store."""

    def __init__(self, store: VectorStore):
        self._store = store
        self._scorer: LexicalScorer | None = None
        self._chunks: list[StoredChunk] = []
        self._indexed_count = -1

    def refresh(self) -> None:
        """Rebuild the keyword index from the vector store."""
        self._chunks = self._store.list_all()
        corpus = [tokenize(chunk.text) for chunk in self._chunks]
        self._scorer = _build_scorer(corpus) if corpus else None
        self._indexed_count = len(self._chunks)
        logger.info("Built keyword index over %d chunks", self._indexed_count)

    def invalidate(self) -> None:
        """Mark the index stale so the next search rebuilds it."""
        self._indexed_count = -1

    def _ensure_current(self) -> None:
        """Rebuild if documents were added or removed since the last build."""
        if self._indexed_count != self._store.count():
            self.refresh()

    def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        self._ensure_current()
        if self._scorer is None:
            return []

        scores = self._scorer.get_scores(tokenize(query))

        candidates = [
            index
            for index, chunk in enumerate(self._chunks)
            if scores[index] > 0 and metadata_matches(chunk.metadata, filters)
        ]
        candidates.sort(key=lambda index: scores[index], reverse=True)

        return [
            ChunkMatch(
                id=self._chunks[index].id,
                text=self._chunks[index].text,
                metadata=self._chunks[index].metadata,
                score=float(scores[index]),
                source=MatchSource.KEYWORD,
                scores={ScoreKind.KEYWORD.value: float(scores[index])},
            )
            for index in candidates[:limit]
        ]
