"""Stage: Hybrid Retrieval.

Combines dense (semantic) and sparse (keyword) retrieval through Reciprocal
Rank Fusion. The two strategies are independent objects, so either can be
replaced or used alone via `RetrievalMode`.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.retrieval import ChunkMatch, RetrievalMode
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import SparseRetriever

logger = get_logger(__name__)


class HybridRetriever:
    """Routes a search to the dense path, the sparse path, or both."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int | None = None,
    ):
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = settings.rrf_k if rrf_k is None else rrf_k

    def retrieve(
        self,
        query: str,
        limit: int,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        """Return the `limit` most relevant chunks for `query`."""
        if not query.strip():
            return []

        if mode is RetrievalMode.KEYWORD:
            matches = self._sparse.search(query, limit, filters)
        elif mode is RetrievalMode.SEMANTIC:
            matches = self._dense.search(query, limit, filters)
        else:
            matches = self._hybrid_search(query, limit, filters)

        logger.info("Retrieved %d chunks for mode=%s", len(matches), mode.value)
        return matches

    def _hybrid_search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[ChunkMatch]:
        dense_matches = self._dense.search(query, limit, filters)
        sparse_matches = self._sparse.search(query, limit, filters)
        return reciprocal_rank_fusion(
            [dense_matches, sparse_matches],
            limit=limit,
            k=self._rrf_k,
        )

    def invalidate_keyword_index(self) -> None:
        """Signal that the corpus changed and the sparse index is stale."""
        self._sparse.invalidate()
