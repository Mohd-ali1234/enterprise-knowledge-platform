"""Dense (semantic) retrieval: embed the query, search the vector store."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.domain.retrieval import ChunkMatch
from app.ports.embedder import Embedder
from app.ports.vector_store import VectorStore

logger = get_logger(__name__)


class DenseRetriever:
    """Nearest-neighbour search over chunk embeddings."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self._embedder = embedder
        self._store = store

    def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        embedding = self._embedder.embed_query(query)
        matches = self._store.search(embedding, limit, filters)
        logger.debug("Dense search returned %d matches", len(matches))
        return matches
