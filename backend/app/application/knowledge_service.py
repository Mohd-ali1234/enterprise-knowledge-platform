"""Use case: search and question answering over indexed knowledge."""

from __future__ import annotations

from typing import Any

from anyio import to_thread

from app.core.logging import get_logger
from app.domain.query import QueryResult
from app.domain.retrieval import ChunkMatch, RetrievalMode
from app.query.pipeline import QueryPipeline

logger = get_logger(__name__)


class KnowledgeService:
    """Thin async facade over the query pipeline."""

    def __init__(self, pipeline: QueryPipeline):
        self._pipeline = pipeline

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        """Retrieve relevant chunks without generating an answer."""
        # Embedding and vector search are blocking, so they run off the loop.
        return await to_thread.run_sync(
            lambda: self._pipeline.retrieve(query, top_k=top_k, mode=mode, filters=filters)
        )

    async def ask(
        self,
        query: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        filters: dict[str, Any] | None = None,
        rewrite: bool = False,
    ) -> QueryResult:
        """Run the full query pipeline and return a grounded answer."""
        return await self._pipeline.run(
            query=query,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            mode=mode,
            filters=filters,
            rewrite=rewrite,
        )
