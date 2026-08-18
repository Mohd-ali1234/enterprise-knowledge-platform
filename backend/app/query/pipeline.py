"""Orchestrator: the query pipeline.

    User Query
      -> Query Understanding
      -> Query Rewriting (optional)
      -> Hybrid Retrieval
      -> Reranking
      -> Context Building
      -> AI Routing
      -> Answer Generation
      -> QueryResult (answer + citations)

Like the ingestion pipeline, this class only sequences stages. Every stage is
injected, so a test can substitute any of them and production can swap an
implementation without editing this file.
"""

from __future__ import annotations

import time
from typing import Any

from anyio import to_thread

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.query import QueryResult, QueryUnderstanding, RewriteResult
from app.domain.retrieval import ChunkMatch, RetrievalMode
from app.ports.answer_generator import AnswerGenerator
from app.ports.query_rewriter import QueryRewriter
from app.ports.reranker import Reranker
from app.query.context import ContextBuilder
from app.query.generation import AnswerGeneratorRegistry
from app.query.routing import AnswerRouter
from app.query.understanding import QueryUnderstandingService
from app.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


class QueryPipeline:
    """Answers a question against the indexed knowledge base."""

    def __init__(
        self,
        retriever: HybridRetriever,
        rewriter: QueryRewriter,
        reranker: Reranker,
        generators: AnswerGeneratorRegistry,
        understanding_service: QueryUnderstandingService | None = None,
        context_builder: ContextBuilder | None = None,
        router: AnswerRouter | None = None,
    ):
        self._retriever = retriever
        self._rewriter = rewriter
        self._reranker = reranker
        self._generators = generators
        self._understanding = understanding_service or QueryUnderstandingService()
        self._context_builder = context_builder or ContextBuilder()
        self._router = router or AnswerRouter()

    async def run(
        self,
        query: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        filters: dict[str, Any] | None = None,
        rewrite: bool = False,
    ) -> QueryResult:
        top_k = top_k or settings.default_top_k
        rerank_top_k = rerank_top_k or settings.default_rerank_top_k
        started = time.perf_counter()

        # 1. Query Understanding
        understanding = self._understanding.understand(query)

        # 2. Query Rewriting (optional)
        rewrite_result = await self._rewrite(query) if rewrite else None
        retrieval_query = self._retrieval_query(query, rewrite_result)

        # 3. Hybrid Retrieval. Embedding and vector search block, so they run
        #    in a worker thread to keep the event loop free.
        retrieved = await to_thread.run_sync(
            lambda: self._retriever.retrieve(
                retrieval_query, limit=top_k, mode=mode, filters=filters
            )
        )

        # 4. Reranking - always against the user's own words, not the rewrite,
        #    so a rewrite can widen recall without distorting final ordering.
        reranked = await to_thread.run_sync(
            lambda: self._reranker.rerank(query, retrieved, rerank_top_k)
        )

        # 5. Context Building
        context = self._context_builder.build(reranked)

        # 6. AI Routing
        route = self._router.choose(understanding, context)

        # 7. Answer Generation
        generator: AnswerGenerator = self._generators.resolve(route.route)
        answer = await generator.generate(query, context)

        logger.info(
            "Answered query in %.2fs (retrieved=%d, reranked=%d, route=%s)",
            time.perf_counter() - started,
            len(retrieved),
            len(reranked),
            route.route.value,
        )

        return QueryResult(
            query=query,
            retrieval_query=retrieval_query,
            understanding=understanding,
            rewritten_queries=rewrite_result.queries if rewrite_result else [],
            rewrite_error=rewrite_result.warning if rewrite_result else None,
            matches=reranked,
            context=context,
            route=route,
            answer=answer,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        """Retrieval only, for callers that want chunks rather than an answer."""
        return self._retriever.retrieve(
            query, limit=top_k or settings.default_top_k, mode=mode, filters=filters
        )

    async def _rewrite(self, query: str) -> RewriteResult:
        return await self._rewriter.rewrite(query)

    @staticmethod
    def _retrieval_query(query: str, rewrite_result: RewriteResult | None) -> str:
        """Search with the best rewrite when one exists, else the original."""
        if rewrite_result is None or rewrite_result.best is None:
            return query
        return rewrite_result.best.query

    @property
    def understanding_service(self) -> QueryUnderstandingService:
        return self._understanding


__all__ = ["QueryPipeline", "QueryUnderstanding"]
