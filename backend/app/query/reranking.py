"""Stage: Reranking.

Rescores fused candidates against the original user query. Retrieval optimises
recall; reranking optimises precision over that candidate set.
"""

from __future__ import annotations

from typing import Sequence

from app.core.logging import get_logger
from app.domain.retrieval import ChunkMatch, ScoreKind
from app.retrieval.tokenization import term_set

logger = get_logger(__name__)


class LexicalReranker:
    """Reranks by term overlap between the query and each chunk.

    A cheap, dependency-free default. A cross-encoder reranker can replace it
    by satisfying the same `Reranker` protocol.
    """

    def rerank(
        self,
        query: str,
        matches: Sequence[ChunkMatch],
        limit: int,
    ) -> list[ChunkMatch]:
        if not matches:
            return []

        query_terms = term_set(query)

        reranked = [
            match.with_score(
                ScoreKind.RERANK,
                match.score + self._overlap_ratio(query_terms, match.text),
            )
            for match in matches
        ]
        reranked.sort(key=lambda match: match.score, reverse=True)

        logger.debug("Reranked %d matches down to %d", len(matches), min(limit, len(reranked)))
        return reranked[:limit]

    @staticmethod
    def _overlap_ratio(query_terms: set[str], text: str) -> float:
        """Fraction of the query's terms that appear in the chunk."""
        if not query_terms:
            return 0.0
        return len(query_terms & term_set(text)) / len(query_terms)
