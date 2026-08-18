"""Port: reorder retrieved chunks by relevance to the query."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from app.domain.retrieval import ChunkMatch


@runtime_checkable
class Reranker(Protocol):
    """Rescores candidate chunks and returns the best `limit` of them.

    Implementations must return matches carrying a `ScoreKind.RERANK` score.
    """

    def rerank(
        self,
        query: str,
        matches: Sequence[ChunkMatch],
        limit: int,
    ) -> list[ChunkMatch]:
        ...
