"""Retrieval-side domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

Vector = list[float]


class RetrievalMode(str, Enum):
    """How the retriever should search."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class MatchSource(str, Enum):
    """Which retrieval strategy surfaced a chunk."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class ScoreKind(str, Enum):
    """Named slots in `ChunkMatch.scores`, one per scoring stage."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    FUSION = "fusion"
    RERANK = "rerank"


class VectorRecord(BaseModel):
    """A chunk prepared for writing into the vector store."""

    id: str
    text: str
    embedding: Vector
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredChunk(BaseModel):
    """A chunk read back out of the vector store (no scores attached)."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMatch(BaseModel):
    """A retrieved chunk plus its relevance scores.

    `score` is always the relevance from the most recent stage, so downstream
    consumers never have to guess between `score` / `rrf_score` / `rerank_score`.
    `scores` keeps the per-stage history for debugging and observability.
    """

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    source: MatchSource = MatchSource.SEMANTIC
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def document_id(self) -> str | None:
        return self.metadata.get("document_id")

    @property
    def section_title(self) -> str | None:
        return self.metadata.get("section_title") or None

    def with_score(self, kind: ScoreKind, value: float) -> "ChunkMatch":
        """Return a copy carrying an additional stage score as the new `score`.

        Matches are treated as immutable so a stage can never corrupt the input
        another stage still holds a reference to.
        """
        updated = self.model_copy(deep=True)
        updated.scores[kind.value] = value
        updated.score = value
        return updated
