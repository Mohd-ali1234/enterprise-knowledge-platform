"""Shared fixtures and in-memory test doubles.

The doubles satisfy the same ports as the production adapters, so the pipelines
under test are the real ones - only the infrastructure is substituted.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from app.domain.documents import Entity
from app.domain.retrieval import (
    ChunkMatch,
    MatchSource,
    ScoreKind,
    StoredChunk,
    Vector,
    VectorRecord,
)
from app.retrieval.filters import matches as metadata_matches
from app.retrieval.tokenization import term_set


class FakeEmbedder:
    """Deterministic bag-of-characters embeddings; no model download."""

    dimensions = 16

    def embed_query(self, text: str) -> Vector:
        return self._encode(text)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        for character in text.lower():
            vector[ord(character) % self.dimensions] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


class FakeVectorStore:
    """In-memory `VectorStore` using cosine similarity."""

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self.records[record.id] = record

    def search(
        self,
        embedding: Vector,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        scored = [
            (self._cosine(embedding, record.embedding), record)
            for record in self.records.values()
            if metadata_matches(record.metadata, filters)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            ChunkMatch(
                id=record.id,
                text=record.text,
                metadata=record.metadata,
                score=score,
                source=MatchSource.SEMANTIC,
                scores={ScoreKind.SEMANTIC.value: score},
            )
            for score, record in scored[:limit]
        ]

    def list_all(self, filters: dict[str, Any] | None = None) -> list[StoredChunk]:
        return [
            StoredChunk(id=record.id, text=record.text, metadata=record.metadata)
            for record in self.records.values()
            if metadata_matches(record.metadata, filters)
        ]

    def count(self) -> int:
        return len(self.records)

    def delete_document(self, document_id: str) -> int:
        doomed = [
            record_id
            for record_id, record in self.records.items()
            if record.metadata.get("document_id") == document_id
        ]
        for record_id in doomed:
            del self.records[record_id]
        return len(doomed)

    def reset(self) -> None:
        self.records.clear()

    @staticmethod
    def _cosine(left: Vector, right: Vector) -> float:
        return sum(a * b for a, b in zip(left, right))


class FakeEntityExtractor:
    """Treats capitalised words as entities, with no spaCy dependency."""

    def extract(self, text: str) -> list[Entity]:
        return [
            Entity(text=word, label="TEST")
            for word in dict.fromkeys(w for w in text.split() if w[:1].isupper())
        ]

    def extract_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        return [self.extract(text) for text in texts]


class StubRetriever:
    """Returns a fixed set of matches, recording what it was asked for."""

    def __init__(self, matches: list[ChunkMatch] | None = None):
        self.matches = matches if matches is not None else []
        self.last_query: str | None = None

    def retrieve(self, query, limit, mode=None, filters=None):  # noqa: ANN001
        self.last_query = query
        return self.matches[:limit]

    def invalidate_keyword_index(self) -> None:  # pragma: no cover - not exercised
        pass


def make_match(match_id: str, text: str, score: float, **metadata: Any) -> ChunkMatch:
    """Build a `ChunkMatch` for tests. Extra kwargs become chunk metadata."""
    return ChunkMatch(
        id=match_id,
        text=text,
        metadata={"document_id": "doc-1", **metadata},
        score=score,
        source=MatchSource.SEMANTIC,
        scores={ScoreKind.SEMANTIC.value: score},
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def entity_extractor() -> FakeEntityExtractor:
    return FakeEntityExtractor()


@pytest.fixture
def policy_matches() -> list[ChunkMatch]:
    return [
        make_match(
            "doc-1_0",
            "The leave policy affects payroll by changing salary deductions.",
            0.9,
            chunk_id=0,
            section_title="Leave Policy",
        ),
        make_match(
            "doc-1_1",
            "Payroll calculations include approved leave and unpaid absence.",
            0.7,
            chunk_id=1,
            section_title="Payroll",
        ),
    ]


@pytest.fixture
def sample_text() -> str:
    return (
        "1. INTRODUCTION\n"
        "This handbook describes company policy for all employees.\n"
        "Priya Sharma works in Finance.\n\n"
        "2. LEAVE POLICY\n"
        "Employees accrue paid leave each month. Unused leave expires annually.\n\n"
        "3. EXPENSES\n"
        "Travel expenses are reimbursed within thirty days of submission.\n"
    )


__all__ = [
    "FakeEmbedder",
    "FakeEntityExtractor",
    "FakeVectorStore",
    "StubRetriever",
    "make_match",
    "term_set",
]
