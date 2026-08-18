"""Port: text -> vector."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from app.domain.retrieval import Vector


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors.

    Queries and documents are embedded through separate methods because
    asymmetric models (instruction-tuned or prefix-based) need different
    treatment for each side.
    """

    def embed_query(self, text: str) -> Vector:
        """Embed a single search query."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of document chunks."""
        ...
