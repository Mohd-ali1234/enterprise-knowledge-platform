"""Port: persistent vector storage and similarity search."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from app.domain.retrieval import ChunkMatch, StoredChunk, Vector, VectorRecord


@runtime_checkable
class VectorStore(Protocol):
    """Stores embedded chunks and answers nearest-neighbour queries."""

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records, keyed by `VectorRecord.id`."""
        ...

    def search(
        self,
        embedding: Vector,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        """Return the `limit` nearest chunks, optionally metadata-filtered."""
        ...

    def list_all(self, filters: dict[str, Any] | None = None) -> list[StoredChunk]:
        """Return every stored chunk. Used to build the sparse index."""
        ...

    def count(self) -> int:
        """Return the number of stored chunks."""
        ...

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk belonging to `document_id`, returning how many."""
        ...

    def reset(self) -> None:
        """Delete every stored chunk."""
        ...
