"""Port: persistence for the knowledge graph."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.knowledge import (
    DocumentKnowledge,
    EntityDetail,
    GraphEntity,
    GraphStats,
    GraphView,
)


@runtime_checkable
class GraphRepository(Protocol):
    """Stores entities and relationships, and answers graph reads.

    The default adapter is SQLite. Swapping in Postgres or a graph database
    only requires satisfying this protocol.
    """

    def save_document(
        self,
        document_id: str,
        knowledge: DocumentKnowledge,
        chunk_for_offset=None,
    ) -> tuple[int, int]:
        """Persist one document's contribution. Returns (entities, relationships).

        Args:
            chunk_for_offset: Optional callable mapping a character offset in
                the cleaned text to a chunk id, for evidence traceability.
        """
        ...

    def delete_document(self, document_id: str) -> None:
        """Remove a document's contribution, pruning anything left orphaned."""
        ...

    def graph(
        self,
        document_id: str | None = None,
        entity_type: str | None = None,
        relationship_type: str | None = None,
        search: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 150,
    ) -> GraphView:
        """Return a bounded slice of the graph."""
        ...

    def entity(self, entity_id: str) -> EntityDetail | None:
        """One entity with its relationships and immediate neighbours."""
        ...

    def neighbors(self, entity_id: str, depth: int = 1, limit: int = 150) -> GraphView:
        """The subgraph within `depth` hops of an entity."""
        ...

    def search(self, query: str, limit: int = 20) -> list[GraphEntity]:
        """Entities whose name matches `query`."""
        ...

    def stats(self) -> GraphStats:
        """Counts describing the whole graph."""
        ...
