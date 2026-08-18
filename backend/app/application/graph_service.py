"""Use case: reading the knowledge graph.

A thin layer over `GraphRepository` that owns the read policy the API needs -
bounding every read, and enriching a subgraph with the NetworkX-derived
centrality the UI uses to size nodes. The repository stays the source of truth;
NetworkX is only ever used on the bounded slice already loaded into memory.
"""

from __future__ import annotations

import networkx as nx

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.knowledge import EntityDetail, GraphEntity, GraphStats, GraphView
from app.ports.graph_repository import GraphRepository

logger = get_logger(__name__)

_MAX_LIMIT = 500
_MAX_DEPTH = 3


class GraphService:
    """Bounded reads over the persisted knowledge graph."""

    def __init__(self, repository: GraphRepository):
        self._repository = repository

    def graph(
        self,
        document_id: str | None = None,
        entity_type: str | None = None,
        relationship_type: str | None = None,
        search: str | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> GraphView:
        """A filtered slice of the graph, never the whole thing unbounded."""
        return self._repository.graph(
            document_id=document_id,
            entity_type=entity_type,
            relationship_type=relationship_type,
            search=search,
            min_confidence=min_confidence,
            limit=self._bounded(limit),
        )

    def entity(self, entity_id: str) -> EntityDetail | None:
        return self._repository.entity(entity_id)

    def neighbors(self, entity_id: str, depth: int = 1, limit: int | None = None) -> GraphView:
        """Expand around one entity, for click-to-expand in the UI."""
        return self._repository.neighbors(
            entity_id, depth=max(1, min(depth, _MAX_DEPTH)), limit=self._bounded(limit)
        )

    def search(self, query: str, limit: int = 20) -> list[GraphEntity]:
        return self._repository.search(query, limit=max(1, min(limit, 100)))

    def stats(self) -> GraphStats:
        return self._repository.stats()

    def degrees(self, view: GraphView) -> dict[str, int]:
        """Degree per node within a slice, used to size nodes in the UI.

        Built with NetworkX over the already-loaded subgraph - the database
        remains the source of truth, and no full-graph load ever happens here.
        """
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(entity.id for entity in view.entities)
        graph.add_edges_from(
            (relationship.source_id, relationship.target_id)
            for relationship in view.relationships
        )
        return {node: graph.degree(node) for node in graph.nodes}

    @staticmethod
    def _bounded(limit: int | None) -> int:
        if limit is None:
            return settings.knowledge_graph_default_limit
        return max(1, min(limit, _MAX_LIMIT))


__all__ = ["GraphService"]
