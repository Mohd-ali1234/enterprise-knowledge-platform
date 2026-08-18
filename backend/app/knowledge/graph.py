"""Stage: Knowledge Graph Construction.

Assembles the per-document view that `ProcessedDocument` carries and the API
reports counts from. The *persistent* graph is a separate concern, owned by
`app.repositories.sqlite_graph`; this module stays a pure in-memory assembly
step with no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.logging import get_logger
from app.domain.documents import Entity, KnowledgeGraph, Relationship
from app.domain.knowledge import DocumentKnowledge, Polarity

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """Assembles extracted entities and relationships into a graph.

    Relationships whose endpoints were never recognised as entities are still
    kept, but their endpoints are promoted to nodes so the graph stays
    connected and has no dangling edges.
    """

    def build(
        self,
        entities: Sequence[Entity],
        relationships: Sequence[Relationship],
    ) -> KnowledgeGraph:
        nodes: dict[str, Entity] = {entity.text: entity for entity in entities}

        for relationship in relationships:
            for endpoint in (relationship.source, relationship.target):
                nodes.setdefault(endpoint, Entity(text=endpoint, label="UNKNOWN"))

        graph = KnowledgeGraph(
            entities=list(nodes.values()),
            relationships=list(relationships),
        )
        logger.debug(
            "Built knowledge graph with %d nodes and %d edges",
            graph.node_count,
            graph.edge_count,
        )
        return graph

    def from_knowledge(self, knowledge: DocumentKnowledge) -> KnowledgeGraph:
        """Project typed extraction onto the document-level graph.

        Negative relations are excluded from the counts: "John does not report
        to Mary" is recorded in the persistent graph for the record, but it is
        not an edge this document asserts.
        """
        return self.build(
            entities=[
                Entity(text=entity.text, label=entity.type.value)
                for entity in knowledge.entities
            ],
            relationships=[
                Relationship(
                    source=relation.source.text,
                    relation=relation.type.value,
                    target=relation.target.text,
                )
                for relation in knowledge.relations
                if relation.polarity is Polarity.POSITIVE
            ],
        )
