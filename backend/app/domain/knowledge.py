"""Knowledge-graph domain models.

These describe the persisted graph, as opposed to `app.domain.documents`,
whose `Entity`/`Relationship` are the raw per-document extraction output.

    Sentence -> ExtractedEntity -> GraphEntity   (node, deduplicated)
             -> ExtractedRelation -> GraphRelationship (edge, with evidence)

Every edge keeps its evidence: the sentence it came from, the chunk, and the
document. That is what makes the graph explainable rather than merely present.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Normalised entity types.

    spaCy's NER labels are mapped onto these so the UI has a small, stable
    vocabulary to filter and colour by.
    """

    PERSON = "PERSON"
    ORG = "ORG"
    LOCATION = "LOCATION"
    DATE = "DATE"
    PRODUCT = "PRODUCT"
    ROLE = "ROLE"
    DEPARTMENT = "DEPARTMENT"
    PROJECT = "PROJECT"
    TECHNOLOGY = "TECHNOLOGY"
    EVENT = "EVENT"
    MONEY = "MONEY"
    OTHER = "OTHER"


class RelationType(str, Enum):
    """Normalised relationship taxonomy.

    `RELATED_TO` is the deliberate escape hatch: a relation that is real but
    does not map onto a known type keeps its original wording in
    `GraphRelationship.predicate` rather than being forced or discarded.
    """

    WORKS_IN = "WORKS_IN"
    WORKS_FOR = "WORKS_FOR"
    REPORTS_TO = "REPORTS_TO"
    MANAGES = "MANAGES"
    MANAGED_BY = "MANAGED_BY"
    OWNS = "OWNS"
    OWNED_BY = "OWNED_BY"
    PART_OF = "PART_OF"
    BELONGS_TO = "BELONGS_TO"
    RESPONSIBLE_FOR = "RESPONSIBLE_FOR"
    LOCATED_IN = "LOCATED_IN"
    USES = "USES"
    DEVELOPS = "DEVELOPS"
    PROVIDES = "PROVIDES"
    ACQUIRED = "ACQUIRED"
    FOUNDED = "FOUNDED"
    CREATED = "CREATED"
    LEADS = "LEADS"
    MEMBER_OF = "MEMBER_OF"
    RELATED_TO = "RELATED_TO"


class Polarity(str, Enum):
    """Whether a sentence asserted or denied the relation.

    "John does not report to Mary" is information, not noise - it is stored
    with `NEGATIVE` polarity and hidden from the default graph read.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ExtractedEntity(BaseModel):
    """One entity mention found in a sentence, before graph deduplication."""

    text: str
    """The surface form exactly as written."""
    canonical_name: str
    """Normalised key used to merge mentions of the same thing."""
    type: EntityType
    start: int = 0
    end: int = 0
    confidence: float = 1.0


class ExtractedRelation(BaseModel):
    """One relation found in a sentence, before graph deduplication."""

    source: ExtractedEntity
    target: ExtractedEntity
    type: RelationType
    predicate: str
    """The verb phrase as written, e.g. "collaborates with"."""
    confidence: float
    polarity: Polarity = Polarity.POSITIVE
    sentence: str = ""
    sentence_start: int = 0
    """Character offset of the sentence in the cleaned document text, used to
    attribute the relation to the chunk it came from."""


class DocumentKnowledge(BaseModel):
    """Everything one document contributed to the graph."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relations


class GraphEntity(BaseModel):
    """A node in the persisted graph."""

    id: str
    canonical_name: str
    display_name: str
    type: EntityType
    description: str = ""
    mentions_count: int = 0
    confidence: float = 1.0
    source_documents: list[str] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RelationshipEvidence(BaseModel):
    """Where one occurrence of a relationship was observed."""

    document_id: str = ""
    chunk_id: str = ""
    sentence: str = ""
    confidence: float = 0.0


class GraphRelationship(BaseModel):
    """An edge in the persisted graph.

    The same claim seen in three documents is one edge with three pieces of
    evidence, not three edges.
    """

    id: str
    source_id: str
    target_id: str
    type: RelationType
    predicate: str = ""
    confidence: float = 0.0
    polarity: Polarity = Polarity.POSITIVE
    evidence: list[RelationshipEvidence] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)


class GraphView(BaseModel):
    """A bounded slice of the graph, which is what every API read returns."""

    entities: list[GraphEntity] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
    truncated: bool = False
    """True when a limit cut the result, so the UI can say so."""


class GraphStats(BaseModel):
    """Counts describing the whole graph."""

    total_entities: int = 0
    total_relationships: int = 0
    entities_by_type: dict[str, int] = Field(default_factory=dict)
    relationships_by_type: dict[str, int] = Field(default_factory=dict)
    documents_with_graph_data: int = 0


class EntityDetail(BaseModel):
    """One entity plus everything it connects to."""

    entity: GraphEntity
    relationships: list[GraphRelationship] = Field(default_factory=list)
    neighbors: list[GraphEntity] = Field(default_factory=list)
