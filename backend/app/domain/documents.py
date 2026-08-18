"""Document-side domain models.

These are the objects that travel through the ingestion pipeline:

    Document -> ExtractedText -> DocumentSection -> DocumentChunk
                              -> Entity / Relationship -> KnowledgeGraph
                              -> ProcessedDocument
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ExtractedText(BaseModel):
    """Raw text pulled out of a source file, before cleaning."""

    text: str
    page_count: int


class Entity(BaseModel):
    """A named entity recognised in the document text."""

    text: str
    label: str


class Relationship(BaseModel):
    """A directed edge between two entities."""

    source: str
    relation: str
    target: str


class KnowledgeGraph(BaseModel):
    """Entities and relationships assembled into a graph.

    This is the single owner of the extracted knowledge; `ProcessedDocument`
    reads through to it rather than keeping duplicate copies.
    """

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.entities)

    @property
    def edge_count(self) -> int:
        return len(self.relationships)


class DocumentSection(BaseModel):
    """A logical section of a document, produced by structural chunking."""

    section_id: int
    title: str
    content: str
    start: int = 0
    """Character offset of `content` within the cleaned document text."""


class DocumentChunk(BaseModel):
    """A retrievable unit of text produced by recursive chunking."""

    chunk_id: int
    document_id: str | None = None
    section_id: int | None = None
    section_title: str | None = None
    text: str
    start: int
    end: int
    entities: list[str] = Field(default_factory=list)

    @property
    def vector_id(self) -> str:
        """Stable identifier used as the vector-store primary key."""
        return f"{self.document_id}_{self.chunk_id}"


class DocumentMetadata(BaseModel):
    """Descriptive metadata about an ingested file."""

    document_id: str
    filename: str
    file_type: str
    file_size: int
    pages: int
    char_count: int
    word_count: int
    uploaded_at: datetime


class IndexedDocument(BaseModel):
    """A document summarised from what is actually in the vector store.

    Unlike `ProcessedDocument`, this survives a restart: it is rebuilt from
    chunk metadata rather than held in memory. `filename` is empty for chunks
    indexed before filenames were recorded.
    """

    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    indexed_at: str = ""

    @property
    def display_name(self) -> str:
        return self.filename or self.document_id


class ProcessedDocument(BaseModel):
    """The complete output of the ingestion pipeline for one document."""

    metadata: DocumentMetadata
    knowledge_graph: KnowledgeGraph
    chunks: list[DocumentChunk] = Field(default_factory=list)
    extracted_knowledge: object | None = None
    """Typed graph extraction (`app.domain.knowledge.DocumentKnowledge`), carried
    to the persistence step. Untyped here to keep this module free of a
    dependency on the knowledge domain."""

    @property
    def document_id(self) -> str:
        return self.metadata.document_id

    @property
    def entities(self) -> list[Entity]:
        return self.knowledge_graph.entities

    @property
    def relationships(self) -> list[Relationship]:
        return self.knowledge_graph.relationships
