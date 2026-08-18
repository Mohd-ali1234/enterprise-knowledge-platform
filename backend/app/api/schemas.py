"""API request/response models.

These are transport DTOs, deliberately separate from `app.domain`: the wire
format can stay stable while domain models evolve.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.documents import IndexedDocument, ProcessedDocument
from app.domain.knowledge import (
    EntityDetail,
    EntityType,
    GraphEntity,
    GraphView,
    Polarity,
    RelationType,
)
from app.domain.query import (
    AnswerRoute,
    Citation,
    QueryIntent,
    QueryResult,
    RewriteStrategy,
)
from app.domain.retrieval import ChunkMatch, RetrievalMode

_QUERY_EXAMPLE = "What is the expense reimbursement policy for international travel?"


# ── Documents ────────────────────────────────────────────────────


class DocumentSummary(BaseModel):
    """Counts describing a processed document."""

    document_id: str
    file_name: str
    total_pages: int
    total_chunks: int
    total_entities: int
    total_relationships: int

    @classmethod
    def from_document(cls, document: ProcessedDocument) -> "DocumentSummary":
        """Build the shared summary fields; subclasses add their own."""
        return DocumentSummary(
            document_id=document.document_id,
            file_name=document.metadata.filename,
            total_pages=document.metadata.pages,
            total_chunks=len(document.chunks),
            total_entities=document.knowledge_graph.node_count,
            total_relationships=document.knowledge_graph.edge_count,
        )


class UploadResponse(DocumentSummary):
    message: str


class IngestUrlRequest(BaseModel):
    """A web page to fetch and ingest."""

    url: str = Field(
        min_length=1,
        description="An http or https URL. The page is fetched server-side and "
        "ingested as an HTML document.",
        examples=["https://example.com/handbook"],
    )


class IndexedDocumentResponse(BaseModel):
    """One document currently present in the vector store."""

    document_id: str
    file_name: str
    file_type: str
    chunk_count: int
    indexed_at: str

    @classmethod
    def from_document(cls, document: IndexedDocument) -> "IndexedDocumentResponse":
        return cls(
            document_id=document.document_id,
            # Falls back to the id for documents indexed before filenames
            # were recorded, so the list is never blank.
            file_name=document.display_name,
            file_type=document.file_type,
            chunk_count=document.chunk_count,
            indexed_at=document.indexed_at,
        )


class DocumentListResponse(BaseModel):
    """Everything currently indexed and searchable."""

    documents: list[IndexedDocumentResponse]
    total_documents: int
    total_chunks: int


class DeleteDocumentsRequest(BaseModel):
    """Documents to remove from the index."""

    document_ids: list[str] = Field(
        min_length=1,
        description="Ids to delete. Unknown ids are reported as 0 chunks deleted.",
    )


class DeleteDocumentsResponse(BaseModel):
    deleted: dict[str, int]
    """Chunks removed per document id."""
    documents_deleted: int
    """How many ids actually matched something."""
    chunks_deleted: int
    message: str


class SupportedFormatsResponse(BaseModel):
    """File extensions the ingestion layer can currently accept."""

    extensions: list[str]
    url_ingestion: bool = True


class IndexResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    message: str


class ProcessAndIndexResponse(DocumentSummary):
    chunks_indexed: int
    message: str


class DocumentDetailResponse(BaseModel):
    """Full metadata for one processed document."""

    document_id: str
    filename: str
    file_type: str
    file_size: int
    pages: int
    char_count: int
    word_count: int
    total_chunks: int
    total_entities: int
    total_relationships: int

    @classmethod
    def from_document(cls, document: ProcessedDocument) -> "DocumentDetailResponse":
        metadata = document.metadata
        return cls(
            document_id=metadata.document_id,
            filename=metadata.filename,
            file_type=metadata.file_type,
            file_size=metadata.file_size,
            pages=metadata.pages,
            char_count=metadata.char_count,
            word_count=metadata.word_count,
            total_chunks=len(document.chunks),
            total_entities=document.knowledge_graph.node_count,
            total_relationships=document.knowledge_graph.edge_count,
        )


# ── Retrieval ────────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    mode: RetrievalMode = RetrievalMode.HYBRID
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Exact metadata filters, e.g. {'document_id': '...'}",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": _QUERY_EXAMPLE,
                "top_k": 5,
                "mode": "hybrid",
                "filters": None,
            }
        }
    }


class RetrieveResult(BaseModel):
    """One retrieved chunk, flattened for the client."""

    rank: int
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    source: str
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage scores: semantic, keyword, fusion, rerank.",
    )

    @classmethod
    def from_match(cls, rank: int, match: ChunkMatch) -> "RetrieveResult":
        return cls(
            rank=rank,
            id=match.id,
            text=match.text,
            metadata=match.metadata,
            score=match.score,
            source=match.source.value,
            scores=match.scores,
        )


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrieveResult]


# ── Ask ──────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=100)
    mode: RetrievalMode = RetrievalMode.HYBRID
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Exact metadata filters, e.g. {'document_id': '...'}",
    )
    rewrite: bool = Field(
        default=False,
        description="Expand the query into clearer variants before retrieving.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": _QUERY_EXAMPLE,
                "top_k": 20,
                "rerank_top_k": 5,
                "mode": "hybrid",
                "filters": None,
                "rewrite": False,
            }
        }
    }


class QueryUnderstandingResponse(BaseModel):
    intent: QueryIntent
    entities: list[str]
    language: str
    complexity_score: float
    signals: dict[str, Any]


class RouteDecisionResponse(BaseModel):
    route: AnswerRoute
    reason: str
    confidence: float


class RewrittenQueryResponse(BaseModel):
    query: str
    strategy: RewriteStrategy
    confidence: float


class AskResponse(BaseModel):
    query: str
    retrieval_query: str
    answer: str
    generator: str
    key_points: list[str] = Field(
        default_factory=list,
        description="Takeaways derived from the same context. Empty when the "
        "generator cannot produce them (the extractive fallback).",
    )
    related_questions: list[str] = Field(
        default_factory=list,
        description="Follow-up questions the retrieved context can answer.",
    )
    understanding: QueryUnderstandingResponse
    route: RouteDecisionResponse
    citations: list[Citation]
    results: list[RetrieveResult]
    rewritten_queries: list[RewrittenQueryResponse] = Field(default_factory=list)
    rewrite_error: str | None = None

    @classmethod
    def from_result(cls, result: QueryResult) -> "AskResponse":
        return cls(
            query=result.query,
            retrieval_query=result.retrieval_query,
            answer=result.answer.text,
            generator=result.answer.generator,
            key_points=result.answer.key_points,
            related_questions=result.answer.related_questions,
            understanding=QueryUnderstandingResponse(**result.understanding.model_dump()),
            route=RouteDecisionResponse(**result.route.model_dump()),
            citations=result.context.citations,
            results=[
                RetrieveResult.from_match(rank, match)
                for rank, match in enumerate(result.matches, start=1)
            ],
            rewritten_queries=[
                RewrittenQueryResponse(**rewritten.model_dump())
                for rewritten in result.rewritten_queries
            ],
            rewrite_error=result.rewrite_error,
        )


# ── System ───────────────────────────────────────────────────────


# ── Knowledge graph ──────────────────────────────────────────────


class GraphNode(BaseModel):
    """One entity, shaped for the graph canvas."""

    id: str
    label: str
    canonical_name: str
    type: EntityType
    mentions_count: int
    confidence: float
    degree: int = 0
    """Edges touching this node within the returned slice; drives node size."""
    source_documents: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """One relationship, shaped for the graph canvas."""

    id: str
    source: str
    target: str
    type: RelationType
    predicate: str = ""
    confidence: float
    polarity: Polarity
    evidence_count: int = 0


class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    """True when a limit cut the result, so the UI can say the view is partial."""

    @classmethod
    def from_view(cls, view: GraphView, degrees: dict[str, int]) -> "GraphResponse":
        return cls(
            nodes=[
                GraphNode(
                    id=entity.id,
                    label=entity.display_name,
                    canonical_name=entity.canonical_name,
                    type=entity.type,
                    mentions_count=entity.mentions_count,
                    confidence=entity.confidence,
                    degree=degrees.get(entity.id, 0),
                    source_documents=entity.source_documents,
                )
                for entity in view.entities
            ],
            edges=[
                GraphEdge(
                    id=relationship.id,
                    source=relationship.source_id,
                    target=relationship.target_id,
                    type=relationship.type,
                    predicate=relationship.predicate,
                    confidence=relationship.confidence,
                    polarity=relationship.polarity,
                    evidence_count=relationship.evidence_count,
                )
                for relationship in view.relationships
            ],
            truncated=view.truncated,
        )


class EvidenceResponse(BaseModel):
    """Why a relationship exists: the sentence it was read from."""

    document_id: str
    chunk_id: str
    sentence: str
    confidence: float


class RelationshipDetailResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    source_label: str = ""
    target_label: str = ""
    type: RelationType
    predicate: str = ""
    confidence: float
    polarity: Polarity
    evidence: list[EvidenceResponse] = Field(default_factory=list)


class EntityDetailResponse(BaseModel):
    """An entity, its relationships and its provenance."""

    id: str
    label: str
    canonical_name: str
    type: EntityType
    description: str = ""
    mentions_count: int
    confidence: float
    source_documents: list[str] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    relationships: list[RelationshipDetailResponse] = Field(default_factory=list)
    neighbors: list[GraphNode] = Field(default_factory=list)

    @classmethod
    def from_detail(cls, detail: EntityDetail) -> "EntityDetailResponse":
        labels = {neighbor.id: neighbor.display_name for neighbor in detail.neighbors}
        labels[detail.entity.id] = detail.entity.display_name

        return cls(
            id=detail.entity.id,
            label=detail.entity.display_name,
            canonical_name=detail.entity.canonical_name,
            type=detail.entity.type,
            description=detail.entity.description,
            mentions_count=detail.entity.mentions_count,
            confidence=detail.entity.confidence,
            source_documents=detail.entity.source_documents,
            source_chunks=detail.entity.source_chunks,
            relationships=[
                RelationshipDetailResponse(
                    id=relationship.id,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    source_label=labels.get(relationship.source_id, ""),
                    target_label=labels.get(relationship.target_id, ""),
                    type=relationship.type,
                    predicate=relationship.predicate,
                    confidence=relationship.confidence,
                    polarity=relationship.polarity,
                    evidence=[
                        EvidenceResponse(**item.model_dump())
                        for item in relationship.evidence
                    ],
                )
                for relationship in detail.relationships
            ],
            neighbors=[
                GraphNode(
                    id=neighbor.id,
                    label=neighbor.display_name,
                    canonical_name=neighbor.canonical_name,
                    type=neighbor.type,
                    mentions_count=neighbor.mentions_count,
                    confidence=neighbor.confidence,
                    source_documents=neighbor.source_documents,
                )
                for neighbor in detail.neighbors
            ],
        )


class GraphSearchResponse(BaseModel):
    query: str
    total: int
    entities: list[GraphEntity] = Field(default_factory=list)


class GraphStatsResponse(BaseModel):
    total_entities: int
    total_relationships: int
    entities_by_type: dict[str, int] = Field(default_factory=dict)
    relationships_by_type: dict[str, int] = Field(default_factory=dict)
    documents_with_graph_data: int


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
    error: str
