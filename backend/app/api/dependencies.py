"""Composition root.

Every adapter is constructed exactly once here and injected downwards, so
concrete implementations are chosen in a single file. Swapping a component -
a different vector store, an LLM answer generator, a cross-encoder reranker -
is an edit to this module and nothing else.
"""

from __future__ import annotations

from functools import lru_cache

from app.application.file_storage import UploadStorage
from app.application.graph_service import GraphService
from app.application.ingestion_service import IngestionService
from app.application.knowledge_service import KnowledgeService
from app.core.logging import get_logger
from app.embeddings.sentence_transformer import SentenceTransformerEmbedder
from app.indexing.indexer import DocumentIndexer
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.llm.gemini import GeminiClient
from app.query.context import ContextBuilder
from app.query.generation import (
    AnswerGeneratorRegistry,
    ExtractiveAnswerGenerator,
    GeminiAnswerGenerator,
    ResilientAnswerGenerator,
)
from app.query.pipeline import QueryPipeline
from app.query.reranking import LexicalReranker
from app.query.rewriting import (
    GeminiQueryRewriter,
    HeuristicQueryRewriter,
    ResilientQueryRewriter,
)
from app.query.routing import AnswerRouter
from app.query.understanding import QueryUnderstandingService
from app.repositories.in_memory import InMemoryDocumentRepository
from app.repositories.sqlite_graph import SqliteGraphRepository
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever
from app.vectorstores.chroma import ChromaVectorStore

logger = get_logger(__name__)


# ── Infrastructure ───────────────────────────────────────────────


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@lru_cache
def get_document_repository() -> InMemoryDocumentRepository:
    return InMemoryDocumentRepository()


@lru_cache
def get_graph_repository() -> SqliteGraphRepository:
    """The persistent knowledge graph.

    SQLite rather than the vector store: a graph needs joins, indexes and
    neighbourhood traversal, none of which Chroma can answer.
    """
    return SqliteGraphRepository()


@lru_cache
def get_graph_service() -> GraphService:
    return GraphService(get_graph_repository())


@lru_cache
def get_upload_storage() -> UploadStorage:
    return UploadStorage()


# ── Retrieval ────────────────────────────────────────────────────


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense=DenseRetriever(get_embedder(), get_vector_store()),
        sparse=SparseRetriever(get_vector_store()),
    )


@lru_cache
def get_gemini_client() -> GeminiClient:
    """One shared Gemini connection pool for every stage that needs an LLM."""
    client = GeminiClient()
    if not client.is_configured:
        logger.warning(
            "GEMINI_API_KEY is not set - query rewriting and answer generation "
            "will use their offline fallbacks."
        )
    return client


@lru_cache
def get_query_rewriter() -> ResilientQueryRewriter:
    """Gemini rewriting with an always-available heuristic fallback.

    `OllamaQueryRewriter` remains in `app.query.rewriting` and can be swapped
    back in here as the primary without touching anything else.
    """
    return ResilientQueryRewriter(
        primary=GeminiQueryRewriter(get_gemini_client()),
        fallback=HeuristicQueryRewriter(),
    )


@lru_cache
def get_answer_generators() -> AnswerGeneratorRegistry:
    """Route -> generator mapping.

    Gemini is registered as the default, so every route - `direct` included -
    is answered by the LLM, with extraction as the fallback if a call fails.
    To keep high-confidence factual queries on the cheap extractive path
    instead, pass `default=extractive` and register Gemini per route:

        registry = AnswerGeneratorRegistry(default=extractive)
        registry.register(AnswerRoute.LOCAL_LLM, gemini)
        registry.register(AnswerRoute.ONLINE_LLM, gemini)

    With no API key the registry is extractive-only, which is exactly the
    behaviour this platform had before Gemini was wired in.
    """
    extractive = ExtractiveAnswerGenerator()
    client = get_gemini_client()

    if not client.is_configured:
        return AnswerGeneratorRegistry(default=extractive)

    return AnswerGeneratorRegistry(
        default=ResilientAnswerGenerator(
            primary=GeminiAnswerGenerator(client),
            fallback=extractive,
        )
    )


# ── Pipelines ────────────────────────────────────────────────────


@lru_cache
def get_ingestion_pipeline() -> DocumentIngestionPipeline:
    return DocumentIngestionPipeline()


@lru_cache
def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(
        embedder=get_embedder(),
        store=get_vector_store(),
        # Keep the keyword index in step with newly indexed documents.
        on_index=get_retriever().invalidate_keyword_index,
    )


@lru_cache
def get_query_pipeline() -> QueryPipeline:
    return QueryPipeline(
        retriever=get_retriever(),
        rewriter=get_query_rewriter(),
        reranker=LexicalReranker(),
        generators=get_answer_generators(),
        understanding_service=QueryUnderstandingService(),
        context_builder=ContextBuilder(),
        router=AnswerRouter(),
    )


# ── Application services (what routers depend on) ────────────────


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        pipeline=get_ingestion_pipeline(),
        indexer=get_indexer(),
        repository=get_document_repository(),
        storage=get_upload_storage(),
        store=get_vector_store(),
        # Deleting chunks must invalidate the cached keyword index too.
        on_index_changed=get_retriever().invalidate_keyword_index,
        graph=get_graph_repository(),
    )


@lru_cache
def get_knowledge_service() -> KnowledgeService:
    return KnowledgeService(pipeline=get_query_pipeline())
