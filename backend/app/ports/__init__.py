"""Ports: the abstract interfaces the pipeline depends on.

Every stage depends on a Protocol declared here rather than on a concrete
implementation, so an adapter can be swapped (a different vector store, a
cross-encoder reranker, an LLM answer generator) without touching pipeline
code. Protocols are structural, so adapters need no base class.
"""

from app.ports.answer_generator import AnswerGenerator
from app.ports.document_repository import DocumentRepository
from app.ports.embedder import Embedder
from app.ports.query_rewriter import QueryRewriter
from app.ports.reranker import Reranker
from app.ports.text_extractor import TextExtractor
from app.ports.vector_store import VectorStore

__all__ = [
    "AnswerGenerator",
    "DocumentRepository",
    "Embedder",
    "QueryRewriter",
    "Reranker",
    "TextExtractor",
    "VectorStore",
]
