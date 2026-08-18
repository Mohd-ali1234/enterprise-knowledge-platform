"""Stage: Embedding Generation + Indexing.

Turns a `ProcessedDocument`'s chunks into vector-store records:

    chunks -> metadata projection -> embeddings -> vector store
"""

from __future__ import annotations

import time
from typing import Any, Callable

from app.core.logging import get_logger
from app.domain.documents import DocumentChunk, DocumentMetadata, ProcessedDocument
from app.domain.retrieval import VectorRecord
from app.ports.embedder import Embedder
from app.ports.vector_store import VectorStore

logger = get_logger(__name__)

_NO_SECTION = -1


class DocumentIndexer:
    """Embeds chunks and writes them into the vector store."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        on_index: Callable[[], None] | None = None,
    ):
        """
        Args:
            on_index: Called after a successful write. Used to invalidate the
                keyword index so new documents are searchable immediately.
        """
        self._embedder = embedder
        self._store = store
        self._on_index = on_index

    def index(self, document: ProcessedDocument) -> int:
        """Index every chunk of `document` and return how many were written."""
        if not document.chunks:
            logger.warning("Document %s has no chunks to index", document.document_id)
            return 0

        started = time.perf_counter()
        texts = [chunk.text for chunk in document.chunks]
        embeddings = self._embedder.embed_documents(texts)

        records = [
            VectorRecord(
                id=chunk.vector_id,
                text=chunk.text,
                embedding=embedding,
                metadata=self._metadata_for(chunk, document.metadata),
            )
            for chunk, embedding in zip(document.chunks, embeddings)
        ]

        self._store.upsert(records)

        if self._on_index is not None:
            self._on_index()

        logger.info(
            "Indexed %d chunks for document %s in %.2fs",
            len(records),
            document.document_id,
            time.perf_counter() - started,
        )
        return len(records)

    @staticmethod
    def _metadata_for(chunk: DocumentChunk, document: DocumentMetadata) -> dict[str, Any]:
        """Project a chunk into vector-store metadata.

        Chroma only accepts scalar metadata values, so the entity list is
        flattened and `None`s are replaced with sentinels.

        The document's own filename and type are copied onto every chunk. That
        is deliberate duplication: the vector store is the only thing that
        survives a restart, so without it an indexed document cannot be named
        or listed once the in-memory repository is gone.
        """
        return {
            "document_id": chunk.document_id or "",
            "chunk_id": chunk.chunk_id,
            "section_id": chunk.section_id if chunk.section_id is not None else _NO_SECTION,
            "section_title": chunk.section_title or "",
            "entities": ", ".join(chunk.entities),
            "filename": document.filename,
            "file_type": document.file_type,
            "indexed_at": document.uploaded_at.isoformat(),
        }
