"""Use case: ingest and index documents.

Composes the ingestion pipeline, the repository and the indexer into the
operations the API exposes. Routers call this; they never touch a pipeline
stage directly.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Callable, Sequence

from anyio import to_thread

from app.application.file_storage import UploadStorage
from app.core.exceptions import DocumentNotFoundError, UnsupportedDocumentError
from app.core.logging import get_logger
from app.domain.documents import IndexedDocument, ProcessedDocument
from app.indexing.indexer import DocumentIndexer
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.ingestion.web import WebPageFetcher
from app.ports.document_repository import DocumentRepository
from app.ports.graph_repository import GraphRepository
from app.ports.vector_store import VectorStore

logger = get_logger(__name__)


def _chunk_locator(document: ProcessedDocument):
    """Map a character offset in the cleaned text to the chunk covering it.

    Chunks already carry `start`/`end` offsets, so a relationship's sentence
    position is enough to say which chunk - and therefore which page - it was
    extracted from.
    """
    spans = sorted(
        ((chunk.start, chunk.end, chunk.vector_id) for chunk in document.chunks),
        key=lambda span: span[0],
    )

    def locate(offset: int) -> str:
        for start, end, chunk_id in spans:
            if start <= offset < end:
                return chunk_id
        return ""

    return locate


class IngestionService:
    """Upload -> process -> index, and lookups over processed documents."""

    def __init__(
        self,
        pipeline: DocumentIngestionPipeline,
        indexer: DocumentIndexer,
        repository: DocumentRepository,
        storage: UploadStorage,
        store: VectorStore,
        fetcher: WebPageFetcher | None = None,
        on_index_changed: Callable[[], None] | None = None,
        graph: GraphRepository | None = None,
    ):
        """
        Args:
            on_index_changed: Called after chunks are removed, so the cached
                keyword index does not keep serving deleted documents.
        """
        self._pipeline = pipeline
        self._indexer = indexer
        self._repository = repository
        self._storage = storage
        self._store = store
        self._fetcher = fetcher or WebPageFetcher()
        self._on_index_changed = on_index_changed
        self._graph = graph

    async def process_upload(self, filename: str, stream: BinaryIO) -> ProcessedDocument:
        """Store an uploaded file, run ingestion, and remember the result."""
        self._require_supported(filename)

        path = self._storage.save(filename, stream)
        # Report the name the user uploaded, not the sanitised name on disk.
        document = await self._process(path, display_name=Path(filename).name)
        self._repository.save(document)
        return document

    async def process_url(self, url: str) -> ProcessedDocument:
        """Fetch a web page and run it through ingestion as an HTML document."""
        page = await self._fetcher.fetch(url)

        path = self._storage.save(page.filename, io.BytesIO(page.content))
        # Record the page's own title rather than the temp filename on disk.
        document = await self._process(path, display_name=page.title)
        self._repository.save(document)
        return document

    async def process_and_index_url(self, url: str) -> tuple[ProcessedDocument, int]:
        """Fetch, process and index a web page in one call."""
        document = await self.process_url(url)
        chunks_indexed = await to_thread.run_sync(self._indexer.index, document)
        return document, chunks_indexed

    async def index_document(self, document_id: str) -> int:
        """Index an already-processed document. Returns chunks written."""
        document = self.get_document(document_id)
        return await to_thread.run_sync(self._indexer.index, document)

    async def process_and_index(
        self,
        filename: str,
        stream: BinaryIO,
    ) -> tuple[ProcessedDocument, int]:
        """Upload, process and index in one call."""
        document = await self.process_upload(filename, stream)
        chunks_indexed = await to_thread.run_sync(self._indexer.index, document)
        return document, chunks_indexed

    def get_document(self, document_id: str) -> ProcessedDocument:
        """Fetch a processed document.

        Raises:
            DocumentNotFoundError: if the id is unknown.
        """
        document = self._repository.get(document_id)
        if document is None:
            raise DocumentNotFoundError(
                f"No processed document found for '{document_id}'. Upload it first."
            )
        return document

    def list_document_ids(self) -> list[str]:
        return self._repository.list_ids()

    def list_indexed_documents(self) -> list[IndexedDocument]:
        """Every document present in the vector store, newest first.

        Rebuilt from chunk metadata rather than the repository, so it reflects
        what is actually searchable and survives a restart. Documents indexed
        before filenames were recorded come back with an empty `filename`.
        """
        documents: dict[str, dict] = {}

        for chunk in self._store.list_all():
            document_id = str(chunk.metadata.get("document_id") or "")
            if not document_id:
                continue

            entry = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "filename": str(chunk.metadata.get("filename") or ""),
                    "file_type": str(chunk.metadata.get("file_type") or ""),
                    "indexed_at": str(chunk.metadata.get("indexed_at") or ""),
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1

        return sorted(
            (IndexedDocument(**entry) for entry in documents.values()),
            key=lambda document: (document.indexed_at, document.display_name),
            reverse=True,
        )

    def delete_documents(self, document_ids: Sequence[str]) -> dict[str, int]:
        """Remove documents from the index and the repository.

        Returns `{document_id: chunks_deleted}`. An id that matched nothing
        maps to 0 rather than raising: deleting an already-deleted document is
        the outcome the caller wanted, not an error.
        """
        deleted: dict[str, int] = {}

        for document_id in document_ids:
            deleted[document_id] = self._store.delete_document(document_id)
            self._repository.delete(document_id)
            self._forget_graph(document_id)

        # Rebuild the keyword index only once, and only if something changed -
        # it is rebuilt from a full scan of the store.
        if any(deleted.values()) and self._on_index_changed is not None:
            self._on_index_changed()

        logger.info(
            "Deleted %d document(s), %d chunk(s)", len(deleted), sum(deleted.values())
        )
        return deleted

    def supported_extensions(self) -> list[str]:
        """Every file extension with a registered extractor, sorted."""
        return sorted(self._pipeline.extractors.supported_extensions)

    async def aclose(self) -> None:
        """Release the web fetcher's connection pool."""
        await self._fetcher.aclose()

    async def _process(self, path: Path, display_name: str) -> ProcessedDocument:
        # Ingestion is CPU-bound (PDF parsing, spaCy, chunking). Running it in a
        # worker thread keeps the event loop responsive to other requests.
        document = await to_thread.run_sync(self._pipeline.run, path, display_name)
        await to_thread.run_sync(self._persist_graph, document)
        return document

    def _persist_graph(self, document: ProcessedDocument) -> None:
        """Write the document's extracted knowledge to the graph store.

        Best-effort by design: the graph is an additional capability, so a
        failure here is logged and the document still ingests, indexes and
        becomes searchable.
        """
        knowledge = getattr(document, "extracted_knowledge", None)
        if self._graph is None or knowledge is None or knowledge.is_empty:
            return

        try:
            self._graph.save_document(
                document.document_id,
                knowledge,
                chunk_for_offset=_chunk_locator(document),
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            logger.warning(
                "Persisting the knowledge graph failed for %s: %s",
                document.document_id,
                exc,
            )

    def _forget_graph(self, document_id: str) -> None:
        if self._graph is None:
            return
        try:
            self._graph.delete_document(document_id)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            logger.warning("Removing graph data for %s failed: %s", document_id, exc)

    def _require_supported(self, filename: str) -> None:
        if not self._pipeline.supports(filename):
            supported = ", ".join(sorted(self._pipeline.extractors.supported_extensions))
            raise UnsupportedDocumentError(
                f"'{Path(filename).suffix or filename}' is not supported. "
                f"Supported types: {supported}."
            )
