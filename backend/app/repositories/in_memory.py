"""Adapter: in-memory `DocumentRepository`.

Bridges the two-step upload -> index flow. Contents are lost on restart; see
the README for the database-backed replacement path.
"""

from __future__ import annotations

from threading import RLock

from app.core.logging import get_logger
from app.domain.documents import ProcessedDocument

logger = get_logger(__name__)


class InMemoryDocumentRepository:
    """Process-local store of processed documents.

    Guarded by a lock because uvicorn serves requests from a thread pool and
    ingestion runs off the event loop.
    """

    def __init__(self) -> None:
        self._documents: dict[str, ProcessedDocument] = {}
        self._lock = RLock()

    def save(self, document: ProcessedDocument) -> None:
        with self._lock:
            self._documents[document.document_id] = document
        logger.debug("Stored document %s", document.document_id)

    def get(self, document_id: str) -> ProcessedDocument | None:
        with self._lock:
            return self._documents.get(document_id)

    def delete(self, document_id: str) -> None:
        with self._lock:
            self._documents.pop(document_id, None)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._documents)
