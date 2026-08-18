"""Port: persistence for processed documents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.documents import ProcessedDocument


@runtime_checkable
class DocumentRepository(Protocol):
    """Stores processed documents between the upload and index steps.

    The default adapter is in-memory. Swapping in a database-backed
    implementation only requires satisfying this protocol.
    """

    def save(self, document: ProcessedDocument) -> None:
        ...

    def get(self, document_id: str) -> ProcessedDocument | None:
        ...

    def delete(self, document_id: str) -> None:
        ...

    def list_ids(self) -> list[str]:
        ...
