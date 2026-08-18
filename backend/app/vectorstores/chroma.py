"""Adapter: `VectorStore` backed by a persistent ChromaDB collection."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.config import settings
from app.core.exceptions import IndexingError, RetrievalError
from app.core.logging import get_logger
from app.domain.retrieval import ChunkMatch, MatchSource, ScoreKind, StoredChunk, Vector, VectorRecord
from app.retrieval.filters import to_chroma_where

logger = get_logger(__name__)


def _similarity(distance: float | None) -> float:
    """Convert a Chroma distance into a 0-1 similarity score."""
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + max(distance, 0.0))


class ChromaVectorStore:
    """Persistent vector storage and dense search over document chunks."""

    def __init__(self, path: str | None = None, collection_name: str | None = None):
        self.path = path or settings.chroma_path
        self.collection_name = collection_name or settings.chroma_collection
        self._collection = None

    @property
    def collection(self):
        """Connect lazily so importing this module never touches the disk."""
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise RetrievalError(f"chromadb is not installed: {exc}") from exc

            logger.info(
                "Opening Chroma collection '%s' at %s", self.collection_name, self.path
            )
            client = chromadb.PersistentClient(path=self.path)
            self._collection = client.get_or_create_collection(name=self.collection_name)

        return self._collection

    # ── Writing ──────────────────────────────────────────────────

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return

        try:
            # upsert (not add) so re-indexing a document replaces its chunks
            # instead of duplicating them.
            self.collection.upsert(
                ids=[record.id for record in records],
                embeddings=[record.embedding for record in records],
                documents=[record.text for record in records],
                metadatas=[record.metadata for record in records],
            )
        except Exception as exc:  # noqa: BLE001 - chromadb raises many error types
            raise IndexingError(f"Writing to the vector store failed: {exc}") from exc

        logger.debug("Upserted %d records into '%s'", len(records), self.collection_name)

    # ── Reading ──────────────────────────────────────────────────

    def search(
        self,
        embedding: Vector,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkMatch]:
        query: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": max(limit, 1),
            "include": ["documents", "metadatas", "distances"],
        }
        where = to_chroma_where(filters)
        if where:
            query["where"] = where

        try:
            response = self.collection.query(**query)
        except Exception as exc:  # noqa: BLE001 - chromadb raises many error types
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        return self._to_matches(response)

    def list_all(self, filters: dict[str, Any] | None = None) -> list[StoredChunk]:
        query: dict[str, Any] = {"include": ["documents", "metadatas"]}
        where = to_chroma_where(filters)
        if where:
            query["where"] = where

        try:
            response = self.collection.get(**query)
        except Exception as exc:  # noqa: BLE001 - chromadb raises many error types
            raise RetrievalError(f"Reading the vector store failed: {exc}") from exc

        ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []

        return [
            StoredChunk(
                id=chunk_id,
                text=documents[index] if index < len(documents) else "",
                metadata=metadatas[index] if index < len(metadatas) else {},
            )
            for index, chunk_id in enumerate(ids)
        ]

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as exc:  # noqa: BLE001 - chromadb raises many error types
            raise RetrievalError(f"Counting the vector store failed: {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk of one document, returning how many were removed.

        The ids are read first so the count is exact: Chroma's delete reports
        nothing, and a caller needs to know whether anything actually matched.
        """
        try:
            existing = self.collection.get(
                where={"document_id": document_id}, include=[]
            )
            ids = existing.get("ids") or []
            if ids:
                self.collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001 - chromadb raises many error types
            raise IndexingError(
                f"Deleting document '{document_id}' from the vector store failed: {exc}"
            ) from exc

        logger.info("Deleted %d chunk(s) for document %s", len(ids), document_id)
        return len(ids)

    def reset(self) -> None:
        """Drop and recreate the collection."""
        import chromadb

        client = chromadb.PersistentClient(path=self.path)
        try:
            client.delete_collection(name=self.collection_name)
        except Exception as exc:  # noqa: BLE001 - absent collection is not an error
            logger.warning("Could not delete collection '%s': %s", self.collection_name, exc)

        self._collection = client.get_or_create_collection(name=self.collection_name)
        logger.info("Reset collection '%s'", self.collection_name)

    # ── Response mapping ─────────────────────────────────────────

    @staticmethod
    def _to_matches(response: dict[str, Any]) -> list[ChunkMatch]:
        """Flatten Chroma's nested per-query response into `ChunkMatch`es."""

        def first(key: str) -> list[Any]:
            values = response.get(key) or [[]]
            return values[0] if values else []

        ids = first("ids")
        documents = first("documents")
        metadatas = first("metadatas")
        distances = first("distances")

        matches: list[ChunkMatch] = []
        for index, chunk_id in enumerate(ids):
            score = _similarity(distances[index] if index < len(distances) else None)
            matches.append(
                ChunkMatch(
                    id=chunk_id,
                    text=documents[index] if index < len(documents) else "",
                    metadata=metadatas[index] if index < len(metadatas) else {},
                    score=score,
                    source=MatchSource.SEMANTIC,
                    scores={ScoreKind.SEMANTIC.value: score},
                )
            )
        return matches
