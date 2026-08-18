"""Adapter: `Embedder` backed by sentence-transformers."""

from __future__ import annotations

from typing import Sequence

from app.core.config import settings
from app.core.exceptions import IndexingError
from app.core.logging import get_logger
from app.domain.retrieval import Vector

logger = get_logger(__name__)


class SentenceTransformerEmbedder:
    """Dense embeddings from a local sentence-transformers model.

    Models are cached per model name for the lifetime of the process; loading
    one costs seconds and hundreds of megabytes, so it must not happen per
    request.
    """

    _models: dict[str, object] = {}

    def __init__(self, model_name: str | None = None, normalize: bool | None = None):
        self.model_name = model_name or settings.embedding_model
        self.normalize = settings.embedding_normalize if normalize is None else normalize

    @property
    def model(self):
        """Load the model on first use rather than at import time."""
        if self.model_name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise IndexingError(f"sentence-transformers is not installed: {exc}") from exc

            logger.info("Loading embedding model '%s'", self.model_name)
            self._models[self.model_name] = SentenceTransformer(self.model_name)

        return self._models[self.model_name]

    def embed_query(self, text: str) -> Vector:
        return self._encode([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        return self._encode(list(texts))

    def _encode(self, texts: list[str]) -> list[Vector]:
        try:
            return self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
            ).tolist()
        except Exception as exc:  # noqa: BLE001 - surfaces as an indexing failure
            raise IndexingError(f"Embedding failed: {exc}") from exc
