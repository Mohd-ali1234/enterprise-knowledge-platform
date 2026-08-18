"""Stage: Chunk Enrichment.

Attaches the owning document id and the entities mentioned inside each chunk,
so retrieval can filter by document and surface entity context per result.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.documents import DocumentChunk
from app.knowledge.entities import EntityExtractor

logger = get_logger(__name__)


class ChunkEnricher:
    """Stamps chunks with their document id and per-chunk entities."""

    def __init__(self, entity_extractor: EntityExtractor):
        self._entity_extractor = entity_extractor

    def enrich(self, chunks: list[DocumentChunk], document_id: str) -> list[DocumentChunk]:
        if not chunks:
            return []

        # Batched so the NLP model runs once over all chunks rather than once
        # per chunk.
        entities_per_chunk = self._entity_extractor.extract_batch(
            [chunk.text for chunk in chunks]
        )

        for chunk, entities in zip(chunks, entities_per_chunk):
            chunk.document_id = document_id
            chunk.entities = sorted({entity.text for entity in entities})

        logger.debug("Enriched %d chunks for document %s", len(chunks), document_id)
        return chunks
