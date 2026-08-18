"""Orchestrator: the document ingestion pipeline.

    Document
      -> Text Extraction
      -> Cleaning
      -> Metadata
      -> Structural Chunking
      -> Entity Extraction
      -> Relationship Extraction
      -> Knowledge Graph
      -> Recursive Chunking
      -> Chunk Enrichment
      -> ProcessedDocument

This class only sequences the stages and hands each one's output to the next.
Every decision about *how* a stage works lives in that stage's module.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger
from app.domain.documents import ProcessedDocument
from app.domain.knowledge import DocumentKnowledge
from app.ingestion.cleaning import TextCleaner
from app.ingestion.enrichment import ChunkEnricher
from app.ingestion.extraction import ExtractorRegistry
from app.ingestion.metadata import DocumentMetadataBuilder
from app.ingestion.recursive_chunking import RecursiveChunker
from app.ingestion.structural_chunking import StructuralChunker
from app.knowledge.entities import EntityExtractor, SpacyEntityExtractor
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.graph import KnowledgeGraphBuilder

logger = get_logger(__name__)


class DocumentIngestionPipeline:
    """Runs a source file through every ingestion stage."""

    def __init__(
        self,
        extractors: ExtractorRegistry | None = None,
        cleaner: TextCleaner | None = None,
        metadata_builder: DocumentMetadataBuilder | None = None,
        structural_chunker: StructuralChunker | None = None,
        recursive_chunker: RecursiveChunker | None = None,
        entity_extractor: EntityExtractor | None = None,
        knowledge_extractor: KnowledgeExtractor | None = None,
        graph_builder: KnowledgeGraphBuilder | None = None,
        enricher: ChunkEnricher | None = None,
    ):
        self.extractors = extractors or ExtractorRegistry()
        self.cleaner = cleaner or TextCleaner()
        self.metadata_builder = metadata_builder or DocumentMetadataBuilder()
        self.structural_chunker = structural_chunker or StructuralChunker()
        self.recursive_chunker = recursive_chunker or RecursiveChunker()
        self.entity_extractor = entity_extractor or SpacyEntityExtractor()
        self.knowledge_extractor = knowledge_extractor or KnowledgeExtractor()
        self.graph_builder = graph_builder or KnowledgeGraphBuilder()
        self.enricher = enricher or ChunkEnricher(self.entity_extractor)

    def supports(self, file_path: Path | str) -> bool:
        """Whether an extractor is registered for this file type."""
        return self.extractors.supports(file_path)

    def _extract_knowledge(self, text: str, name: str) -> DocumentKnowledge:
        """Run graph extraction, swallowing any failure.

        The knowledge graph is an enhancement. A parser error, an unexpected
        input or a missing spaCy model must not stop a document from being
        chunked, embedded and made searchable.
        """
        try:
            return self.knowledge_extractor.extract(text)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            logger.warning("Knowledge extraction failed for %s: %s", name, exc)
            return DocumentKnowledge()

    def run(self, file_path: Path | str, display_name: str | None = None) -> ProcessedDocument:
        """Process one file into a `ProcessedDocument`.

        Args:
            file_path: Location of the file on disk.
            display_name: Name to record in metadata. Defaults to the on-disk
                name; uploads pass the original client-supplied name here.

        Raises:
            DocumentProcessingError: if the file is missing, unsupported, or
                yields no usable text.
        """
        path = Path(file_path)
        started = time.perf_counter()
        logger.info("Ingesting %s", path.name)

        if not path.is_file():
            raise DocumentProcessingError(f"File not found: {path.name}")

        # 1. Text Extraction
        extracted = self.extractors.extract(path)

        # 2. Cleaning
        text = self.cleaner.clean(extracted.text)
        if not text:
            raise DocumentProcessingError(
                f"No extractable text in '{path.name}'. "
                "The file may be empty or a scanned image requiring OCR."
            )

        # 3. Metadata
        metadata = self.metadata_builder.build(
            path, extracted.page_count, text, display_name=display_name
        )

        # 4. Structural Chunking
        sections = self.structural_chunker.split(text)

        # 5-7. Sentence segmentation -> Entities -> Relationships -> Graph.
        #      Graph extraction is best-effort: a failure here costs the
        #      document its edges, never its ingestion.
        knowledge = self._extract_knowledge(text, path.name)
        knowledge_graph = self.graph_builder.from_knowledge(knowledge)

        # 8. Recursive Chunking
        chunks = self.recursive_chunker.chunk(sections)

        # 9. Chunk Enrichment
        chunks = self.enricher.enrich(chunks, metadata.document_id)

        document = ProcessedDocument(
            metadata=metadata,
            knowledge_graph=knowledge_graph,
            chunks=chunks,
        )
        # Carried alongside the document so `IngestionService` can persist the
        # graph without re-parsing the text.
        document.extracted_knowledge = knowledge

        logger.info(
            "Ingested %s in %.2fs: %d pages, %d sections, %d chunks, %d entities",
            path.name,
            time.perf_counter() - started,
            metadata.pages,
            len(sections),
            len(chunks),
            knowledge_graph.node_count,
        )
        return document
