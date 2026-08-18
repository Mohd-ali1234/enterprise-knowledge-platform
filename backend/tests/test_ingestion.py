"""Tests for the document ingestion stages and orchestrator."""

from __future__ import annotations

import pytest

from app.core.exceptions import DocumentProcessingError, UnsupportedDocumentError
from app.domain.documents import Entity, ExtractedText, Relationship
from app.ingestion.cleaning import TextCleaner
from app.ingestion.enrichment import ChunkEnricher
from app.ingestion.extraction import ExtractorRegistry
from app.ingestion.metadata import DocumentMetadataBuilder
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.ingestion.recursive_chunking import RecursiveChunker
from app.ingestion.structural_chunking import StructuralChunker
from app.knowledge.graph import KnowledgeGraphBuilder


class TextFileExtractor:
    """Test extractor so ingestion can be exercised without a real PDF."""

    supported_extensions = frozenset({".txt"})

    def extract(self, file_path):  # noqa: ANN001
        text = file_path.read_text(encoding="utf-8")
        return ExtractedText(text=text, page_count=1)


# ── Cleaning ─────────────────────────────────────────────────────


class TestTextCleaner:
    def test_normalises_whitespace_and_line_endings(self):
        cleaned = TextCleaner().clean("a\t\tb\r\nc   d\n\n\n\ne")
        assert cleaned == "a b\nc d\n\ne"

    def test_strips_zero_width_characters(self):
        assert TextCleaner().clean("po​licy﻿") == "policy"

    def test_empty_input_returns_empty_string(self):
        assert TextCleaner().clean("") == ""


# ── Structural chunking ──────────────────────────────────────────


class TestStructuralChunker:
    def test_splits_on_numbered_headings(self, sample_text):
        sections = StructuralChunker().split(sample_text)

        assert [section.section_id for section in sections] == [0, 1, 2]
        assert sections[1].title.startswith("2. LEAVE POLICY")

    def test_section_offsets_point_at_the_original_text(self, sample_text):
        for section in StructuralChunker().split(sample_text):
            assert sample_text[section.start : section.start + len(section.content)] == (
                section.content
            )

    def test_blank_document_yields_no_sections(self):
        assert StructuralChunker().split("   \n\n ") == []


# ── Recursive chunking ───────────────────────────────────────────


class TestRecursiveChunker:
    def test_chunks_carry_their_section_identity(self, sample_text):
        sections = StructuralChunker().split(sample_text)
        chunks = RecursiveChunker(chunk_size=60, chunk_overlap=10).chunk(sections)

        assert chunks
        assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))
        assert all(chunk.section_title for chunk in chunks)

    def test_chunk_offsets_resolve_to_the_chunk_text(self, sample_text):
        sections = StructuralChunker().split(sample_text)
        chunks = RecursiveChunker(chunk_size=60, chunk_overlap=10).chunk(sections)

        for chunk in chunks:
            assert sample_text[chunk.start : chunk.end] == chunk.text

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)


# ── Metadata ─────────────────────────────────────────────────────


class TestMetadataBuilder:
    def test_builds_metadata_from_file_and_text(self, tmp_path):
        path = tmp_path / "Handbook Draft.txt"
        path.write_text("one two three", encoding="utf-8")

        metadata = DocumentMetadataBuilder().build(path, page_count=4, text="one two three")

        assert metadata.filename == "Handbook Draft.txt"
        assert metadata.file_type == "txt"
        assert metadata.pages == 4
        assert metadata.word_count == 3
        assert metadata.char_count == 13
        assert metadata.uploaded_at.tzinfo is not None


# ── Knowledge extraction ─────────────────────────────────────────


# Relationship extraction now lives in tests/test_knowledge_graph.py, which
# exercises the dependency-parse extractor that replaced the phrase matcher.


class TestKnowledgeGraphBuilder:
    def test_promotes_relationship_endpoints_to_nodes(self):
        graph = KnowledgeGraphBuilder().build(
            entities=[Entity(text="Finance", label="ORG")],
            relationships=[
                Relationship(source="Priya", relation="WORKS_IN", target="Finance")
            ],
        )

        assert {entity.text for entity in graph.entities} == {"Finance", "Priya"}
        assert graph.edge_count == 1

    def test_deduplicates_entities_by_text(self):
        graph = KnowledgeGraphBuilder().build(
            entities=[Entity(text="Finance", label="ORG"), Entity(text="Finance", label="ORG")],
            relationships=[],
        )

        assert graph.node_count == 1


# ── Enrichment ───────────────────────────────────────────────────


class TestChunkEnricher:
    def test_stamps_document_id_and_entities(self, entity_extractor, sample_text):
        sections = StructuralChunker().split(sample_text)
        chunks = RecursiveChunker(chunk_size=80, chunk_overlap=10).chunk(sections)

        enriched = ChunkEnricher(entity_extractor).enrich(chunks, "doc-42")

        assert all(chunk.document_id == "doc-42" for chunk in enriched)
        assert enriched[0].vector_id == "doc-42_0"
        assert any(chunk.entities for chunk in enriched)

    def test_empty_chunk_list_is_a_no_op(self, entity_extractor):
        assert ChunkEnricher(entity_extractor).enrich([], "doc-42") == []


# ── Orchestrator ─────────────────────────────────────────────────


class TestIngestionPipeline:
    @pytest.fixture
    def pipeline(self, entity_extractor) -> DocumentIngestionPipeline:
        return DocumentIngestionPipeline(
            extractors=ExtractorRegistry([TextFileExtractor()]),
            recursive_chunker=RecursiveChunker(chunk_size=120, chunk_overlap=20),
            entity_extractor=entity_extractor,
        )

    def test_runs_every_stage(self, pipeline, tmp_path, sample_text):
        path = tmp_path / "handbook.txt"
        path.write_text(sample_text, encoding="utf-8")

        document = pipeline.run(path)

        assert document.metadata.filename == "handbook.txt"
        assert document.chunks
        assert all(chunk.document_id == document.document_id for chunk in document.chunks)
        assert document.entities
        assert any(rel.relation == "WORKS_IN" for rel in document.relationships)

    def test_rejects_unsupported_file_types(self, pipeline, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("hello", encoding="utf-8")

        with pytest.raises(UnsupportedDocumentError):
            pipeline.run(path)

    def test_rejects_missing_files(self, pipeline, tmp_path):
        with pytest.raises(DocumentProcessingError, match="File not found"):
            pipeline.run(tmp_path / "absent.txt")

    def test_rejects_documents_with_no_extractable_text(self, pipeline, tmp_path):
        path = tmp_path / "blank.txt"
        path.write_text("   \n\n  ", encoding="utf-8")

        with pytest.raises(DocumentProcessingError, match="No extractable text"):
            pipeline.run(path)
