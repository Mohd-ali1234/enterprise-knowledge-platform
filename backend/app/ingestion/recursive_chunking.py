"""Stage: Recursive Chunking.

Splits each section into overlapping, embedding-sized chunks. LangChain's
recursive splitter is used when installed; a character-window splitter with the
same size/overlap semantics is used otherwise, so ingestion has no hard
dependency on LangChain.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.documents import DocumentChunk, DocumentSection

logger = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    """Turns sections into `DocumentChunk`s with document-level offsets."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._splitter = self._build_splitter()

    def _build_splitter(self):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            logger.warning(
                "langchain-text-splitters not installed; using window-based chunking"
            )
            return None

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_SEPARATORS,
        )

    def chunk(self, sections: list[DocumentSection]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for section in sections:
            for text in self._split_text(section.content):
                start = self._locate(section, text, search_from=self._cursor(chunks, section))
                chunks.append(
                    DocumentChunk(
                        chunk_id=len(chunks),
                        section_id=section.section_id,
                        section_title=section.title,
                        text=text,
                        start=start,
                        end=start + len(text),
                    )
                )

        logger.debug("Produced %d chunks from %d sections", len(chunks), len(sections))
        return chunks

    def _split_text(self, text: str) -> list[str]:
        if self._splitter is not None:
            pieces = self._splitter.split_text(text)
        else:
            pieces = self._window_split(text)

        return [piece for piece in (p.strip() for p in pieces) if piece]

    def _window_split(self, text: str) -> list[str]:
        """Fixed-width sliding window, used when LangChain is unavailable."""
        step = self.chunk_size - self.chunk_overlap
        return [text[start : start + self.chunk_size] for start in range(0, len(text), step)]

    @staticmethod
    def _cursor(chunks: list[DocumentChunk], section: DocumentSection) -> int:
        """Where in the section to resume searching for the next chunk."""
        for chunk in reversed(chunks):
            if chunk.section_id == section.section_id:
                return max(chunk.start - section.start, 0)
        return 0

    @staticmethod
    def _locate(section: DocumentSection, text: str, search_from: int) -> int:
        """Resolve a chunk's absolute offset in the cleaned document text.

        Splitters may normalise whitespace, so a chunk is not guaranteed to be
        found verbatim; the section start is a safe fallback.
        """
        position = section.content.find(text, search_from)
        if position == -1:
            position = section.content.find(text)
        return section.start + max(position, 0)
