"""Stage: Structural Chunking.

Splits a document along its headings so that later recursive chunking never
merges text from two unrelated sections, and every chunk can carry the title of
the section it came from.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.domain.documents import DocumentSection

logger = get_logger(__name__)

# A heading is a numbered clause ("1. Scope"), an ALL-CAPS line, or a markdown
# heading. The newline is consumed and the heading itself starts the section.
_HEADING_BOUNDARY = re.compile(r"\n(?=(?:\d+\.\s+[A-Z]|[A-Z][A-Z\s]{3,}|#+\s+))")

_MAX_TITLE_LENGTH = 100


class StructuralChunker:
    """Splits cleaned text into titled sections that know their offsets."""

    def split(self, text: str) -> list[DocumentSection]:
        if not text.strip():
            return []

        sections: list[DocumentSection] = []

        for start, end in self._boundaries(text):
            block = text[start:end]
            stripped = block.strip()
            if not stripped:
                continue

            section_id = len(sections)
            sections.append(
                DocumentSection(
                    section_id=section_id,
                    title=self._title_for(stripped, section_id),
                    content=stripped,
                    # Offset of the stripped content, not the raw block.
                    start=start + (len(block) - len(block.lstrip())),
                )
            )

        logger.debug("Split document into %d sections", len(sections))
        return sections

    @staticmethod
    def _boundaries(text: str) -> list[tuple[int, int]]:
        """Return (start, end) offsets of each heading-delimited block."""
        cuts = [0, *(match.end() for match in _HEADING_BOUNDARY.finditer(text)), len(text)]
        return list(zip(cuts, cuts[1:]))

    @staticmethod
    def _title_for(content: str, section_id: int) -> str:
        first_line = content.splitlines()[0].strip()
        return first_line[:_MAX_TITLE_LENGTH] or f"Section {section_id}"
