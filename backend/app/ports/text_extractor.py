"""Port: file -> raw text."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.domain.documents import ExtractedText


@runtime_checkable
class TextExtractor(Protocol):
    """Reads text out of a source file.

    `supported_extensions` lets `ExtractorRegistry` dispatch on file type, so
    adding DOCX or HTML support means registering one more extractor.
    """

    supported_extensions: frozenset[str]

    def extract(self, file_path: Path) -> ExtractedText:
        """Extract text and page count.

        Raises:
            DocumentProcessingError: if the file cannot be read.
        """
        ...
