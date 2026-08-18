"""Storage for uploaded source files."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM_LENGTH = 80


class UploadStorage:
    """Writes uploaded files to disk under sanitised names."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or settings.upload_dir)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, stream: BinaryIO) -> Path:
        """Persist `stream` and return the path it was written to.

        The client-supplied name is never trusted as a path: it is stripped to
        its basename, sanitised, and suffixed with a short unique token so
        concurrent uploads of the same name cannot overwrite each other.
        """
        destination = self.directory / self._safe_name(filename)

        try:
            with destination.open("wb") as target:
                shutil.copyfileobj(stream, target)
        except OSError as exc:
            raise DocumentProcessingError(f"Could not save '{filename}': {exc}") from exc

        logger.debug("Saved upload to %s", destination)
        return destination

    @staticmethod
    def _safe_name(filename: str) -> str:
        original = Path(filename or "document").name  # discards any directory part
        stem = _UNSAFE_CHARS.sub("_", Path(original).stem)[:_MAX_STEM_LENGTH] or "document"
        suffix = _UNSAFE_CHARS.sub("", Path(original).suffix).lower()
        return f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"
