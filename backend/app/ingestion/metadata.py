"""Stage: Metadata Extraction."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.domain.documents import DocumentMetadata


class DocumentMetadataBuilder:
    """Builds descriptive metadata for an ingested file."""

    def build(
        self,
        file_path: Path | str,
        page_count: int,
        text: str,
        display_name: str | None = None,
    ) -> DocumentMetadata:
        """Build metadata for a file.

        `display_name` is the name to report back to the caller. It exists
        because uploads are stored under a sanitised, uniquified name while the
        user should still see the name they uploaded.
        """
        path = Path(file_path)

        return DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename=display_name or path.name,
            file_type=path.suffix.lstrip(".").lower(),
            file_size=path.stat().st_size,
            pages=page_count,
            char_count=len(text),
            word_count=len(text.split()),
            uploaded_at=datetime.now(timezone.utc),
        )
