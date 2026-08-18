"""Stage: Cleaning.

Normalises extracted text so that chunk boundaries and embeddings are not
polluted by PDF whitespace artefacts.
"""

from __future__ import annotations

import re

_SUBSTITUTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\r\n?"), "\n"),                  # normalise line endings
    (re.compile("[​-‏﻿]"), ""),     # zero-width & BOM characters
    (re.compile(r"\t"), " "),                      # tabs -> spaces
    (re.compile(r" {2,}"), " "),                   # collapse runs of spaces
    (re.compile(r"[ \t]+\n"), "\n"),               # strip trailing whitespace per line
    (re.compile(r"\n{3,}"), "\n\n"),               # collapse excessive blank lines
)


class TextCleaner:
    """Applies whitespace and control-character normalisation."""

    def clean(self, text: str) -> str:
        if not text:
            return ""

        for pattern, replacement in _SUBSTITUTIONS:
            text = pattern.sub(replacement, text)

        return text.strip()
