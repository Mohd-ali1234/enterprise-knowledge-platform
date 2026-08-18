"""Shared lexical tokenisation.

The sparse retriever and the lexical reranker must agree on what a "term" is,
otherwise their scores are not comparable. Both import from here.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+")

_MIN_TERM_LENGTH = 2

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "am", "an", "and", "are", "as", "at",
        "be", "by", "can", "come", "for", "from", "how",
        "i", "in", "is", "it", "me", "my", "of", "on", "or",
        "our", "the", "to", "we", "what", "when", "where",
        "which", "who", "why", "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word characters, and drop stopwords/short tokens."""
    return [
        token
        for token in _WORD.findall(text.lower())
        if len(token) >= _MIN_TERM_LENGTH and token not in STOPWORDS
    ]


def term_set(text: str) -> set[str]:
    """Unique terms in `text`."""
    return set(tokenize(text))
