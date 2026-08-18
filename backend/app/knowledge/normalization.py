"""Entity normalisation and conservative merging.

Two jobs:

* `canonical_name` turns a surface form into a stable merge key, so
  "Microsoft Corporation", "Microsoft Corp." and "Microsoft" collapse to one
  node.
* `EntityResolver` decides whether a new mention is the entity it looks like,
  using RapidFuzz similarity above a high threshold.

The bias throughout is against merging. A wrong merge silently fuses two real
people into one node and corrupts every edge attached to them; a missed merge
just leaves two nodes the user can see are related. So merging requires a high
score *and* a matching entity type.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from app.core.config import settings
from app.domain.knowledge import EntityType

# Legal and corporate suffixes stripped when building a canonical key.
_ORG_SUFFIXES = (
    "corporation", "corp", "incorporated", "inc", "limited", "ltd", "llc",
    "llp", "plc", "gmbh", "ag", "sa", "nv", "bv", "co", "company", "holdings",
    "group", "technologies", "technology", "labs", "laboratories",
)

# Leading determiners and honorifics that never belong in a canonical key.
_LEADING_NOISE = (
    "the", "a", "an", "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "sir", "madam",
)

_PUNCTUATION = re.compile(r"[^\w\s&-]")
_WHITESPACE = re.compile(r"\s+")
_POSSESSIVE = re.compile(r"'s\b|’s\b", re.IGNORECASE)


def canonical_name(text: str, entity_type: EntityType | None = None) -> str:
    """Build the merge key for a surface form.

    Lower-cased, stripped of possessives, punctuation, leading determiners and
    honorifics, and - for organisations - trailing legal suffixes.
    """
    cleaned = _POSSESSIVE.sub("", text or "")
    cleaned = _PUNCTUATION.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip().lower()
    if not cleaned:
        return ""

    words = cleaned.split()
    while words and words[0] in _LEADING_NOISE:
        words.pop(0)

    # Only organisations lose suffixes: "Apple Inc" and "Apple" are the same
    # company, but a person's surname must never be trimmed this way.
    if entity_type in (EntityType.ORG, EntityType.DEPARTMENT, None):
        while len(words) > 1 and words[-1] in _ORG_SUFFIXES:
            words.pop()

    return " ".join(words)


def display_name(text: str) -> str:
    """Tidy a surface form for presentation, preserving its original casing."""
    cleaned = _WHITESPACE.sub(" ", (text or "").strip())
    cleaned = _POSSESSIVE.sub("", cleaned)
    words = cleaned.split()
    while words and words[0].lower() in ("the", "a", "an"):
        words.pop(0)
    return " ".join(words) or cleaned


class EntityResolver:
    """Resolves mentions onto canonical entities within one ingestion run.

    Exact canonical matches are free. Anything else needs a fuzzy score above
    `knowledge_graph_entity_merge_threshold` and an identical type, and an
    acronym is only expanded when its letters match the initials of exactly one
    candidate - "IT" must not swallow "Intuit".
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = (
            threshold
            if threshold is not None
            else settings.knowledge_graph_entity_merge_threshold
        )
        self._known: dict[tuple[EntityType, str], str] = {}

    def resolve(self, name: str, entity_type: EntityType) -> str:
        """Return the canonical key this mention belongs to, registering it."""
        key = canonical_name(name, entity_type)
        if not key:
            return ""

        exact = (entity_type, key)
        if exact in self._known:
            return self._known[exact]

        match = self._best_match(key, entity_type)
        resolved = match or key
        self._known[exact] = resolved
        return resolved

    def _best_match(self, key: str, entity_type: EntityType) -> str | None:
        candidates = [
            existing for (known_type, _), existing in self._known.items()
            if known_type == entity_type
        ]
        if not candidates:
            return None

        # An acronym matches only if it uniquely expands to one candidate.
        if key.isupper() or (len(key) <= 5 and " " not in key):
            initials = [c for c in set(candidates) if _initials(c) == key.replace(" ", "")]
            return initials[0] if len(initials) == 1 else None

        best, best_score = None, 0.0
        for candidate in set(candidates):
            # token_sort_ratio so word order does not defeat the match, but a
            # containment bonus is deliberately not applied: "Finance" and
            # "Finance Committee" are different things.
            score = fuzz.token_sort_ratio(key, candidate)
            if score > best_score:
                best, best_score = candidate, score

        return best if best_score >= self.threshold else None


def _initials(name: str) -> str:
    return "".join(word[0] for word in name.split() if word).lower()


__all__ = ["EntityResolver", "canonical_name", "display_name"]
