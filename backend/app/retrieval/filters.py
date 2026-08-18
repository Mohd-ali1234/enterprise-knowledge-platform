"""Metadata filtering, shared by every retrieval strategy.

Both the dense and the sparse path must interpret a filter dictionary
identically, so the normalisation rules live here once. This is also where
OpenAPI placeholder values (`{"additionalProp1": {}}`) are discarded, which
otherwise silently return zero results.
"""

from __future__ import annotations

from typing import Any


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Drop keys whose value carries no constraint.

    Empty dicts, empty lists and `None` are treated as "no filter" rather than
    as "match nothing".
    """
    if not filters:
        return {}

    return {
        key: value
        for key, value in filters.items()
        if value is not None and value != {} and value != []
    }


def matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Whether a chunk's metadata satisfies every filter.

    A list value means "any of"; any other value means exact equality.
    """
    for key, expected in normalize_filters(filters).items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False

    return True


def to_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate filters into a Chroma `where` clause.

    Returns `None` when nothing constrains the query, because Chroma rejects an
    empty `where`.
    """
    normalized = normalize_filters(filters)
    if not normalized:
        return None

    clauses = [
        {key: {"$in": value}} if isinstance(value, list) else {key: value}
        for key, value in normalized.items()
    ]

    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
