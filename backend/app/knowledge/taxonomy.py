"""Relationship taxonomy: verb phrase -> normalised `RelationType`.

Kept declarative and separate from extraction so the vocabulary can grow
without touching the parser. A predicate that matches nothing is not dropped -
it becomes `RELATED_TO` and keeps its original wording, which is how the graph
records "these two things are connected, and here is how it was phrased".

Passive voice flips direction rather than inventing a type: "X is managed by Y"
is `MANAGED_BY`, the declared inverse of `MANAGES`.
"""

from __future__ import annotations

from app.domain.knowledge import RelationType

# Lemmatised predicate -> type. Matched against the verb lemma plus any
# preposition, e.g. "report to", "work in", "be part of".
PREDICATE_TYPES: dict[str, RelationType] = {
    # Employment and reporting
    "work in": RelationType.WORKS_IN,
    "work at": RelationType.WORKS_IN,
    "work for": RelationType.WORKS_FOR,
    "report to": RelationType.REPORTS_TO,
    "employ": RelationType.WORKS_FOR,
    # Management
    "manage": RelationType.MANAGES,
    "oversee": RelationType.MANAGES,
    "supervise": RelationType.MANAGES,
    "lead": RelationType.LEADS,
    "head": RelationType.LEADS,
    "run": RelationType.MANAGES,
    "be responsible for": RelationType.RESPONSIBLE_FOR,
    "be accountable for": RelationType.RESPONSIBLE_FOR,
    # Ownership and composition
    "own": RelationType.OWNS,
    "be part of": RelationType.PART_OF,
    "belong to": RelationType.BELONGS_TO,
    "be member of": RelationType.MEMBER_OF,
    "be a member of": RelationType.MEMBER_OF,
    "join": RelationType.MEMBER_OF,
    "include": RelationType.PART_OF,
    # Location
    "be located in": RelationType.LOCATED_IN,
    "be based in": RelationType.LOCATED_IN,
    "headquarter in": RelationType.LOCATED_IN,
    "locate in": RelationType.LOCATED_IN,
    "base in": RelationType.LOCATED_IN,
    "operate in": RelationType.LOCATED_IN,
    # Products and technology
    "use": RelationType.USES,
    "adopt": RelationType.USES,
    "develop": RelationType.DEVELOPS,
    "build": RelationType.DEVELOPS,
    "maintain": RelationType.DEVELOPS,
    "provide": RelationType.PROVIDES,
    "offer": RelationType.PROVIDES,
    "supply": RelationType.PROVIDES,
    # Corporate events
    "acquire": RelationType.ACQUIRED,
    "buy": RelationType.ACQUIRED,
    "purchase": RelationType.ACQUIRED,
    "found": RelationType.FOUNDED,
    "establish": RelationType.FOUNDED,
    "create": RelationType.CREATED,
    "launch": RelationType.CREATED,
}

# Passive constructions map to the inverse type: "Finance is managed by Priya"
# keeps Finance as the subject and records MANAGED_BY.
PASSIVE_INVERSES: dict[RelationType, RelationType] = {
    RelationType.MANAGES: RelationType.MANAGED_BY,
    RelationType.OWNS: RelationType.OWNED_BY,
    RelationType.LEADS: RelationType.MANAGED_BY,
    RelationType.WORKS_FOR: RelationType.WORKS_FOR,
    RelationType.DEVELOPS: RelationType.DEVELOPS,
    RelationType.CREATED: RelationType.CREATED,
    RelationType.PROVIDES: RelationType.PROVIDES,
    RelationType.ACQUIRED: RelationType.ACQUIRED,
    RelationType.FOUNDED: RelationType.FOUNDED,
    RelationType.USES: RelationType.USES,
}

# Types whose meaning depends on argument order being right. Used by scoring:
# a directional type extracted from a passive clause without a clear agent is
# less trustworthy than one from an active clause.
DIRECTIONAL_TYPES = frozenset(
    {
        RelationType.REPORTS_TO,
        RelationType.MANAGES,
        RelationType.MANAGED_BY,
        RelationType.OWNS,
        RelationType.OWNED_BY,
        RelationType.PART_OF,
        RelationType.ACQUIRED,
    }
)


def swaps_arguments_when_passive(relation_type: RelationType) -> bool:
    """Whether a passive clause of this type needs its arguments swapped.

    A type with a declared inverse keeps its argument order - "Finance is
    managed by Priya" is `Finance MANAGED_BY Priya`. A type that is its own
    inverse has no way to record the flip, so the arguments must swap instead:
    "Helios is used by Engineering" is `Engineering USES Helios`, not the
    reverse.
    """
    return PASSIVE_INVERSES.get(relation_type) is relation_type


def classify(lemma: str, preposition: str = "", passive: bool = False) -> tuple[RelationType, str]:
    """Map a predicate onto the taxonomy.

    Returns the type and the predicate string that was matched, so callers can
    store the original wording alongside a `RELATED_TO` fallback.

    Args:
        lemma: The verb's lemma, e.g. "report", "be".
        preposition: Attached preposition or complement, e.g. "to", "part of".
        passive: Whether the clause was passive, which selects the inverse type.
    """
    predicate = f"{lemma} {preposition}".strip().lower()

    relation = PREDICATE_TYPES.get(predicate)
    if relation is None and preposition:
        # "is part of" parses as lemma="be", prep="part of"; also try the bare
        # verb so "manages the team" and "manage" agree.
        relation = PREDICATE_TYPES.get(lemma.lower())
    if relation is None:
        relation = PREDICATE_TYPES.get(lemma.lower())

    if relation is None:
        return RelationType.RELATED_TO, predicate

    if passive:
        relation = PASSIVE_INVERSES.get(relation, relation)

    return relation, predicate


__all__ = [
    "DIRECTIONAL_TYPES",
    "PASSIVE_INVERSES",
    "PREDICATE_TYPES",
    "classify",
]
