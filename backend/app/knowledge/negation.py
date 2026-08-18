"""Negation detection over a spaCy parse.

The old extractor turned

    "Valve is a flat organization where nobody reports to anyone else."

into `Valve -> REPORTS_TO -> anyone else`, asserting the opposite of the
sentence. Two independent checks stop that class of error:

* `is_negated` walks the dependency tree around the predicate looking for a
  negation marker attached to it, or a negative-quantifier subject
  ("nobody", "no one", "neither").
* `is_vacuous_argument` rejects arguments that are negative or indefinite
  pronouns ("anyone else", "nothing"), which never name a real entity.

Both are conservative: they suppress rather than guess.
"""

from __future__ import annotations

# Markers that negate the clause they attach to.
NEGATION_TOKENS = frozenset(
    {
        "not", "n't", "never", "no", "none", "nobody", "no-one", "noone",
        "nothing", "nowhere", "neither", "nor", "without", "cannot",
    }
)

# Subjects that assert the *absence* of a relation. "Nobody reports to X"
# means the REPORTS_TO edge does not exist.
NEGATIVE_QUANTIFIERS = frozenset(
    {"nobody", "no one", "noone", "none", "nothing", "neither", "no-one"}
)

# Arguments that never name an entity, negative or otherwise.
VACUOUS_ARGUMENTS = frozenset(
    {
        "anyone", "anyone else", "anybody", "anybody else", "someone",
        "somebody", "everyone", "everybody", "no one", "nobody", "nothing",
        "anything", "something", "everything", "each other", "one another",
        "it", "they", "them", "this", "that", "these", "those", "he", "she",
        "him", "her", "his", "hers", "we", "us", "you", "i", "me", "who",
        "which", "what", "there", "here",
    }
)


def is_negated(token) -> bool:  # noqa: ANN001 - spaCy Token, untyped
    """Whether the predicate token sits inside a negated clause."""
    # Direct negation attached to the verb: "does not report", "never reports".
    for child in token.children:
        if child.dep_ == "neg" or child.lower_ in NEGATION_TOKENS:
            return True

    # A negative quantifier as subject: "nobody reports to anyone else".
    for child in token.children:
        if child.dep_ in ("nsubj", "nsubjpass") and _is_negative_quantifier(child):
            return True

    # Auxiliaries carry the negation in some parses: "cannot report".
    for child in token.children:
        if child.dep_ == "aux" and child.lower_ in NEGATION_TOKENS:
            return True

    return False


def is_vacuous_argument(text: str) -> bool:
    """Whether an argument is a pronoun or quantifier rather than an entity."""
    cleaned = " ".join((text or "").lower().split())
    if not cleaned:
        return True
    if cleaned in VACUOUS_ARGUMENTS:
        return True
    # "anyone else", "nobody at all" - the head word decides.
    head = cleaned.split()[0]
    return head in VACUOUS_ARGUMENTS or head in NEGATIVE_QUANTIFIERS


def _is_negative_quantifier(token) -> bool:  # noqa: ANN001 - spaCy Token
    text = token.lower_
    if text in NEGATIVE_QUANTIFIERS:
        return True
    # "no employee reports to..." - negation sits on the determiner.
    return any(child.lower_ in ("no", "neither") for child in token.children)


__all__ = [
    "NEGATION_TOKENS",
    "NEGATIVE_QUANTIFIERS",
    "VACUOUS_ARGUMENTS",
    "is_negated",
    "is_vacuous_argument",
]
