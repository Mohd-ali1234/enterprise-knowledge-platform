"""Stage: Entity Extraction.

Two audiences, one spaCy parse:

* `SpacyEntityExtractor` satisfies the existing `EntityExtractor` protocol and
  feeds `ChunkEnricher`, which stamps entity names onto vector metadata. That
  behaviour is unchanged.
* `entity_from_span` types a span for the knowledge graph, mapping spaCy's NER
  labels onto our vocabulary and recognising the enterprise types the general
  model has no head for - ROLE, DEPARTMENT, PROJECT, TECHNOLOGY.

The heuristic fallback still exists for when spaCy is unavailable, but it is no
longer the graph's source of entities: turning every capitalised phrase into a
node is exactly the noise this rewrite removes. The graph only accepts spans
that spaCy's NER labelled, or whose head noun identifies a known type.
"""

from __future__ import annotations

import re
from typing import Iterable, Protocol, Sequence, runtime_checkable

from app.core.logging import get_logger
from app.domain.documents import Entity
from app.domain.knowledge import EntityType, ExtractedEntity
from app.knowledge.normalization import canonical_name, display_name
from app.knowledge.nlp import SPACY_LABEL_TYPES, NlpPipeline, classify_span

logger = get_logger(__name__)

_CAPITALISED_PHRASE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*\b")

# Types that are attributes of an entity rather than entities worth linking.
_NON_LINKABLE = frozenset({EntityType.DATE, EntityType.MONEY})

# Types a general-purpose NER model has no head for. When a noun phrase reads
# as one of these, it is a better answer than whatever NER guessed.
ENTERPRISE_TYPES = frozenset(
    {EntityType.ROLE, EntityType.DEPARTMENT, EntityType.PROJECT, EntityType.TECHNOLOGY}
)

_MIN_ENTITY_CHARS = 2


@runtime_checkable
class EntityExtractor(Protocol):
    """Finds named entities in text."""

    def extract(self, text: str) -> list[Entity]:
        ...

    def extract_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        ...


def _deduplicate(entities: Iterable[Entity]) -> list[Entity]:
    seen: set[tuple[str, str]] = set()
    unique: list[Entity] = []
    for entity in entities:
        key = (entity.text, entity.label)
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    return unique


# ── Graph-facing extraction ──────────────────────────────────────


def entity_from_span(span) -> ExtractedEntity | None:  # noqa: ANN001 - spaCy Span
    """Type a span for the knowledge graph, or reject it.

    Returns None when the span is not a usable entity: too short, a bare
    pronoun, or a phrase whose type cannot be established. Rejecting is the
    point - an untyped node is worse than no node.
    """
    if span is None or not len(span):
        return None

    # NER spans keep their determiner ("the Head of Finance"); the graph wants
    # the name, so the label reads the same however the sentence phrased it.
    text = display_name(span.text)
    if len(text) < _MIN_ENTITY_CHARS:
        return None

    entity_type = _type_for(span)
    if entity_type is None:
        return None

    key = canonical_name(text, entity_type)
    if not key:
        return None

    return ExtractedEntity(
        text=text,
        canonical_name=key,
        type=entity_type,
        start=span.start_char,
        end=span.end_char,
        confidence=0.9 if entity_type is not EntityType.OTHER else 0.6,
    )


def _type_for(span) -> EntityType | None:  # noqa: ANN001 - spaCy Span
    """Resolve a span's type, preferring the most specific reading.

    The head noun is consulted *before* the NER label, because a general model
    has no enterprise vocabulary: it labels "CFO", "Project Atlas" and "the
    Engineering team" all as ORG. Their head nouns identify them precisely, so
    the specific reading wins where there is one.
    """
    specific = classify_span(span)
    if specific is not None:
        return specific

    # An NER label on the span itself, or on an entity covering it.
    label = getattr(span, "label_", "") or ""
    if label:
        return SPACY_LABEL_TYPES.get(label, EntityType.OTHER)

    for ent in span.ents:
        if ent.start <= span.root.i < ent.end:
            return SPACY_LABEL_TYPES.get(ent.label_, EntityType.OTHER)

    # Accept only a proper noun. spaCy's tagger separates a real name
    # ("Priya", "Acme") from a merely capitalised common noun at the start of a
    # sentence, which is the distinction the old regex extractor could not make.
    if span.root.pos_ == "PROPN":
        return EntityType.OTHER

    return None


def is_linkable(entity: ExtractedEntity) -> bool:
    """Whether an entity should become a graph node.

    Dates and money are real extractions but poor nodes - they connect
    everything to everything and drown the graph.
    """
    return entity.type not in _NON_LINKABLE


# ── Protocol implementations (chunk enrichment) ──────────────────


class RegexEntityExtractor:
    """Heuristic extractor: treats capitalised phrases as entities.

    Used as the fallback when spaCy or its model is unavailable. It feeds chunk
    metadata only; the knowledge graph does not accept its output.
    """

    label = "UNKNOWN"

    def extract(self, text: str) -> list[Entity]:
        candidates = (
            Entity(text=match.strip(), label=self.label)
            for match in _CAPITALISED_PHRASE.findall(text)
            if len(match.strip()) > 1
        )
        return _deduplicate(candidates)

    def extract_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        return [self.extract(text) for text in texts]


class SpacyEntityExtractor:
    """spaCy NER, with automatic degradation to `RegexEntityExtractor`.

    Delegates model loading to the shared `NlpPipeline`, so the graph stages and
    chunk enrichment use one loaded model rather than two.
    """

    def __init__(self, model_name: str | None = None, fallback: EntityExtractor | None = None):
        self._pipeline = NlpPipeline(model_name)
        self.model_name = self._pipeline.model_name
        self.fallback = fallback or RegexEntityExtractor()

    def extract(self, text: str) -> list[Entity]:
        doc = self._pipeline.parse(text)
        if doc is None:
            return self.fallback.extract(text)

        return _deduplicate(
            Entity(text=ent.text.strip(), label=ent.label_) for ent in doc.ents
        )

    def extract_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        """Extract from many texts at once, in one pass through the model."""
        if not self._pipeline.is_available:
            return self.fallback.extract_batch(list(texts))

        return [
            _deduplicate(Entity(text=ent.text.strip(), label=ent.label_) for ent in doc.ents)
            for doc in self._pipeline.pipe(list(texts))
        ]


__all__ = [
    "ENTERPRISE_TYPES",
    "EntityExtractor",
    "RegexEntityExtractor",
    "SpacyEntityExtractor",
    "display_name",
    "entity_from_span",
    "is_linkable",
]
