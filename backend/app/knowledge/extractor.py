"""Orchestrator: text -> `DocumentKnowledge`.

Sequences the graph-side stages and owns the single spaCy parse they share:

    clean text
      -> sentence segmentation
      -> entity extraction   (typed, NER-backed)
      -> entity normalisation (canonical names, conservative merging)
      -> relationship extraction (dependency parse)
      -> relationship validation (threshold, self-loops, vacuous arguments)

The parse happens once per document and both stages read it, which is what
keeps this affordable to run on every ingest.
"""

from __future__ import annotations

import time

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.knowledge import (
    DocumentKnowledge,
    ExtractedEntity,
    ExtractedRelation,
    Polarity,
)
from app.knowledge.entities import entity_from_span, is_linkable
from app.knowledge.normalization import EntityResolver
from app.knowledge.nlp import NlpPipeline, sentences
from app.knowledge.relationships import DependencyRelationshipExtractor

logger = get_logger(__name__)

_TERMINAL_PUNCTUATION = (".", "!", "?", ":", ";")
_MAX_HEADING_WORDS = 8


def _terminate_headings(text: str) -> tuple[str, list[int]]:
    """Give unpunctuated headings a full stop so they end a sentence.

    Headings arrive on their own line with no terminal punctuation, so spaCy
    merges them into the sentence that follows. That produced subjects like
    "Acme Corporation Organisation Handbook Leadership Priya Sharma" instead of
    "Priya Sharma". A short line with no closing punctuation is a heading, and
    ending it keeps the following sentence clean.

    Deliberately conservative: prose wraps to long lines, so the word limit
    leaves ordinary text untouched.

    Returns the rewritten text and the offsets of every inserted character, so
    positions can be mapped back onto the original text. Evidence traceability
    depends on those offsets pointing at the real chunk.
    """
    lines = text.split("\n")
    insertions: list[int] = []
    position = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        is_heading = (
            stripped
            and not stripped.endswith(_TERMINAL_PUNCTUATION)
            and len(stripped.split()) <= _MAX_HEADING_WORDS
        )
        if is_heading:
            lines[index] = line.rstrip() + "."
            insertions.append(position + len(line.rstrip()))
        position += len(lines[index]) + 1  # +1 for the newline

    return "\n".join(lines), insertions


def _original_offset(offset: int, insertions: list[int]) -> int:
    """Map an offset in the rewritten text back to the original text."""
    shift = sum(1 for insertion in insertions if insertion < offset)
    return max(0, offset - shift)


class KnowledgeExtractor:
    """Extracts a document's entities and relationships in one pass."""

    def __init__(
        self,
        pipeline: NlpPipeline | None = None,
        relationships: DependencyRelationshipExtractor | None = None,
        threshold: float | None = None,
    ):
        self._pipeline = pipeline or NlpPipeline()
        self._relationships = relationships or DependencyRelationshipExtractor(self._pipeline)
        self.threshold = (
            threshold
            if threshold is not None
            else settings.knowledge_graph_relationship_threshold
        )

    @property
    def is_available(self) -> bool:
        return self._pipeline.is_available

    def extract(self, text: str) -> DocumentKnowledge:
        """Extract entities and relationships from a whole document."""
        if not text.strip():
            return DocumentKnowledge()

        prepared, insertions = _terminate_headings(text)
        doc = self._pipeline.parse(prepared)
        if doc is None:
            logger.debug("spaCy unavailable; document contributes no graph data")
            return DocumentKnowledge()

        started = time.perf_counter()
        resolver = EntityResolver()
        entities: dict[str, ExtractedEntity] = {}
        relations: list[ExtractedRelation] = []

        for sentence in sentences(doc):
            for entity in self._sentence_entities(sentence, resolver):
                entities.setdefault(entity.canonical_name, entity)

            for relation in self._relationships.extract_from_sentence(sentence):
                relation.sentence_start = _original_offset(relation.sentence_start, insertions)
                resolved = self._resolve(relation, resolver)
                if self._is_valid(resolved):
                    relations.append(resolved)
                    # Endpoints become nodes even if NER missed them elsewhere,
                    # so the graph never holds an edge to a non-existent node.
                    for endpoint in (resolved.source, resolved.target):
                        entities.setdefault(endpoint.canonical_name, endpoint)

        knowledge = DocumentKnowledge(
            entities=list(entities.values()),
            relations=self._deduplicate(relations),
        )

        logger.info(
            "Extracted %d entities and %d relationships in %.2fs",
            len(knowledge.entities),
            len(knowledge.relations),
            time.perf_counter() - started,
        )
        return knowledge

    # ── Internals ────────────────────────────────────────────────

    def _sentence_entities(self, sentence, resolver: EntityResolver) -> list[ExtractedEntity]:  # noqa: ANN001
        found: list[ExtractedEntity] = []
        for ent in sentence.ents:
            entity = entity_from_span(ent)
            if entity is None or not is_linkable(entity):
                continue
            entity.canonical_name = resolver.resolve(entity.text, entity.type)
            found.append(entity)
        return found

    @staticmethod
    def _resolve(relation: ExtractedRelation, resolver: EntityResolver) -> ExtractedRelation:
        """Point a relation's endpoints at their resolved canonical entities."""
        relation.source.canonical_name = resolver.resolve(
            relation.source.text, relation.source.type
        )
        relation.target.canonical_name = resolver.resolve(
            relation.target.text, relation.target.type
        )
        return relation

    def _is_valid(self, relation: ExtractedRelation) -> bool:
        """Final gate before a relation is allowed into the graph."""
        if not relation.source.canonical_name or not relation.target.canonical_name:
            return False
        # Normalisation can collapse both endpoints onto one entity.
        if relation.source.canonical_name == relation.target.canonical_name:
            return False
        if not is_linkable(relation.source) or not is_linkable(relation.target):
            return False
        # Negative statements are kept for the record but are never asserted.
        if relation.polarity is Polarity.NEGATIVE:
            return True
        return relation.confidence >= self.threshold

    @staticmethod
    def _deduplicate(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
        """Collapse repeats within one document, keeping the best-scoring one."""
        best: dict[tuple[str, str, str, str], ExtractedRelation] = {}
        for relation in relations:
            key = (
                relation.source.canonical_name,
                relation.type.value,
                relation.target.canonical_name,
                relation.polarity.value,
            )
            existing = best.get(key)
            if existing is None or relation.confidence > existing.confidence:
                best[key] = relation
        return list(best.values())


__all__ = ["KnowledgeExtractor"]
