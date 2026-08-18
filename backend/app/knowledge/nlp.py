"""Shared spaCy pipeline.

Entity extraction and relationship extraction both need the same parse, and
loading a spaCy model is expensive relative to a document, so the pipeline is
loaded **once per process** and handed out here. Both stages then work from a
single `Doc`, rather than parsing the same text twice.

The model is `en_core_web_sm` (~12 MB): it ships a dependency parser and an NER
head, runs on CPU in milliseconds, and needs no GPU, no network and no LLM.

Everything degrades rather than fails. If spaCy or its model is missing,
`is_available` is False, entity extraction falls back to a capitalisation
heuristic, and relationship extraction returns nothing - a document still
ingests, it simply contributes no edges.
"""

from __future__ import annotations

from typing import Iterator

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.knowledge import EntityType

logger = get_logger(__name__)

# spaCy NER label -> our normalised vocabulary.
SPACY_LABEL_TYPES: dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORG,
    "GPE": EntityType.LOCATION,
    "LOC": EntityType.LOCATION,
    "FAC": EntityType.LOCATION,
    "DATE": EntityType.DATE,
    "TIME": EntityType.DATE,
    "PRODUCT": EntityType.PRODUCT,
    "EVENT": EntityType.EVENT,
    "MONEY": EntityType.MONEY,
    "WORK_OF_ART": EntityType.OTHER,
    "LAW": EntityType.OTHER,
    "NORP": EntityType.OTHER,
    "LANGUAGE": EntityType.OTHER,
}

# Head nouns that identify a span spaCy's NER does not label, e.g. "the Finance
# department", "Project Alpha", "the CFO". These are the enterprise types the
# general-purpose model has no head for.
ROLE_HEADS = frozenset(
    {
        "ceo", "cfo", "cto", "coo", "cio", "ciso", "chair", "chairman",
        "chairwoman", "president", "director", "manager", "supervisor",
        "head", "lead", "officer", "administrator", "analyst", "engineer",
        "architect", "designer", "controller", "treasurer", "secretary",
        "partner", "principal", "founder", "owner", "executive", "vp",
    }
)

DEPARTMENT_HEADS = frozenset(
    {
        "department", "team", "division", "unit", "group", "office", "bureau",
        "function", "practice", "squad", "guild",
    }
)

PROJECT_HEADS = frozenset({"project", "programme", "program", "initiative", "workstream"})

TECHNOLOGY_HEADS = frozenset(
    {
        "platform", "framework", "database", "api", "service", "system",
        "library", "toolkit", "sdk", "protocol", "pipeline", "stack",
    }
)


class NlpPipeline:
    """Lazily loaded, process-wide spaCy pipeline."""

    _nlp = None
    _load_attempted = False

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.spacy_model

    @classmethod
    def _load(cls, model_name: str):
        if cls._load_attempted:
            return cls._nlp

        cls._load_attempted = True
        try:
            import spacy

            cls._nlp = spacy.load(model_name)
            logger.info("Loaded spaCy model '%s' (parser + NER)", model_name)
        except Exception as exc:  # noqa: BLE001 - any failure means "degrade"
            logger.warning(
                "spaCy model '%s' unavailable (%s); knowledge graph extraction "
                "is disabled and entity extraction falls back to a heuristic",
                model_name,
                exc,
            )
            cls._nlp = None
        return cls._nlp

    @property
    def nlp(self):
        return self._load(self.model_name)

    @property
    def is_available(self) -> bool:
        return self.nlp is not None

    def parse(self, text: str):
        """Parse one text into a spaCy `Doc`, or None when unavailable."""
        nlp = self.nlp
        return None if nlp is None else nlp(text)

    def pipe(self, texts: list[str]) -> Iterator:
        """Parse many texts, which is markedly faster than a loop."""
        nlp = self.nlp
        if nlp is None:
            return iter(())
        return nlp.pipe(texts)

    @classmethod
    def reset(cls) -> None:
        """Forget the cached model. Used by tests that stub the pipeline."""
        cls._nlp = None
        cls._load_attempted = False


def sentences(doc, max_chars: int | None = None) -> list:
    """Sentence-segment a parsed doc, skipping ones too long to be useful.

    A very long "sentence" is nearly always a parsing artefact of a table or a
    run-on heading block; parsing relations out of it produces noise.
    """
    limit = max_chars or settings.knowledge_graph_max_sentence_chars
    return [sent for sent in doc.sents if 0 < len(sent.text.strip()) <= limit]


def classify_span(span) -> EntityType | None:  # noqa: ANN001 - spaCy Span
    """Type a span that spaCy's NER did not label, from its head noun.

    Returns None when the span does not look like an entity at all, which is
    how "every capitalised phrase becomes an entity" is avoided.
    """
    words = [token.lower_ for token in span if not token.is_punct]
    if not words:
        return None

    head = span.root.lower_
    if head in ROLE_HEADS or words[0] in ROLE_HEADS:
        return EntityType.ROLE
    if head in DEPARTMENT_HEADS:
        return EntityType.DEPARTMENT
    if words[0] in PROJECT_HEADS:
        return EntityType.PROJECT
    if head in TECHNOLOGY_HEADS:
        return EntityType.TECHNOLOGY
    return None


__all__ = [
    "DEPARTMENT_HEADS",
    "NlpPipeline",
    "PROJECT_HEADS",
    "ROLE_HEADS",
    "SPACY_LABEL_TYPES",
    "TECHNOLOGY_HEADS",
    "classify_span",
    "sentences",
]
