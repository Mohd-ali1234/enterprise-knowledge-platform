"""Stage: Relationship Extraction.

Replaces the previous phrase-matching extractor, which split a *line* on one of
six hardcoded strings and kept whatever fell either side. That produced both
false positives ("Valve is a flat organization where nobody" -> REPORTS_TO ->
"anyone else") and silent misses (any sentence containing two linking phrases
was discarded entirely).

This extractor works from spaCy's dependency parse instead:

    sentence -> predicate tokens -> (subject, object) arguments
             -> noun-phrase expansion -> taxonomy -> negation -> confidence

Handled explicitly:

* **Active clauses** - "Acme acquired XYZ" -> ACQUIRED.
* **Passive clauses** - "Finance is managed by Priya" keeps Finance as the
  subject and records the inverse type, MANAGED_BY.
* **Copular predicates** - "X is part of Y" reads the complement, not just the
  verb, so "be" + "part of" classifies as PART_OF.
* **Relative clauses** - "Priya reports to the Head of Finance, who reports to
  the CFO" resolves "who" to its antecedent, yielding *both* edges rather than
  discarding the sentence.
* **Conjunctions** - "X and Y report to Z" yields an edge for each conjunct.
* **Negation** - delegated to `app.knowledge.negation`; negated clauses are
  recorded with negative polarity, never as positive assertions.

Everything is local and deterministic. No LLM, no network, no paid API.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.knowledge import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    Polarity,
    RelationType,
)
from app.knowledge.negation import is_negated, is_vacuous_argument
from app.knowledge.nlp import NlpPipeline, sentences
from app.knowledge.taxonomy import (
    DIRECTIONAL_TYPES,
    classify,
    swaps_arguments_when_passive,
)

logger = get_logger(__name__)

# Dependency labels that introduce a clause's subject.
_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "expl"})
# Dependency labels that introduce a clause's object.
_OBJECT_DEPS = frozenset({"dobj", "obj", "dative", "attr", "oprd"})

# Complement nouns that combine with a copula into a relational predicate:
# "is part of", "is a member of", "is responsible for".
_COPULAR_COMPLEMENTS = frozenset(
    {"part", "member", "responsible", "accountable", "owner", "parent", "subsidiary"}
)

# Noun heads that take an "of" complement belonging to the same entity:
# "Head of Finance" is one role, not "Head" plus "Finance".
_OF_ABSORBING_HEADS = frozenset(
    {
        "head", "director", "chief", "manager", "lead", "vp", "president",
        "owner", "member", "part", "chair", "chairman", "officer", "board",
    }
)

_MAX_ARGUMENT_TOKENS = 8


class DependencyRelationshipExtractor:
    """Extracts typed, scored relationships from a parsed sentence."""

    def __init__(self, pipeline: NlpPipeline | None = None, threshold: float | None = None):
        self._pipeline = pipeline or NlpPipeline()
        self.threshold = (
            threshold
            if threshold is not None
            else settings.knowledge_graph_relationship_threshold
        )

    # ── Entry points ─────────────────────────────────────────────

    def extract(self, text: str) -> list[ExtractedRelation]:
        """Extract every relation in `text`.

        Returns an empty list when spaCy is unavailable, so a missing model
        costs the graph but never the ingestion.
        """
        doc = self._pipeline.parse(text)
        if doc is None:
            return []

        relations: list[ExtractedRelation] = []
        for sentence in sentences(doc):
            relations.extend(self.extract_from_sentence(sentence))
        return relations

    def extract_from_sentence(self, sentence) -> list[ExtractedRelation]:  # noqa: ANN001
        """Extract relations from one parsed sentence span."""
        entities = self._entity_index(sentence)
        relations: list[ExtractedRelation] = []

        for token in sentence:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            relations.extend(self._from_predicate(token, sentence, entities))

        return [r for r in relations if r.confidence >= self.threshold or r.polarity is Polarity.NEGATIVE]

    # ── Predicate handling ───────────────────────────────────────

    def _from_predicate(self, verb, sentence, entities) -> list[ExtractedRelation]:  # noqa: ANN001
        relation_type, predicate, passive = self._classify_predicate(verb)
        subjects = self._subjects(verb, sentence)
        objects = self._objects(verb, passive)

        if not subjects or not objects:
            return []

        negated = is_negated(verb)
        results: list[ExtractedRelation] = []

        # "Helios is used by Engineering" means Engineering USES Helios: a type
        # that is its own passive inverse can only record the flip by swapping.
        swap = passive and swaps_arguments_when_passive(relation_type)

        for subject_token, from_relcl in subjects:
            for object_token in objects:
                first, second = (
                    (object_token, subject_token) if swap else (subject_token, object_token)
                )
                relation = self._build(
                    first,
                    second,
                    relation_type,
                    predicate,
                    sentence,
                    entities,
                    negated=negated,
                    from_relcl=from_relcl,
                    passive=passive,
                )
                if relation is not None:
                    results.append(relation)

        return results

    def _classify_predicate(self, verb) -> tuple[RelationType, str, bool]:  # noqa: ANN001
        """Determine the relation type, predicate wording and voice."""
        passive = any(child.dep_ == "auxpass" for child in verb.children) or any(
            child.dep_ == "nsubjpass" for child in verb.children
        )

        # Copular predicates: the meaning sits in the complement, not "be".
        complement = ""
        for child in verb.children:
            if child.dep_ in ("acomp", "attr", "oprd") and child.lower_ in _COPULAR_COMPLEMENTS:
                preposition = next(
                    (g.lower_ for g in child.children if g.dep_ == "prep"), ""
                )
                complement = f"{child.lower_} {preposition}".strip()
                break

        preposition = next((c.lower_ for c in verb.children if c.dep_ == "prep"), "")
        relation_type, predicate = classify(
            verb.lemma_.lower(), complement or preposition, passive=passive
        )
        return relation_type, predicate, passive

    # ── Arguments ────────────────────────────────────────────────

    def _subjects(self, verb, sentence) -> list[tuple[object, bool]]:  # noqa: ANN001
        """Subjects of the clause, each flagged if resolved through a relative.

        A relative pronoun ("who", "which", "that") is replaced by the noun it
        modifies, which is what makes the second half of "Priya reports to the
        Head of Finance, who reports to the CFO" extractable.
        """
        found: list[tuple[object, bool]] = []

        for child in verb.children:
            if child.dep_ not in _SUBJECT_DEPS:
                continue
            if child.tag_ == "WP" or child.lower_ in ("who", "which", "that"):
                antecedent = self._relative_antecedent(verb)
                if antecedent is not None:
                    found.append((antecedent, True))
                continue
            found.append((child, False))
            found.extend((conjunct, False) for conjunct in self._conjuncts(child))

        # A relative clause with no explicit subject still modifies a noun.
        if not found and verb.dep_ == "relcl":
            antecedent = self._relative_antecedent(verb)
            if antecedent is not None:
                found.append((antecedent, True))

        # Conjoined clause whose subject the parser attached elsewhere:
        # "...part of Finance and Finance is managed by Priya" parses the
        # second "Finance" as a conjunct rather than as the subject of
        # "managed". Recover it from the nearest preceding noun.
        if not found and verb.dep_ == "conj":
            recovered = self._nearest_preceding_noun(verb, sentence)
            if recovered is not None:
                found.append((recovered, True))

        return found

    @staticmethod
    def _nearest_preceding_noun(verb, sentence):  # noqa: ANN001
        """Nearest noun to the left of `verb`, stopping at the previous verb."""
        for index in range(verb.i - 1, sentence.start - 1, -1):
            token = sentence.doc[index]
            if token.pos_ in ("VERB",):
                return None
            if token.pos_ in ("NOUN", "PROPN"):
                return token
        return None

    def _objects(self, verb, passive: bool) -> list[object]:  # noqa: ANN001
        """Objects of the clause, following prepositions and agents."""
        found: list[object] = []

        for child in verb.children:
            # "is managed by Priya" - the agent is the logical object.
            if passive and child.dep_ == "agent":
                found.extend(g for g in child.children if g.dep_ == "pobj")
                continue

            if child.dep_ in _OBJECT_DEPS:
                # "is part of Finance": the complement's own prep holds the target.
                if child.lower_ in _COPULAR_COMPLEMENTS:
                    for grandchild in child.children:
                        if grandchild.dep_ == "prep":
                            found.extend(g for g in grandchild.children if g.dep_ == "pobj")
                    continue
                found.append(child)
                found.extend(self._conjuncts(child))

            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        found.append(grandchild)
                        found.extend(self._conjuncts(grandchild))

        return found

    @staticmethod
    def _conjuncts(token) -> list[object]:  # noqa: ANN001
        return [child for child in token.children if child.dep_ == "conj"]

    @staticmethod
    def _relative_antecedent(verb):  # noqa: ANN001
        """The noun a relative clause modifies."""
        head = verb.head
        return head if head is not verb else None

    # ── Span expansion ───────────────────────────────────────────

    def _argument_span(self, token, sentence):  # noqa: ANN001
        """Expand a token into the noun phrase that names its entity.

        "Head" becomes "Head of Finance"; "team" becomes "the engineering
        team". Bounded so a runaway subtree cannot swallow a whole clause.
        """
        doc = sentence.doc
        start, end = token.i, token.i + 1

        for child in token.children:
            # Determiners, adjectives and compounds belong to the phrase.
            if child.dep_ in ("compound", "amod", "nmod", "poss", "nummod"):
                start = min(start, child.left_edge.i)
                end = max(end, child.right_edge.i + 1)

        # "Head of Finance", "member of the board" - absorb the of-phrase.
        if token.lower_ in _OF_ABSORBING_HEADS:
            for child in token.children:
                if child.dep_ == "prep" and child.lower_ == "of":
                    end = max(end, child.right_edge.i + 1)

        if end - start > _MAX_ARGUMENT_TOKENS:
            end = start + _MAX_ARGUMENT_TOKENS

        span = doc[start:end]
        # Trim leading determiners so "the Finance department" keys on the noun.
        while len(span) > 1 and span[0].pos_ == "DET":
            span = doc[span.start + 1 : span.end]
        return span

    def _entity_index(self, sentence) -> dict[int, ExtractedEntity]:  # noqa: ANN001
        """Map token index -> the NER entity covering it, when there is one."""
        from app.knowledge.entities import entity_from_span

        index: dict[int, ExtractedEntity] = {}
        for ent in sentence.ents:
            extracted = entity_from_span(ent)
            if extracted is None:
                continue
            for offset in range(ent.start, ent.end):
                index[offset] = extracted
        return index

    # ── Assembly and scoring ─────────────────────────────────────

    def _build(
        self,
        subject_token,  # noqa: ANN001
        object_token,  # noqa: ANN001
        relation_type: RelationType,
        predicate: str,
        sentence,  # noqa: ANN001
        entities: dict[int, ExtractedEntity],
        negated: bool,
        from_relcl: bool,
        passive: bool,
    ) -> ExtractedRelation | None:
        source = self._best_entity(subject_token, sentence, entities)
        target = self._best_entity(object_token, sentence, entities)

        if source is None or target is None:
            return None
        if source.canonical_name == target.canonical_name:
            return None
        # Pronouns and quantifiers never name a real entity.
        if is_vacuous_argument(source.text) or is_vacuous_argument(target.text):
            return None

        confidence = self._score(
            source, target, relation_type, subject_token, object_token, from_relcl, passive
        )

        return ExtractedRelation(
            source=source,
            target=target,
            type=relation_type,
            predicate=predicate,
            confidence=round(confidence, 3),
            polarity=Polarity.NEGATIVE if negated else Polarity.POSITIVE,
            sentence=sentence.text.strip(),
            sentence_start=sentence.start_char,
        )

    def _best_entity(self, token, sentence, entities):  # noqa: ANN001
        """Choose between the NER span and the full noun phrase.

        NER is usually right, but a general model clips enterprise names: it
        tags "Project Atlas" as just "Atlas" (PERSON). When expanding the noun
        phrase yields one of the specific enterprise types, that reading is
        both longer and better, so it wins.
        """
        from app.knowledge.entities import ENTERPRISE_TYPES, entity_from_span

        from_ner = entities.get(token.i)
        from_span = entity_from_span(self._argument_span(token, sentence))

        if from_span is not None and from_span.type in ENTERPRISE_TYPES:
            return from_span
        return from_ner or from_span

    @staticmethod
    def _score(
        source: ExtractedEntity,
        target: ExtractedEntity,
        relation_type: RelationType,
        subject_token,  # noqa: ANN001
        object_token,  # noqa: ANN001
        from_relcl: bool,
        passive: bool,
    ) -> float:
        """Deterministic confidence from real extraction signals.

        Every term corresponds to something observed in the parse - there are
        no arbitrary constants standing in for "seems right".
        """
        score = 0.45

        # A predicate that matched the taxonomy is far stronger evidence than
        # an unclassified verb kept as RELATED_TO.
        if relation_type is not RelationType.RELATED_TO:
            score += 0.25

        # Both arguments recognised as real, typed entities.
        typed = sum(
            1 for entity in (source, target) if entity.type is not EntityType.OTHER
        )
        score += 0.10 * typed

        # Grammatical directness: subject and object hanging off the same verb.
        if subject_token.head is object_token.head:
            score += 0.05

        # Proximity - distant arguments are more often mis-attached.
        distance = abs(subject_token.i - object_token.i)
        if distance <= 6:
            score += 0.05
        elif distance > 15:
            score -= 0.10

        # Resolved through a relative pronoun: correct often, but a step removed.
        if from_relcl:
            score -= 0.05

        # A passive clause whose direction matters is easier to get backwards.
        if passive and relation_type in DIRECTIONAL_TYPES:
            score -= 0.05

        return max(0.0, min(1.0, score))


__all__ = ["DependencyRelationshipExtractor"]
