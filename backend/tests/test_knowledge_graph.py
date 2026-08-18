"""Tests for knowledge graph extraction and persistence.

Everything here runs locally: spaCy's `en_core_web_sm` parser and an in-memory
SQLite database. No LLM, no network, no API key.
"""

from __future__ import annotations

import pytest

from app.domain.knowledge import (
    DocumentKnowledge,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    Polarity,
    RelationType,
)
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.negation import is_vacuous_argument
from app.knowledge.nlp import NlpPipeline
from app.knowledge.normalization import EntityResolver, canonical_name
from app.knowledge.taxonomy import classify
from app.repositories.sqlite_graph import SqliteGraphRepository, entity_id

pytestmark = pytest.mark.skipif(
    not NlpPipeline().is_available,
    reason="spaCy model unavailable; run `python -m spacy download en_core_web_sm`",
)


@pytest.fixture(scope="module")
def extractor() -> KnowledgeExtractor:
    """One extractor for the module: loading spaCy per test would be wasteful."""
    return KnowledgeExtractor()


def edges(extractor: KnowledgeExtractor, text: str) -> set[tuple[str, str, str]]:
    """Positive edges as (source, TYPE, target), lower-cased for comparison."""
    knowledge = extractor.extract(text)
    return {
        (r.source.text.lower(), r.type.value, r.target.text.lower())
        for r in knowledge.relations
        if r.polarity is Polarity.POSITIVE
    }


def types_of(extractor: KnowledgeExtractor, text: str) -> set[str]:
    return {r.type.value for r in extractor.extract(text).relations}


class TestRequiredExamples:
    """The acceptance cases the rewrite was specified against."""

    def test_1_works_in(self, extractor):
        assert ("priya sharma", "WORKS_IN", "finance") in edges(
            extractor, "Priya Sharma works in Finance."
        )

    def test_2_reports_to_a_role(self, extractor):
        assert ("priya", "REPORTS_TO", "cfo") in edges(extractor, "Priya reports to the CFO.")

    def test_3_two_relations_in_one_sentence(self, extractor):
        """The old extractor discarded any sentence with two linking phrases."""
        found = edges(
            extractor, "Priya reports to the Head of Finance, who reports to the CFO."
        )

        assert ("priya", "REPORTS_TO", "head of finance") in found
        assert ("head of finance", "REPORTS_TO", "cfo") in found

    def test_4_manages(self, extractor):
        assert ("john", "MANAGES", "engineering team") in edges(
            extractor, "John manages the engineering team."
        )

    def test_5_acquired(self, extractor):
        assert ("acme", "ACQUIRED", "xyz") in edges(extractor, "Acme acquired XYZ in 2025.")

    def test_6_explicit_negation_is_not_asserted(self, extractor):
        """"John does not report to Mary" must not create a positive edge."""
        assert not edges(extractor, "John does not report to Mary.")

    def test_6_negation_is_recorded_with_polarity(self, extractor):
        relations = extractor.extract("John does not report to Mary.").relations

        assert relations, "the statement should be recorded, just not asserted"
        assert all(r.polarity is Polarity.NEGATIVE for r in relations)

    def test_7_negative_quantifier_produces_nothing(self, extractor):
        """The exact false positive the previous implementation produced."""
        found = edges(
            extractor, "Valve is a flat organization where nobody reports to anyone else."
        )

        assert found == set()

    def test_8_conjoined_clauses_both_extract(self, extractor):
        found = edges(
            extractor,
            "The payroll team is part of Finance and Finance is managed by Priya.",
        )

        assert ("payroll team", "PART_OF", "finance") in found
        assert ("finance", "MANAGED_BY", "priya") in found


class TestPassiveVoice:
    def test_passive_keeps_the_subject_and_inverts_the_type(self, extractor):
        assert ("project alpha", "OWNED_BY", "engineering department") in edges(
            extractor, "Project Alpha is owned by the Engineering department."
        )

    def test_a_self_inverse_type_swaps_its_arguments_instead(self, extractor):
        """"Helios is used by Engineering" means Engineering USES Helios."""
        assert ("engineering team", "USES", "helios platform") in edges(
            extractor, "The Helios platform is used by the Engineering team."
        )

    def test_location_predicates_are_typed(self, extractor):
        assert ("acme corporation", "LOCATED_IN", "berlin") in edges(
            extractor, "Acme Corporation is headquartered in Berlin."
        )


class TestHeadings:
    """Headings carry no terminal punctuation and would otherwise merge into
    the sentence that follows, poisoning its subject."""

    def test_a_heading_does_not_become_part_of_the_next_subject(self, extractor):
        found = edges(
            extractor,
            "Acme Corporation Handbook\n\nLeadership\nPriya Sharma works in Finance.",
        )

        assert ("priya sharma", "WORKS_IN", "finance") in found

    def test_evidence_offsets_survive_the_rewrite(self, extractor):
        """Inserted punctuation must not shift recorded sentence positions."""
        text = "Leadership\nPriya Sharma works in Finance."
        relation = extractor.extract(text).relations[0]

        assert text[relation.sentence_start :].startswith("Priya Sharma works in Finance")


class TestNegation:
    @pytest.mark.parametrize(
        "sentence",
        [
            "John does not report to Mary.",
            "John never reports to Mary.",
            "Nobody reports to Mary.",
            "John cannot report to Mary.",
        ],
    )
    def test_negated_sentences_assert_nothing(self, extractor, sentence):
        assert not edges(extractor, sentence)

    @pytest.mark.parametrize(
        "text", ["anyone else", "nobody", "it", "they", "someone", "each other", ""]
    )
    def test_pronouns_are_never_entities(self, text):
        assert is_vacuous_argument(text)

    def test_a_real_name_is_not_vacuous(self):
        assert not is_vacuous_argument("Priya Sharma")


class TestEntityTyping:
    """A general NER model labels enterprise nouns as ORG; the head noun is
    more specific and must win."""

    def entities(self, extractor, text) -> dict[str, str]:
        return {e.text.lower(): e.type.value for e in extractor.extract(text).entities}

    def test_a_job_title_is_a_role_not_an_org(self, extractor):
        found = self.entities(extractor, "Priya reports to the CFO.")

        assert found.get("cfo") == "ROLE"

    def test_a_team_is_a_department(self, extractor):
        found = self.entities(extractor, "John manages the Engineering team.")

        assert found.get("engineering team") == "DEPARTMENT"

    def test_an_unrecognised_name_is_skipped_rather_than_guessed(self, extractor):
        """`en_core_web_sm` tags uncommon names as NOUN, not PROPN.

        Precision over recall: an unknown capitalised word is left out instead
        of being invented as an entity. A larger model recognises more names.
        """
        assert not edges(extractor, "Ravi manages the Engineering team.")

    def test_a_project_is_typed_as_a_project(self, extractor):
        found = self.entities(extractor, "Project Atlas is owned by the Engineering team.")

        assert found.get("project atlas") == "PROJECT"

    def test_a_person_is_still_a_person(self, extractor):
        found = self.entities(extractor, "Priya Sharma works in Finance.")

        assert found.get("priya sharma") == "PERSON"

    def test_a_place_is_still_a_location(self, extractor):
        found = self.entities(extractor, "Acme Corporation is headquartered in Berlin.")

        assert found.get("berlin") == "LOCATION"


class TestPrecision:
    """Precision over recall: silence beats invention."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "Nothing relational here.",
            "The weather was pleasant yesterday.",
            "This document describes the process.",
        ],
    )
    def test_sentences_without_relations_produce_none(self, extractor, sentence):
        assert not edges(extractor, sentence)

    def test_no_self_loops(self, extractor):
        for relation in extractor.extract("Finance is part of Finance.").relations:
            assert relation.source.canonical_name != relation.target.canonical_name

    def test_confidence_is_bounded_and_deterministic(self, extractor):
        text = "Priya Sharma works in Finance."
        first = extractor.extract(text).relations
        second = extractor.extract(text).relations

        assert [r.confidence for r in first] == [r.confidence for r in second]
        assert all(0.0 <= r.confidence <= 1.0 for r in first)

    def test_a_higher_threshold_keeps_fewer_relations(self):
        text = "Priya reports to the CFO and Acme acquired XYZ."
        permissive = len(KnowledgeExtractor(threshold=0.1).extract(text).relations)
        strict = len(KnowledgeExtractor(threshold=0.99).extract(text).relations)

        assert strict < permissive


class TestTaxonomy:
    @pytest.mark.parametrize(
        "lemma,preposition,expected",
        [
            ("report", "to", RelationType.REPORTS_TO),
            ("work", "in", RelationType.WORKS_IN),
            ("manage", "", RelationType.MANAGES),
            ("acquire", "", RelationType.ACQUIRED),
            ("be", "part of", RelationType.PART_OF),
        ],
    )
    def test_known_predicates_map_to_types(self, lemma, preposition, expected):
        assert classify(lemma, preposition)[0] is expected

    def test_unknown_predicates_fall_back_to_related_to(self):
        relation_type, predicate = classify("collaborate", "with")

        assert relation_type is RelationType.RELATED_TO
        assert predicate == "collaborate with"

    def test_passive_selects_the_inverse_type(self):
        assert classify("manage", "", passive=True)[0] is RelationType.MANAGED_BY


class TestNormalization:
    @pytest.mark.parametrize(
        "raw",
        ["Microsoft Corporation", "Microsoft Corp.", "Microsoft", "the Microsoft"],
    )
    def test_corporate_suffixes_collapse_to_one_key(self, raw):
        assert canonical_name(raw, EntityType.ORG) == "microsoft"

    def test_a_person_surname_is_never_trimmed(self):
        """The suffix list must not eat part of a person's name."""
        assert canonical_name("Priya Group", EntityType.PERSON) == "priya group"

    def test_possessives_and_articles_are_stripped(self):
        assert (
            canonical_name("The Finance Department's", EntityType.DEPARTMENT)
            == "finance department"
        )

    def test_a_department_is_not_collapsed_into_its_parent_name(self):
        """Conservative by design: "Finance Department" may not be "Finance"."""
        assert canonical_name("Finance Department", EntityType.DEPARTMENT) != canonical_name(
            "Finance", EntityType.ORG
        )

    def test_the_same_name_under_two_types_is_two_nodes(self):
        """Type is part of identity, so a person and a company never merge."""
        assert entity_id(EntityType.ORG.value, "microsoft") != entity_id(
            EntityType.PERSON.value, "microsoft"
        )

    def test_resolver_does_not_merge_distinct_names(self):
        resolver = EntityResolver()

        assert resolver.resolve("Priya Sharma", EntityType.PERSON) != resolver.resolve(
            "Ravi Kumar", EntityType.PERSON
        )

    def test_resolver_merges_a_variant_spelling(self):
        resolver = EntityResolver()
        first = resolver.resolve("Microsoft Corporation", EntityType.ORG)
        second = resolver.resolve("Microsoft Corp", EntityType.ORG)

        assert first == second


# ── Persistence ──────────────────────────────────────────────────


def entity(name: str, entity_type: EntityType = EntityType.PERSON) -> ExtractedEntity:
    return ExtractedEntity(text=name, canonical_name=name.lower(), type=entity_type)


def relation(source: str, relation_type: RelationType, target: str, **kwargs) -> ExtractedRelation:
    return ExtractedRelation(
        source=entity(source),
        target=entity(target, EntityType.ORG),
        type=relation_type,
        predicate=kwargs.pop("predicate", "reports to"),
        confidence=kwargs.pop("confidence", 0.9),
        sentence=kwargs.pop("sentence", f"{source} reports to {target}."),
        **kwargs,
    )


@pytest.fixture
def repository() -> SqliteGraphRepository:
    store = SqliteGraphRepository(":memory:")
    yield store
    store.close()


@pytest.fixture
def knowledge() -> DocumentKnowledge:
    return DocumentKnowledge(
        entities=[entity("Priya"), entity("Finance", EntityType.ORG)],
        relations=[relation("Priya", RelationType.REPORTS_TO, "Finance")],
    )


class TestPersistence:
    def test_saves_entities_and_relationships(self, repository, knowledge):
        entities, relationships = repository.save_document("doc-1", knowledge)

        assert (entities, relationships) == (2, 1)

    def test_the_graph_survives_a_new_connection(self, tmp_path, knowledge):
        path = tmp_path / "graph.db"
        first = SqliteGraphRepository(str(path))
        first.save_document("doc-1", knowledge)
        first.close()

        second = SqliteGraphRepository(str(path))
        try:
            assert second.stats().total_relationships == 1
        finally:
            second.close()

    def test_reingesting_a_document_does_not_duplicate(self, repository, knowledge):
        repository.save_document("doc-1", knowledge)
        repository.save_document("doc-1", knowledge)

        stats = repository.stats()
        assert stats.total_entities == 2
        assert stats.total_relationships == 1

    def test_the_same_claim_in_two_documents_is_one_edge_with_two_evidences(
        self, repository, knowledge
    ):
        repository.save_document("doc-1", knowledge)
        repository.save_document(
            "doc-2",
            DocumentKnowledge(
                entities=knowledge.entities,
                relations=[
                    relation(
                        "Priya",
                        RelationType.REPORTS_TO,
                        "Finance",
                        sentence="A second document says Priya reports to Finance.",
                    )
                ],
            ),
        )

        assert repository.stats().total_relationships == 1
        view = repository.graph()
        detail = repository.entity(view.relationships[0].source_id)
        assert detail.relationships[0].evidence_count == 2

    def test_evidence_records_where_a_relationship_came_from(self, repository, knowledge):
        repository.save_document("doc-1", knowledge)

        detail = repository.entity(repository.graph().relationships[0].source_id)
        evidence = detail.relationships[0].evidence[0]

        assert evidence.document_id == "doc-1"
        assert "Priya reports to Finance" in evidence.sentence

    def test_deleting_a_document_removes_its_contribution(self, repository, knowledge):
        repository.save_document("doc-1", knowledge)
        repository.delete_document("doc-1")

        stats = repository.stats()
        assert stats.total_entities == 0
        assert stats.total_relationships == 0

    def test_deleting_one_document_keeps_another_documents_edges(self, repository, knowledge):
        repository.save_document("doc-1", knowledge)
        repository.save_document("doc-2", knowledge)

        repository.delete_document("doc-1")

        assert repository.stats().total_relationships == 1

    def test_negative_relationships_are_stored_but_not_counted(self, repository):
        repository.save_document(
            "doc-1",
            DocumentKnowledge(
                entities=[entity("John"), entity("Mary", EntityType.ORG)],
                relations=[
                    relation(
                        "John", RelationType.REPORTS_TO, "Mary", polarity=Polarity.NEGATIVE
                    )
                ],
            ),
        )

        assert repository.stats().total_relationships == 0
        assert repository.graph().relationships == []


class TestGraphReads:
    @pytest.fixture
    def populated(self, repository) -> SqliteGraphRepository:
        repository.save_document(
            "doc-1",
            DocumentKnowledge(
                entities=[
                    entity("Priya"),
                    entity("Finance", EntityType.ORG),
                    entity("Ravi"),
                ],
                relations=[
                    relation("Priya", RelationType.REPORTS_TO, "Finance"),
                    relation("Ravi", RelationType.WORKS_IN, "Finance", confidence=0.8),
                ],
            ),
        )
        return repository

    def test_stats_break_down_by_type(self, populated):
        stats = populated.stats()

        assert stats.total_entities == 3
        assert stats.relationships_by_type["REPORTS_TO"] == 1
        assert stats.documents_with_graph_data == 1

    def test_search_finds_an_entity_by_name(self, populated):
        assert [e.display_name for e in populated.search("fina")] == ["Finance"]

    def test_search_is_empty_for_a_blank_query(self, populated):
        assert populated.search("   ") == []

    def test_filtering_by_relationship_type(self, populated):
        view = populated.graph(relationship_type="WORKS_IN")

        assert len(view.relationships) == 1
        assert view.relationships[0].type is RelationType.WORKS_IN

    def test_filtering_by_document(self, populated):
        assert populated.graph(document_id="doc-1").relationships
        assert not populated.graph(document_id="absent").relationships

    def test_a_confidence_floor_excludes_weaker_edges(self, populated):
        assert len(populated.graph(min_confidence=0.85).relationships) == 1

    def test_a_limit_marks_the_view_truncated(self, populated):
        view = populated.graph(limit=1)

        assert len(view.relationships) == 1
        assert view.truncated is True

    def test_entity_detail_lists_neighbors_and_sources(self, populated):
        finance = next(e for e in populated.search("finance"))
        detail = populated.entity(finance.id)

        assert {n.display_name for n in detail.neighbors} == {"Priya", "Ravi"}
        assert detail.entity.source_documents == ["doc-1"]

    def test_entity_detail_is_none_for_an_unknown_id(self, populated):
        assert populated.entity("e_missing") is None

    def test_neighbors_expand_from_one_entity(self, populated):
        priya = next(e for e in populated.search("priya"))

        view = populated.neighbors(priya.id, depth=1)

        assert {n.display_name for n in view.entities} == {"Priya", "Finance"}

    def test_depth_two_reaches_a_second_hop(self, populated):
        priya = next(e for e in populated.search("priya"))

        view = populated.neighbors(priya.id, depth=2)

        # Priya -> Finance <- Ravi
        assert "Ravi" in {n.display_name for n in view.entities}
