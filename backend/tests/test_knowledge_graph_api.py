"""End-to-end tests over the knowledge graph HTTP layer.

The graph repository is substituted through the composition root with an
in-memory SQLite database, so these exercise the real routers and service.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.application.graph_service import GraphService
from app.domain.knowledge import (
    DocumentKnowledge,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    Polarity,
    RelationType,
)
from app.main import create_app
from app.repositories.sqlite_graph import SqliteGraphRepository


def entity(name: str, entity_type: EntityType) -> ExtractedEntity:
    return ExtractedEntity(text=name, canonical_name=name.lower(), type=entity_type)


def relation(
    source: str, source_type: EntityType, relation_type: RelationType,
    target: str, target_type: EntityType, confidence: float = 0.9,
) -> ExtractedRelation:
    return ExtractedRelation(
        source=entity(source, source_type),
        target=entity(target, target_type),
        type=relation_type,
        predicate=relation_type.value.lower().replace("_", " "),
        confidence=confidence,
        sentence=f"{source} {relation_type.value.lower().replace('_', ' ')} {target}.",
    )


@pytest.fixture
def graph() -> SqliteGraphRepository:
    repository = SqliteGraphRepository(":memory:")
    repository.save_document(
        "doc-1",
        DocumentKnowledge(
            entities=[
                entity("Priya", EntityType.PERSON),
                entity("Finance", EntityType.ORG),
                entity("CFO", EntityType.ROLE),
            ],
            relations=[
                relation("Priya", EntityType.PERSON, RelationType.WORKS_IN,
                         "Finance", EntityType.ORG),
                relation("Priya", EntityType.PERSON, RelationType.REPORTS_TO,
                         "CFO", EntityType.ROLE, confidence=0.75),
            ],
        ),
    )
    yield repository
    repository.close()


@pytest.fixture
def client(graph) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_graph_service] = lambda: GraphService(graph)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


class TestReadGraph:
    def test_returns_nodes_and_edges(self, client):
        body = client.get("/api/v1/knowledge-graph").json()

        assert len(body["nodes"]) == 3
        assert len(body["edges"]) == 2

    def test_nodes_carry_a_degree_for_sizing(self, client):
        body = client.get("/api/v1/knowledge-graph").json()
        priya = next(n for n in body["nodes"] if n["label"] == "Priya")

        assert priya["degree"] == 2

    def test_filtering_by_relationship_type(self, client):
        body = client.get("/api/v1/knowledge-graph?relationship_type=WORKS_IN").json()

        assert [edge["type"] for edge in body["edges"]] == ["WORKS_IN"]

    def test_filtering_by_entity_type(self, client):
        body = client.get("/api/v1/knowledge-graph?entity_type=ROLE").json()

        assert body["edges"]
        assert any(node["type"] == "ROLE" for node in body["nodes"])

    def test_filtering_by_document(self, client):
        assert client.get("/api/v1/knowledge-graph?document_id=doc-1").json()["edges"]
        assert not client.get("/api/v1/knowledge-graph?document_id=nope").json()["edges"]

    def test_a_confidence_floor_excludes_weaker_edges(self, client):
        body = client.get("/api/v1/knowledge-graph?min_confidence=0.8").json()

        assert [edge["type"] for edge in body["edges"]] == ["WORKS_IN"]

    def test_a_limit_reports_truncation(self, client):
        body = client.get("/api/v1/knowledge-graph?limit=1").json()

        assert len(body["edges"]) == 1
        assert body["truncated"] is True

    def test_an_out_of_range_limit_is_rejected(self, client):
        assert client.get("/api/v1/knowledge-graph?limit=99999").status_code == 422


class TestStats:
    def test_reports_totals_and_breakdowns(self, client):
        body = client.get("/api/v1/knowledge-graph/stats").json()

        assert body["total_entities"] == 3
        assert body["total_relationships"] == 2
        assert body["entities_by_type"]["PERSON"] == 1
        assert body["relationships_by_type"]["WORKS_IN"] == 1
        assert body["documents_with_graph_data"] == 1


class TestSearch:
    def test_finds_an_entity_by_partial_name(self, client):
        body = client.get("/api/v1/knowledge-graph/search?q=fin").json()

        assert [e["display_name"] for e in body["entities"]] == ["Finance"]

    def test_an_empty_query_is_rejected(self, client):
        assert client.get("/api/v1/knowledge-graph/search?q=").status_code == 422

    def test_no_match_returns_an_empty_list(self, client):
        assert client.get("/api/v1/knowledge-graph/search?q=zzz").json()["entities"] == []


class TestEntityDetail:
    def entity_id(self, client, name: str) -> str:
        body = client.get(f"/api/v1/knowledge-graph/search?q={name}").json()
        return body["entities"][0]["id"]

    def test_returns_relationships_with_readable_labels(self, client):
        detail = client.get(
            f"/api/v1/knowledge-graph/entities/{self.entity_id(client, 'priya')}"
        ).json()

        assert detail["label"] == "Priya"
        targets = {r["target_label"] for r in detail["relationships"]}
        assert targets == {"Finance", "CFO"}

    def test_relationships_carry_their_evidence(self, client):
        detail = client.get(
            f"/api/v1/knowledge-graph/entities/{self.entity_id(client, 'priya')}"
        ).json()
        evidence = detail["relationships"][0]["evidence"][0]

        assert evidence["document_id"] == "doc-1"
        assert evidence["sentence"]

    def test_lists_source_documents(self, client):
        detail = client.get(
            f"/api/v1/knowledge-graph/entities/{self.entity_id(client, 'priya')}"
        ).json()

        assert detail["source_documents"] == ["doc-1"]

    def test_an_unknown_entity_is_a_404(self, client):
        assert client.get("/api/v1/knowledge-graph/entities/e_nope").status_code == 404


class TestNeighbors:
    def test_expands_around_an_entity(self, client):
        priya = client.get("/api/v1/knowledge-graph/search?q=priya").json()["entities"][0]["id"]

        body = client.get(f"/api/v1/knowledge-graph/entities/{priya}/neighbors").json()

        assert {node["label"] for node in body["nodes"]} == {"Priya", "Finance", "CFO"}

    def test_depth_is_capped(self, client):
        priya = client.get("/api/v1/knowledge-graph/search?q=priya").json()["entities"][0]["id"]

        assert (
            client.get(f"/api/v1/knowledge-graph/entities/{priya}/neighbors?depth=99").status_code
            == 422
        )


class TestNoRegression:
    """The graph is additive: existing endpoints must be unaffected."""

    def test_health_still_works(self, client):
        assert client.get("/api/v1/system/health").status_code == 200

    def test_document_formats_still_works(self, client):
        assert client.get("/api/v1/documents/formats").status_code == 200


class TestNegativePolarity:
    def test_negative_edges_are_hidden_from_the_graph(self, client, graph):
        graph.save_document(
            "doc-2",
            DocumentKnowledge(
                entities=[entity("John", EntityType.PERSON), entity("Mary", EntityType.PERSON)],
                relations=[
                    ExtractedRelation(
                        source=entity("John", EntityType.PERSON),
                        target=entity("Mary", EntityType.PERSON),
                        type=RelationType.REPORTS_TO,
                        predicate="report to",
                        confidence=0.9,
                        polarity=Polarity.NEGATIVE,
                        sentence="John does not report to Mary.",
                    )
                ],
            ),
        )

        edges = client.get("/api/v1/knowledge-graph").json()["edges"]

        assert all(edge["polarity"] == "positive" for edge in edges)
