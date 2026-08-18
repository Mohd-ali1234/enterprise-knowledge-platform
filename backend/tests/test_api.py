"""End-to-end tests over the HTTP layer.

Infrastructure is substituted through the composition root, so these exercise
the real routers, services and pipelines.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.application.file_storage import UploadStorage
from app.application.ingestion_service import IngestionService
from app.application.knowledge_service import KnowledgeService
from app.domain.documents import ExtractedText
from app.indexing.indexer import DocumentIndexer
from app.ingestion.extraction import ExtractorRegistry
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.ingestion.recursive_chunking import RecursiveChunker
from app.main import create_app
from app.query.generation import AnswerGeneratorRegistry, ExtractiveAnswerGenerator
from app.query.pipeline import QueryPipeline
from app.query.reranking import LexicalReranker
from app.query.rewriting import HeuristicQueryRewriter
from app.repositories.in_memory import InMemoryDocumentRepository
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever
from tests.conftest import FakeEmbedder, FakeEntityExtractor

DOCUMENT_TEXT = (
    "1. LEAVE POLICY\n"
    "Employees accrue paid leave every month.\n\n"
    "2. EXPENSES\n"
    "Travel expenses are reimbursed within thirty days of submission.\n"
)


class TextFileExtractor:
    supported_extensions = frozenset({".txt"})

    def extract(self, file_path):  # noqa: ANN001
        return ExtractedText(text=file_path.read_text(encoding="utf-8"), page_count=1)


@pytest.fixture
def client(tmp_path, vector_store):
    embedder = FakeEmbedder()
    # Taken from the shared fixture so a test can seed or inspect the index.
    store = vector_store
    retriever = HybridRetriever(
        dense=DenseRetriever(embedder, store),
        sparse=SparseRetriever(store),
    )
    ingestion = IngestionService(
        pipeline=DocumentIngestionPipeline(
            extractors=ExtractorRegistry([TextFileExtractor()]),
            recursive_chunker=RecursiveChunker(chunk_size=120, chunk_overlap=20),
            entity_extractor=FakeEntityExtractor(),
        ),
        indexer=DocumentIndexer(
            embedder, store, on_index=retriever.invalidate_keyword_index
        ),
        repository=InMemoryDocumentRepository(),
        storage=UploadStorage(tmp_path),
        store=store,
        on_index_changed=retriever.invalidate_keyword_index,
    )
    knowledge = KnowledgeService(
        QueryPipeline(
            retriever=retriever,
            rewriter=HeuristicQueryRewriter(),
            reranker=LexicalReranker(),
            generators=AnswerGeneratorRegistry(default=ExtractiveAnswerGenerator()),
        )
    )

    app = create_app()
    app.dependency_overrides[dependencies.get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[dependencies.get_knowledge_service] = lambda: knowledge

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def upload(client, name: str = "handbook.txt", text: str = DOCUMENT_TEXT):
    return client.post(
        "/api/v1/documents/upload-and-index",
        files={"file": (name, io.BytesIO(text.encode()), "text/plain")},
    )


class TestSystemEndpoints:
    def test_health(self, client):
        response = client.get("/api/v1/system/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestDocumentListing:
    """`GET /documents` is read from the index, not the in-memory repository."""

    def test_empty_index_lists_nothing(self, client):
        body = client.get("/api/v1/documents").json()

        assert body == {"documents": [], "total_documents": 0, "total_chunks": 0}

    def test_lists_an_uploaded_document_with_its_name_and_type(self, client):
        upload(client, name="policy.txt")

        body = client.get("/api/v1/documents").json()

        assert body["total_documents"] == 1
        assert body["documents"][0]["file_name"] == "policy.txt"
        assert body["documents"][0]["file_type"] == "txt"
        assert body["documents"][0]["chunk_count"] > 0

    def test_chunk_counts_add_up_to_the_total(self, client):
        upload(client, name="a.txt")
        upload(client, name="b.txt")

        body = client.get("/api/v1/documents").json()

        assert body["total_documents"] == 2
        assert body["total_chunks"] == sum(d["chunk_count"] for d in body["documents"])

    def test_reindexing_the_same_document_does_not_duplicate_it(self, client):
        """Chunks are upserted by a stable id, so a re-upload replaces them."""
        first = upload(client, name="a.txt").json()
        client.post(f"/api/v1/documents/{first['document_id']}/index")

        body = client.get("/api/v1/documents").json()

        assert body["total_documents"] == 1

    def test_documents_are_listed_newest_first(self, client):
        upload(client, name="older.txt")
        upload(client, name="newer.txt")

        names = [d["file_name"] for d in client.get("/api/v1/documents").json()["documents"]]

        assert names.index("newer.txt") < names.index("older.txt")

    def test_a_document_indexed_without_a_filename_falls_back_to_its_id(
        self, client, vector_store
    ):
        """Chunks written before filenames were recorded must still be listed."""
        from app.domain.retrieval import VectorRecord

        vector_store.upsert(
            [
                VectorRecord(
                    id="legacy_0",
                    text="older chunk",
                    embedding=[0.0] * 16,
                    metadata={"document_id": "legacy-id", "chunk_id": 0},
                )
            ]
        )

        body = client.get("/api/v1/documents").json()

        assert body["documents"][0]["file_name"] == "legacy-id"
        assert body["documents"][0]["file_type"] == ""


class TestDocumentDeletion:
    def test_deleting_removes_it_from_the_listing(self, client):
        document_id = upload(client, name="a.txt").json()["document_id"]
        upload(client, name="b.txt")

        client.request("DELETE", f"/api/v1/documents/{document_id}")

        names = [d["file_name"] for d in client.get("/api/v1/documents").json()["documents"]]
        assert names == ["b.txt"]

    def test_deleting_reports_how_many_chunks_went(self, client):
        document = upload(client, name="a.txt").json()

        body = client.request("DELETE", f"/api/v1/documents/{document['document_id']}").json()

        assert body["documents_deleted"] == 1
        assert body["chunks_deleted"] == document["total_chunks"]

    def test_bulk_delete_removes_every_selected_document(self, client):
        first = upload(client, name="a.txt").json()["document_id"]
        second = upload(client, name="b.txt").json()["document_id"]
        upload(client, name="keep.txt")

        body = client.post(
            "/api/v1/documents/delete", json={"document_ids": [first, second]}
        ).json()

        assert body["documents_deleted"] == 2
        names = [d["file_name"] for d in client.get("/api/v1/documents").json()["documents"]]
        assert names == ["keep.txt"]

    def test_deleting_an_unknown_id_is_not_an_error(self, client):
        """Deleting something already gone is the outcome the caller wanted."""
        body = client.request("DELETE", "/api/v1/documents/does-not-exist").json()

        assert body["documents_deleted"] == 0
        assert body["chunks_deleted"] == 0

    def test_deleted_content_stops_being_retrievable(self, client):
        """The cached keyword index must not keep serving deleted chunks."""
        document_id = upload(client).json()["document_id"]
        assert client.post("/api/v1/knowledge/retrieve", json={"query": "leave policy"}).json()[
            "results"
        ]

        client.request("DELETE", f"/api/v1/documents/{document_id}")

        after = client.post("/api/v1/knowledge/retrieve", json={"query": "leave policy"}).json()
        assert after["results"] == []

    def test_an_empty_id_list_is_rejected(self, client):
        assert client.post("/api/v1/documents/delete", json={"document_ids": []}).status_code == 422


class TestDocumentEndpoints:
    def test_upload_and_index_reports_counts(self, client):
        response = upload(client)

        assert response.status_code == 200
        body = response.json()
        assert body["file_name"] == "handbook.txt"
        assert body["total_chunks"] > 0
        assert body["chunks_indexed"] == body["total_chunks"]

    def test_two_step_upload_then_index(self, client):
        uploaded = client.post(
            "/api/v1/documents/upload",
            files={"file": ("handbook.txt", io.BytesIO(DOCUMENT_TEXT.encode()), "text/plain")},
        )
        assert uploaded.status_code == 200
        document_id = uploaded.json()["document_id"]

        indexed = client.post(f"/api/v1/documents/{document_id}/index")

        assert indexed.status_code == 200
        assert indexed.json()["chunks_indexed"] > 0

    def test_document_detail(self, client):
        document_id = upload(client).json()["document_id"]

        response = client.get(f"/api/v1/documents/{document_id}")

        assert response.status_code == 200
        assert response.json()["filename"] == "handbook.txt"

    def test_unknown_document_is_404(self, client):
        response = client.get("/api/v1/documents/does-not-exist")

        assert response.status_code == 404
        assert response.json()["error"] == "DocumentNotFoundError"

    def test_indexing_an_unknown_document_is_404(self, client):
        assert client.post("/api/v1/documents/nope/index").status_code == 404

    def test_unsupported_file_type_is_415(self, client):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("photo.png", io.BytesIO(b"\x89PNG"), "image/png")},
        )

        assert response.status_code == 415
        assert response.json()["error"] == "UnsupportedDocumentError"

    def test_traversal_in_the_filename_is_neutralised(self, client, tmp_path):
        response = upload(client, name="../../escape.txt")

        assert response.status_code == 200
        assert not (tmp_path.parent.parent / "escape.txt").exists()
        assert list(tmp_path.glob("escape-*.txt"))


class TestKnowledgeEndpoints:
    def test_retrieve_returns_ranked_chunks(self, client):
        upload(client)

        response = client.post(
            "/api/v1/knowledge/retrieve", json={"query": "leave", "top_k": 3}
        )

        assert response.status_code == 200
        results = response.json()["results"]
        assert results
        assert [result["rank"] for result in results] == list(range(1, len(results) + 1))

    def test_retrieve_honours_metadata_filters(self, client):
        document_id = upload(client).json()["document_id"]

        response = client.post(
            "/api/v1/knowledge/retrieve",
            json={"query": "leave", "filters": {"document_id": document_id}},
        )

        assert response.status_code == 200
        assert all(
            result["metadata"]["document_id"] == document_id
            for result in response.json()["results"]
        )

    def test_retrieve_ignores_openapi_placeholder_filters(self, client):
        upload(client)

        response = client.post(
            "/api/v1/knowledge/retrieve",
            json={"query": "leave", "filters": {"additionalProp1": {}}},
        )

        assert response.status_code == 200
        assert response.json()["results"]

    def test_ask_returns_a_grounded_answer_with_citations(self, client):
        upload(client)

        response = client.post(
            "/api/v1/knowledge/ask", json={"query": "What is the leave policy?"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["generator"] == "extractive"
        assert body["citations"]
        assert body["route"]["route"] in {"direct", "local_llm", "online_llm"}
        assert body["understanding"]["intent"] == "question_answering"

    def test_ask_with_rewriting_falls_back_without_ollama(self, client):
        upload(client)

        response = client.post(
            "/api/v1/knowledge/ask",
            json={"query": "hey dude tell me about leave", "rewrite": True},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["rewritten_queries"]) == 3
        assert body["retrieval_query"] != body["query"]

    def test_ask_on_an_empty_index_still_succeeds(self, client):
        response = client.post("/api/v1/knowledge/ask", json={"query": "anything"})

        assert response.status_code == 200
        assert response.json()["citations"] == []

    def test_blank_query_is_rejected(self, client):
        assert client.post("/api/v1/knowledge/ask", json={"query": ""}).status_code == 422
