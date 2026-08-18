"""Document ingestion endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_ingestion_service
from app.api.schemas import (
    DeleteDocumentsRequest,
    DeleteDocumentsResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    IndexedDocumentResponse,
    IndexResponse,
    IngestUrlRequest,
    ProcessAndIndexResponse,
    SupportedFormatsResponse,
    UploadResponse,
)
from app.application.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["Documents"])

IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents(service: IngestionDep) -> DocumentListResponse:
    """Every document in the vector store, newest first.

    Read from the index rather than the in-memory repository, so the list
    survives a restart and reflects what is actually searchable.
    """
    documents = service.list_indexed_documents()

    return DocumentListResponse(
        documents=[IndexedDocumentResponse.from_document(d) for d in documents],
        total_documents=len(documents),
        total_chunks=sum(d.chunk_count for d in documents),
    )


@router.get(
    "/formats",
    response_model=SupportedFormatsResponse,
    summary="List the file types ingestion accepts",
)
async def supported_formats(service: IngestionDep) -> SupportedFormatsResponse:
    """Report which extensions have a registered extractor."""
    return SupportedFormatsResponse(extensions=service.supported_extensions())


@router.post("/upload", response_model=UploadResponse, summary="Upload & process a document")
async def upload_document(service: IngestionDep, file: UploadFile = File(...)) -> UploadResponse:
    """Extract, clean, chunk and analyse a document.

    The result is held in the document repository; call
    `/{document_id}/index` next to make it searchable.
    """
    document = await service.process_upload(file.filename or "document", file.file)

    return UploadResponse(
        **DocumentSummary.from_document(document).model_dump(),
        message="Document processed. Call /{document_id}/index to embed & store.",
    )


@router.post(
    "/{document_id}/index",
    response_model=IndexResponse,
    summary="Embed & index a processed document",
)
async def index_document(document_id: str, service: IngestionDep) -> IndexResponse:
    """Embed a processed document's chunks and write them to the vector store."""
    chunks_indexed = await service.index_document(document_id)

    return IndexResponse(
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        message="Document indexed successfully. Ready for retrieval.",
    )


@router.post(
    "/upload-and-index",
    response_model=ProcessAndIndexResponse,
    summary="Upload, process and index in one step",
)
async def upload_and_index(
    service: IngestionDep,
    file: UploadFile = File(...),
) -> ProcessAndIndexResponse:
    """Run the full ingestion flow end to end."""
    document, chunks_indexed = await service.process_and_index(
        file.filename or "document", file.file
    )

    return ProcessAndIndexResponse(
        **DocumentSummary.from_document(document).model_dump(),
        chunks_indexed=chunks_indexed,
        message="Document uploaded, processed and indexed successfully.",
    )


@router.post(
    "/ingest-url",
    response_model=ProcessAndIndexResponse,
    summary="Fetch, process and index a web page",
)
async def ingest_url(
    request: IngestUrlRequest,
    service: IngestionDep,
) -> ProcessAndIndexResponse:
    """Download a web page server-side and ingest it as an HTML document."""
    document, chunks_indexed = await service.process_and_index_url(request.url)

    return ProcessAndIndexResponse(
        **DocumentSummary.from_document(document).model_dump(),
        chunks_indexed=chunks_indexed,
        message="Web page fetched, processed and indexed successfully.",
    )


@router.post(
    "/delete",
    response_model=DeleteDocumentsResponse,
    summary="Delete one or more documents from the index",
)
async def delete_documents(
    request: DeleteDocumentsRequest,
    service: IngestionDep,
) -> DeleteDocumentsResponse:
    """Remove documents and every chunk belonging to them.

    Declared as POST rather than DELETE because it carries a body of ids;
    `DELETE /{document_id}` handles the single-document case.
    """
    return _deletion_response(service.delete_documents(request.document_ids))


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentsResponse,
    summary="Delete one document from the index",
)
async def delete_document(document_id: str, service: IngestionDep) -> DeleteDocumentsResponse:
    """Remove a single document and every chunk belonging to it."""
    return _deletion_response(service.delete_documents([document_id]))


def _deletion_response(deleted: dict[str, int]) -> DeleteDocumentsResponse:
    matched = sum(1 for count in deleted.values() if count)
    chunks = sum(deleted.values())

    return DeleteDocumentsResponse(
        deleted=deleted,
        documents_deleted=matched,
        chunks_deleted=chunks,
        message=(
            f"Deleted {matched} document(s) and {chunks} chunk(s)."
            if matched
            else "Nothing was deleted; no document matched."
        ),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get metadata for a processed document",
)
async def get_document(document_id: str, service: IngestionDep) -> DocumentDetailResponse:
    """Return metadata, chunk and entity counts for a processed document."""
    return DocumentDetailResponse.from_document(service.get_document(document_id))
