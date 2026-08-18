"""Retrieval and question-answering endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_knowledge_service
from app.api.schemas import (
    AskRequest,
    AskResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
)
from app.application.knowledge_service import KnowledgeService

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])

KnowledgeDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve relevant chunks for a query",
)
async def retrieve(body: RetrieveRequest, service: KnowledgeDep) -> RetrieveResponse:
    """Hybrid search over the indexed knowledge base, without answer generation."""
    matches = await service.retrieve(
        query=body.query,
        top_k=body.top_k,
        mode=body.mode,
        filters=body.filters,
    )

    return RetrieveResponse(
        query=body.query,
        results=[
            RetrieveResult.from_match(rank, match)
            for rank, match in enumerate(matches, start=1)
        ],
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Run the full query pipeline and return a grounded answer",
)
async def ask(body: AskRequest, service: KnowledgeDep) -> AskResponse:
    """Understand -> rewrite -> retrieve -> rerank -> build context -> route -> answer."""
    result = await service.ask(
        query=body.query,
        top_k=body.top_k,
        rerank_top_k=body.rerank_top_k,
        mode=body.mode,
        filters=body.filters,
        rewrite=body.rewrite,
    )
    return AskResponse.from_result(result)
