"""Knowledge graph endpoints.

Every read is bounded. There is deliberately no "return the whole graph" route:
a corpus of any size would produce a payload the browser cannot lay out, so the
UI starts from a capped slice and expands on demand via `/neighbors`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_graph_service
from app.api.schemas import (
    EntityDetailResponse,
    GraphResponse,
    GraphSearchResponse,
    GraphStatsResponse,
)
from app.application.graph_service import GraphService

router = APIRouter(prefix="/knowledge-graph", tags=["Knowledge Graph"])

GraphDep = Annotated[GraphService, Depends(get_graph_service)]


@router.get("", response_model=GraphResponse, summary="Read a slice of the graph")
async def read_graph(
    service: GraphDep,
    document_id: str | None = Query(None, description="Restrict to one document"),
    entity_type: str | None = Query(None, description="e.g. PERSON, ORG, ROLE"),
    relationship_type: str | None = Query(None, description="e.g. REPORTS_TO"),
    search: str | None = Query(None, description="Substring match on entity names"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(150, ge=1, le=500),
) -> GraphResponse:
    """Return entities and relationships matching the filters."""
    view = service.graph(
        document_id=document_id,
        entity_type=entity_type,
        relationship_type=relationship_type,
        search=search,
        min_confidence=min_confidence,
        limit=limit,
    )
    return GraphResponse.from_view(view, service.degrees(view))


@router.get("/stats", response_model=GraphStatsResponse, summary="Graph statistics")
async def graph_stats(service: GraphDep) -> GraphStatsResponse:
    """Totals and per-type breakdowns for the whole graph."""
    return GraphStatsResponse(**service.stats().model_dump())


@router.get("/search", response_model=GraphSearchResponse, summary="Search entities")
async def search_entities(
    service: GraphDep,
    q: str = Query(..., min_length=1, description="Entity name to search for"),
    limit: int = Query(20, ge=1, le=100),
) -> GraphSearchResponse:
    """Find entities by name, best match first."""
    entities = service.search(q, limit=limit)
    return GraphSearchResponse(query=q, entities=entities, total=len(entities))


@router.get(
    "/entities/{entity_id}",
    response_model=EntityDetailResponse,
    summary="One entity with its relationships and sources",
)
async def read_entity(entity_id: str, service: GraphDep) -> EntityDetailResponse:
    """Return an entity, everything it connects to, and where it came from."""
    detail = service.entity(entity_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No entity '{entity_id}' in the graph.")
    return EntityDetailResponse.from_detail(detail)


@router.get(
    "/entities/{entity_id}/neighbors",
    response_model=GraphResponse,
    summary="Expand the neighbourhood around an entity",
)
async def read_neighbors(
    entity_id: str,
    service: GraphDep,
    depth: int = Query(1, ge=1, le=3),
    limit: int = Query(150, ge=1, le=500),
) -> GraphResponse:
    """Breadth-first expansion, for click-to-expand in the UI."""
    view = service.neighbors(entity_id, depth=depth, limit=limit)
    return GraphResponse.from_view(view, service.degrees(view))
