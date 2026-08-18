"""Service health endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.config import settings

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Liveness probe. Does not touch the model or the vector store."""
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
    )
