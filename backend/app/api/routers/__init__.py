"""Aggregates every router under a single API router."""

from fastapi import APIRouter

from app.api.routers.documents import router as documents_router
from app.api.routers.knowledge import router as knowledge_router
from app.api.routers.knowledge_graph import router as knowledge_graph_router
from app.api.routers.system import router as system_router

api_router = APIRouter()
api_router.include_router(documents_router)
api_router.include_router(knowledge_router)
api_router.include_router(knowledge_graph_router)
api_router.include_router(system_router)

__all__ = ["api_router"]
