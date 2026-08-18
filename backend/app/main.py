"""Application entry point.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import (
    get_gemini_client,
    get_ingestion_service,
    get_query_rewriter,
)
from app.api.error_handlers import register_error_handlers
from app.api.routers import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start-up and shut-down hooks.

    Heavy resources (the embedding model, the Chroma collection) load lazily on
    first use rather than here, so the service becomes healthy immediately.
    """
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    yield
    await get_query_rewriter().aclose()
    # The generator shares this client with the rewriter; closing twice is safe.
    await get_gemini_client().aclose()
    await get_ingestion_service().aclose()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description="Upload, process, index, and retrieve knowledge from your documents.",
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    return app


app = create_app()
