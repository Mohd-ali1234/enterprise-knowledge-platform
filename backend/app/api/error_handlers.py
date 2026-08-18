"""Maps domain exceptions to HTTP responses.

Routers therefore never wrap calls in `try/except Exception`: they raise (or
let stages raise) domain errors, and the mapping lives here.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import PlatformError
from app.core.logging import get_logger

logger = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(PlatformError)
    async def handle_platform_error(_: Request, exc: PlatformError) -> JSONResponse:
        logger.warning("%s: %s", type(exc).__name__, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "error": type(exc).__name__},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Log the traceback but never leak internals to the client.
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal error occurred.",
                "error": "InternalServerError",
            },
        )
