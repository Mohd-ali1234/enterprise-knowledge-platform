"""Domain exceptions.

Pipeline stages raise these instead of leaking library-specific errors, and the
API layer maps them to HTTP responses in a single place
(`app.api.error_handlers`). This keeps `except Exception -> 500` out of routers.
"""

from __future__ import annotations


class PlatformError(Exception):
    """Base class for every error raised by the platform."""

    status_code: int = 500
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.default_message)
        self.message = message or self.default_message


# ── Ingestion ────────────────────────────────────────────────────


class DocumentProcessingError(PlatformError):
    """Raised when a document cannot be parsed or processed."""

    status_code = 422
    default_message = "The document could not be processed."


class UnsupportedDocumentError(DocumentProcessingError):
    """Raised when a file type has no registered extractor."""

    status_code = 415
    default_message = "This file type is not supported."


class DocumentNotFoundError(PlatformError):
    """Raised when a document id is unknown to the repository."""

    status_code = 404
    default_message = "Document not found."


class IndexingError(PlatformError):
    """Raised when embedding or writing to the vector store fails."""

    status_code = 502
    default_message = "The document could not be indexed."


# ── Query ────────────────────────────────────────────────────────


class RetrievalError(PlatformError):
    """Raised when the retrieval stack fails to answer a search."""

    status_code = 502
    default_message = "Retrieval failed."


class QueryRewriteError(PlatformError):
    """Raised when a rewriter cannot produce usable query variants.

    This is normally recoverable: `ResilientQueryRewriter` catches it and falls
    back to the local heuristic rewriter.
    """

    status_code = 502
    default_message = "Query rewriting failed."


class AnswerGenerationError(PlatformError):
    """Raised when no generator could produce an answer."""

    status_code = 502
    default_message = "Answer generation failed."


# ── LLM providers ────────────────────────────────────────────────


class LLMError(PlatformError):
    """Raised when a hosted LLM is unreachable, unconfigured or unusable.

    Stages catch this and re-raise it as their own stage error, so a provider
    outage degrades to the offline path rather than failing the request.
    """

    status_code = 502
    default_message = "The language model is unavailable."
