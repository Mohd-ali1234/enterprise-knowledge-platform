"""Port: expand an ambiguous query into clearer variants."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.query import RewriteResult


@runtime_checkable
class QueryRewriter(Protocol):
    """Produces alternative phrasings of a query without answering it."""

    name: str

    async def rewrite(self, query: str) -> RewriteResult:
        """Return clearer variants, best first.

        Raises:
            QueryRewriteError: if no usable variant could be produced.
        """
        ...
