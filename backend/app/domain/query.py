"""Query-side domain models.

These describe the state that flows through the query pipeline:

    User Query -> QueryUnderstanding -> RewrittenQuery[] -> ChunkMatch[]
               -> AnswerContext -> RouteDecision -> GeneratedAnswer
               -> QueryResult
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.retrieval import ChunkMatch


class QueryIntent(str, Enum):
    """What the user is trying to do."""

    LOOKUP = "lookup"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"


class RewriteStrategy(str, Enum):
    """How a rewritten variant differs from the original query."""

    EXPAND = "EXPAND"
    DISAMBIGUATE = "DISAMBIGUATE"
    SPECIFY = "SPECIFY"


class AnswerRoute(str, Enum):
    """Which answering strategy the router selected."""

    DIRECT = "direct"
    LOCAL_LLM = "local_llm"
    ONLINE_LLM = "online_llm"


class QueryUnderstanding(BaseModel):
    """Structured reading of the raw user query."""

    intent: QueryIntent
    entities: list[str] = Field(default_factory=list)
    language: str = "en"
    complexity_score: float = 0.0
    signals: dict[str, Any] = Field(default_factory=dict)


class RewrittenQuery(BaseModel):
    """A single clearer variant of the original query."""

    query: str
    strategy: RewriteStrategy
    confidence: float = Field(ge=0.0, le=1.0)


class RewriteResult(BaseModel):
    """The output of the rewriting stage.

    Carries which rewriter produced the variants and, when the primary
    rewriter failed and a fallback was used, why.
    """

    queries: list[RewrittenQuery] = Field(default_factory=list)
    rewriter: str
    warning: str | None = None

    @property
    def best(self) -> RewrittenQuery | None:
        """The highest-priority variant, or `None` if there are no variants."""
        return self.queries[0] if self.queries else None


class Citation(BaseModel):
    """A numbered pointer from the answer back to its source chunk."""

    citation_id: int
    chunk_id: str | None = None
    document_id: str | None = None
    section_title: str | None = None


class AnswerContext(BaseModel):
    """The grounded context handed to an answer generator."""

    text: str = ""
    chunks: list[ChunkMatch] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class RouteDecision(BaseModel):
    """The router's choice of answering strategy, with its rationale."""

    route: AnswerRoute
    reason: str
    confidence: float = 0.0


class GeneratedAnswer(BaseModel):
    """An answer plus the name of the generator that produced it.

    `key_points` and `related_questions` are extras an LLM generator can derive
    from the same grounded context in the same call. Generators that cannot
    produce them - the extractive one - leave them empty rather than inventing
    them, so an empty list means "not available", never "none found".
    """

    text: str
    generator: str
    key_points: list[str] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """The complete outcome of one trip through the query pipeline."""

    query: str
    retrieval_query: str
    understanding: QueryUnderstanding
    rewritten_queries: list[RewrittenQuery] = Field(default_factory=list)
    rewrite_error: str | None = None
    matches: list[ChunkMatch] = Field(default_factory=list)
    context: AnswerContext = Field(default_factory=AnswerContext)
    route: RouteDecision
    answer: GeneratedAnswer
