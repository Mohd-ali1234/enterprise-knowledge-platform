"""Port: turn grounded context into an answer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.query import AnswerContext, GeneratedAnswer


@runtime_checkable
class AnswerGenerator(Protocol):
    """Produces the final answer from retrieved context.

    Implementations are registered per `AnswerRoute` in
    `app.query.generation.AnswerGeneratorRegistry`, which is how a local or
    hosted LLM generator is added without changing the pipeline.
    """

    name: str

    async def generate(self, query: str, context: AnswerContext) -> GeneratedAnswer:
        ...
