"""Stage: AI Routing.

Chooses which answering strategy should handle the request, based on how
confident retrieval was and how complex the query is. The decision is data, not
control flow - `AnswerGeneratorRegistry` turns it into a generator.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.query import AnswerContext, AnswerRoute, QueryIntent, QueryUnderstanding, RouteDecision

logger = get_logger(__name__)

_FACTUAL_INTENTS = frozenset({QueryIntent.LOOKUP, QueryIntent.QUESTION_ANSWERING})


class AnswerRouter:
    """Selects an `AnswerRoute` for a query and its retrieved context."""

    def __init__(
        self,
        confidence_threshold: float | None = None,
        complexity_threshold: float | None = None,
    ):
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.route_confidence_threshold
        )
        self.complexity_threshold = (
            complexity_threshold
            if complexity_threshold is not None
            else settings.route_complexity_threshold
        )

    def choose(self, understanding: QueryUnderstanding, context: AnswerContext) -> RouteDecision:
        confidence = self._confidence(context)

        if context.is_empty:
            decision = RouteDecision(
                route=AnswerRoute.LOCAL_LLM,
                reason="No matching chunks were retrieved.",
                confidence=confidence,
            )
        elif confidence >= self.confidence_threshold and understanding.intent in _FACTUAL_INTENTS:
            decision = RouteDecision(
                route=AnswerRoute.DIRECT,
                reason="High retrieval confidence with a direct factual query.",
                confidence=confidence,
            )
        elif understanding.complexity_score >= self.complexity_threshold:
            decision = RouteDecision(
                route=AnswerRoute.ONLINE_LLM,
                reason="Complex multi-step query may need stronger reasoning.",
                confidence=confidence,
            )
        else:
            decision = RouteDecision(
                route=AnswerRoute.LOCAL_LLM,
                reason="Retrieved context is relevant, but answer synthesis is useful.",
                confidence=confidence,
            )

        logger.info("Routed query to '%s': %s", decision.route.value, decision.reason)
        return decision

    @staticmethod
    def _confidence(context: AnswerContext) -> float:
        """Confidence is the relevance of the best chunk, clamped to 0-1."""
        if context.is_empty:
            return 0.0
        return round(min(max(context.chunks[0].score, 0.0), 1.0), 3)
