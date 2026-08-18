"""Stage: Query Understanding.

Derives intent, entities and a complexity score from the raw query. These
signals drive routing later in the pipeline, so this stage stays cheap and
deterministic - no model call.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.domain.query import QueryIntent, QueryUnderstanding

logger = get_logger(__name__)

_WORD = re.compile(r"\w+")
_PROPER_NOUN_PHRASE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*\b")

_QUESTION_WORDS = frozenset({"what", "when", "where", "which", "who", "why", "how"})
_SUMMARY_TERMS = frozenset({"summarize", "summarise", "summary", "overview", "brief"})
_ANALYSIS_TERMS = frozenset(
    {"compare", "analyze", "analyse", "impact", "reason", "explain", "pros", "cons"}
)
_BREADTH_TERMS = frozenset({"all", "multiple", "across", "every"})

_MAX_ENTITIES = 10


class QueryUnderstandingService:
    """Classifies a query's intent and estimates how hard it is to answer."""

    def understand(self, query: str) -> QueryUnderstanding:
        tokens = _WORD.findall(query.lower())
        unique_tokens = set(tokens)
        entities = _PROPER_NOUN_PHRASE.findall(query)

        intent = self._classify_intent(unique_tokens)
        complexity = self._score_complexity(intent, tokens, unique_tokens, entities)

        understanding = QueryUnderstanding(
            intent=intent,
            entities=entities[:_MAX_ENTITIES],
            language="en",
            complexity_score=complexity,
            signals={
                "token_count": len(tokens),
                "entity_count": len(entities),
                "has_question_word": bool(unique_tokens & _QUESTION_WORDS),
            },
        )

        logger.debug(
            "Understood query as intent=%s complexity=%.2f",
            intent.value,
            complexity,
        )
        return understanding

    @staticmethod
    def _classify_intent(unique_tokens: set[str]) -> QueryIntent:
        if unique_tokens & _SUMMARY_TERMS:
            return QueryIntent.SUMMARIZATION
        if unique_tokens & _ANALYSIS_TERMS:
            return QueryIntent.ANALYSIS
        if unique_tokens & _QUESTION_WORDS:
            return QueryIntent.QUESTION_ANSWERING
        return QueryIntent.LOOKUP

    @staticmethod
    def _score_complexity(
        intent: QueryIntent,
        tokens: list[str],
        unique_tokens: set[str],
        entities: list[str],
    ) -> float:
        """Blend length, intent, entity count and breadth into a 0-1 score."""
        score = 0.15
        score += min(len(tokens) / 80, 0.35)
        if intent in {QueryIntent.ANALYSIS, QueryIntent.SUMMARIZATION}:
            score += 0.20
        if len(entities) > 2:
            score += 0.15
        if unique_tokens & _BREADTH_TERMS:
            score += 0.15

        return round(min(score, 1.0), 3)
