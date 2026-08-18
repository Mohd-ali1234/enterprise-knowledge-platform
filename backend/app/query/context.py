"""Stage: Context Building.

Packs the highest-ranked chunks into a token-budgeted, de-duplicated,
citation-numbered context block for the answer generator.
"""

from __future__ import annotations

from typing import Sequence

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.query import AnswerContext, Citation
from app.domain.retrieval import ChunkMatch

logger = get_logger(__name__)


class ContextBuilder:
    """Selects chunks that fit the budget and numbers them for citation."""

    def __init__(self, token_budget: int | None = None, chars_per_token: int | None = None):
        self.token_budget = token_budget or settings.context_token_budget
        self.chars_per_token = chars_per_token or settings.context_chars_per_token

    @property
    def char_budget(self) -> int:
        return self.token_budget * self.chars_per_token

    def build(self, matches: Sequence[ChunkMatch]) -> AnswerContext:
        selected = self._select(matches)

        context = AnswerContext(
            text="\n\n".join(
                f"[{position}] {match.text}" for position, match in enumerate(selected, start=1)
            ),
            chunks=selected,
            citations=[
                Citation(
                    citation_id=position,
                    chunk_id=match.id,
                    document_id=match.document_id,
                    section_title=match.section_title,
                )
                for position, match in enumerate(selected, start=1)
            ],
        )

        logger.debug(
            "Built context from %d/%d chunks (%d char budget)",
            len(selected),
            len(matches),
            self.char_budget,
        )
        return context

    def _select(self, matches: Sequence[ChunkMatch]) -> list[ChunkMatch]:
        """Take chunks in rank order until the budget is spent, skipping repeats."""
        selected: list[ChunkMatch] = []
        seen: set[str] = set()
        used_chars = 0

        for match in matches:
            normalized = " ".join(match.text.split())
            if not normalized or normalized in seen:
                continue

            # Always admit the top chunk, even if it alone exceeds the budget:
            # an over-long answer beats an empty one.
            if selected and used_chars + len(normalized) > self.char_budget:
                break

            seen.add(normalized)
            used_chars += len(normalized)
            selected.append(match)

        return selected
