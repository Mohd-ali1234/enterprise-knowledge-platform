"""Reciprocal Rank Fusion.

Merges ranked lists from independent retrieval strategies. RRF uses rank rather
than raw score, which is what makes it safe to combine a cosine similarity with
a BM25 score even though the two are on unrelated scales.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import settings
from app.domain.retrieval import ChunkMatch, MatchSource, ScoreKind


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ChunkMatch]],
    limit: int,
    k: int | None = None,
) -> list[ChunkMatch]:
    """Fuse ranked lists into one, best first.

    Args:
        ranked_lists: Each list ordered best-first by its own strategy.
        limit: Maximum number of fused results to return.
        k: Smoothing constant; larger values flatten the weight of top ranks.

    Returns:
        Fused matches carrying a `ScoreKind.FUSION` score. A chunk found by
        more than one strategy is marked `MatchSource.HYBRID`.
    """
    k = settings.rrf_k if k is None else k

    fused_scores: dict[str, float] = {}
    best_match: dict[str, ChunkMatch] = {}
    seen_sources: dict[str, set[MatchSource]] = {}

    for ranked in ranked_lists:
        for rank, match in enumerate(ranked):
            fused_scores[match.id] = fused_scores.get(match.id, 0.0) + 1.0 / (k + rank + 1)
            seen_sources.setdefault(match.id, set()).add(match.source)

            # Keep the first-seen payload but merge every strategy's scores so
            # no provenance is lost during fusion.
            if match.id in best_match:
                best_match[match.id].scores.update(match.scores)
            else:
                best_match[match.id] = match.model_copy(deep=True)

    ordered = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)

    results: list[ChunkMatch] = []
    for chunk_id in ordered[:limit]:
        match = best_match[chunk_id].with_score(ScoreKind.FUSION, fused_scores[chunk_id])
        if len(seen_sources[chunk_id]) > 1:
            match.source = MatchSource.HYBRID
        results.append(match)

    return results
