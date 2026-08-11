"""Rank-cutoff metrics used by the AIC 2026 BTC rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


FINAL_SCORE_CUTOFFS = (1, 5, 20, 50, 100)


class FinalScoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RankMetrics:
    recall_at: dict[int, float]
    final_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            **{f"R@{cutoff}": self.recall_at[cutoff] for cutoff in FINAL_SCORE_CUTOFFS},
            "final_score": self.final_score,
        }


def calculate_rank_metrics(scores: Sequence[float]) -> RankMetrics:
    """Compute BTC R@k=max R-Score among the first k answers and their mean."""

    if len(scores) > max(FINAL_SCORE_CUTOFFS):
        raise FinalScoreError("BTC accepts at most 100 ranked answers per query")
    normalized = tuple(float(score) for score in scores)
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in normalized):
        raise FinalScoreError("R-Scores must be finite values in [0, 1]")
    recall_at = {
        cutoff: max(normalized[:cutoff], default=0.0)
        for cutoff in FINAL_SCORE_CUTOFFS
    }
    return RankMetrics(
        recall_at=recall_at,
        final_score=sum(recall_at.values()) / len(FINAL_SCORE_CUTOFFS),
    )
