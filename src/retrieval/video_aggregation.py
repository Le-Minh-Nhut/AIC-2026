"""Baseline keyframe-to-video score aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Sequence

from domain.models import Candidate, VideoCandidate


AggregationMethod = Literal["max", "mean_top_m"]


def aggregate_video_candidates(
    candidates: Sequence[Candidate],
    method: AggregationMethod = "max",
    top_m: int = 3,
) -> tuple[VideoCandidate, ...]:
    if method not in {"max", "mean_top_m"}:
        raise ValueError(f"Unsupported video aggregation method: {method}")
    if top_m < 1:
        raise ValueError("top_m must be at least 1")
    by_video: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_video[candidate.video_id].append(candidate)
    aggregated: list[VideoCandidate] = []
    for video_id, values in by_video.items():
        ordered = sorted(values, key=lambda item: (-item.score, item.rank, item.keyframe_uid))
        score = ordered[0].score if method == "max" else sum(item.score for item in ordered[:top_m]) / min(top_m, len(ordered))
        aggregated.append(
            VideoCandidate(
                video_id=video_id,
                score=score,
                best_keyframe_uid=ordered[0].keyframe_uid,
                contributing_keyframes=len(ordered),
            )
        )
    return tuple(sorted(aggregated, key=lambda item: (-item.score, item.video_id)))
