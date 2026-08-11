"""Configurable temporal suppression for nearby keyframes from the same video."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from domain.models import Candidate


def temporal_nms(
    candidates: Sequence[Candidate],
    window_sec: float,
    max_candidates: int | None = None,
) -> tuple[Candidate, ...]:
    if window_sec < 0:
        raise ValueError("Temporal NMS window must be non-negative")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be at least 1 when provided")
    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.rank, item.keyframe_uid)):
        near_existing = any(
            candidate.video_id == existing.video_id
            and abs(candidate.timestamp_sec - existing.timestamp_sec) <= window_sec
            for existing in selected
        )
        if near_existing:
            continue
        selected.append(candidate)
        if max_candidates is not None and len(selected) >= max_candidates:
            break
    return tuple(replace(candidate, rank=rank) for rank, candidate in enumerate(selected, start=1))
