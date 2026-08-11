"""Deterministic Top-100 diversity policies for frames and TRAKE sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class RankingDiversityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FrameDiversityConfig:
    max_results: int = 100
    max_per_video: int = 20
    temporal_window_sec: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_results <= 100:
            raise RankingDiversityError("max_results must be in [1, 100]")
        if self.max_per_video < 1:
            raise RankingDiversityError("max_per_video must be at least 1")
        if not math.isfinite(self.temporal_window_sec) or self.temporal_window_sec < 0:
            raise RankingDiversityError("temporal_window_sec must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SequenceDiversityConfig:
    max_results: int = 100
    max_per_video: int = 20
    near_duplicate_frame_window: int = 15

    def __post_init__(self) -> None:
        if not 1 <= self.max_results <= 100:
            raise RankingDiversityError("max_results must be in [1, 100]")
        if self.max_per_video < 1:
            raise RankingDiversityError("max_per_video must be at least 1")
        if self.near_duplicate_frame_window < 0:
            raise RankingDiversityError("near_duplicate_frame_window must be non-negative")


@dataclass(frozen=True, slots=True)
class RankedFrame:
    video_id: str
    frame_id: int
    timestamp_sec: float
    score: float
    answer: str | None = None

    def __post_init__(self) -> None:
        if not self.video_id or self.frame_id < 0 or not math.isfinite(self.timestamp_sec):
            raise RankingDiversityError("Ranked frame needs a video ID, non-negative frame ID, and finite timestamp")
        if not math.isfinite(self.score):
            raise RankingDiversityError("Ranked frame score must be finite")


@dataclass(frozen=True, slots=True)
class RankedSequence:
    video_id: str
    frame_ids: tuple[int, ...]
    score: float

    def __post_init__(self) -> None:
        if not self.video_id or not self.frame_ids or any(frame_id < 0 for frame_id in self.frame_ids):
            raise RankingDiversityError("Ranked sequence needs a video ID and non-negative frame IDs")
        if any(left >= right for left, right in zip(self.frame_ids, self.frame_ids[1:])):
            raise RankingDiversityError("Ranked sequence frame IDs must be strictly increasing")
        if not math.isfinite(self.score):
            raise RankingDiversityError("Ranked sequence score must be finite")


def diversify_ranked_frames(
    candidates: Sequence[RankedFrame],
    config: FrameDiversityConfig,
) -> tuple[RankedFrame, ...]:
    """Keep the highest scores while capping videos and near-identical frame times."""

    kept: list[RankedFrame] = []
    per_video: dict[str, int] = {}
    seen_frames: set[tuple[str, int]] = set()
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.video_id, item.frame_id)):
        key = (candidate.video_id, candidate.frame_id)
        if key in seen_frames or per_video.get(candidate.video_id, 0) >= config.max_per_video:
            continue
        same_video = [item for item in kept if item.video_id == candidate.video_id]
        if any(abs(item.timestamp_sec - candidate.timestamp_sec) <= config.temporal_window_sec for item in same_video):
            continue
        kept.append(candidate)
        seen_frames.add(key)
        per_video[candidate.video_id] = per_video.get(candidate.video_id, 0) + 1
        if len(kept) == config.max_results:
            break
    return tuple(kept)


def diversify_ranked_sequences(
    candidates: Sequence[RankedSequence],
    config: SequenceDiversityConfig,
) -> tuple[RankedSequence, ...]:
    """Keep distinct TRAKE temporal basins with a deterministic per-video cap."""

    kept: list[RankedSequence] = []
    per_video: dict[str, int] = {}
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.video_id, item.frame_ids)):
        key = (candidate.video_id, candidate.frame_ids)
        if key in seen or per_video.get(candidate.video_id, 0) >= config.max_per_video:
            continue
        if any(_near_duplicate_sequence(candidate, item, config.near_duplicate_frame_window) for item in kept):
            continue
        kept.append(candidate)
        seen.add(key)
        per_video[candidate.video_id] = per_video.get(candidate.video_id, 0) + 1
        if len(kept) == config.max_results:
            break
    return tuple(kept)


def _near_duplicate_sequence(
    left: RankedSequence,
    right: RankedSequence,
    window: int,
) -> bool:
    return (
        left.video_id == right.video_id
        and len(left.frame_ids) == len(right.frame_ids)
        and all(abs(left_frame - right_frame) <= window for left_frame, right_frame in zip(left.frame_ids, right.frame_ids, strict=True))
    )
