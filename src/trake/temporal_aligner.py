"""K-best dynamic-programming alignment for ordered TRAKE events."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from query.event_decomposer import EventQuery
from trake.event_candidates import TemporalEventCandidate


class TemporalAlignmentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TemporalAlignmentConfig:
    min_temporal_gap_sec: float = 0.0
    max_temporal_gap_sec: float | None = None
    gap_penalty: float = 0.0
    k_best_sequences: int = 10
    sequence_dedup_window_sec: float = 0.5

    def __post_init__(self) -> None:
        values = {
            "min_temporal_gap_sec": self.min_temporal_gap_sec,
            "gap_penalty": self.gap_penalty,
            "sequence_dedup_window_sec": self.sequence_dedup_window_sec,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise TemporalAlignmentError(f"{name} must be finite and non-negative")
        if self.max_temporal_gap_sec is not None:
            if not math.isfinite(self.max_temporal_gap_sec) or self.max_temporal_gap_sec < 0:
                raise TemporalAlignmentError("max_temporal_gap_sec must be finite and non-negative")
            if self.max_temporal_gap_sec < self.min_temporal_gap_sec:
                raise TemporalAlignmentError("max_temporal_gap_sec must be at least min_temporal_gap_sec")
        if self.k_best_sequences < 1:
            raise TemporalAlignmentError("k_best_sequences must be at least 1")


@dataclass(frozen=True, slots=True)
class TemporalCandidateMatrix:
    """One video's event × temporally ordered candidate matrix."""

    video_id: str
    events: tuple[EventQuery, ...]
    rows: tuple[tuple[TemporalEventCandidate, ...], ...]

    def __post_init__(self) -> None:
        if not self.video_id:
            raise TemporalAlignmentError("Candidate matrix video_id must be non-empty")
        if len(self.events) != len(self.rows):
            raise TemporalAlignmentError("Candidate matrix needs one row for every event")
        for event, row in zip(self.events, self.rows, strict=True):
            for candidate in row:
                if candidate.event.index != event.index:
                    raise TemporalAlignmentError("Candidate matrix row contains the wrong event")
                if candidate.video_id != self.video_id:
                    raise TemporalAlignmentError("Candidate matrix contains another video's frame")
                if candidate.original_frame_id < 0 or not math.isfinite(candidate.timestamp_sec):
                    raise TemporalAlignmentError("Temporal candidates need valid frame IDs and timestamps")
                if not math.isfinite(candidate.score):
                    raise TemporalAlignmentError("Temporal candidates need finite scores")


@dataclass(frozen=True, slots=True)
class TemporalAlignment:
    video_id: str
    matches: tuple[TemporalEventCandidate, ...]
    event_scores: tuple[float, ...]
    transition_penalty: float
    total_score: float
    rank: int = 0

    def __post_init__(self) -> None:
        if len(self.matches) != len(self.event_scores):
            raise TemporalAlignmentError("Alignment needs one score for every event match")
        if not self.matches:
            raise TemporalAlignmentError("Alignment cannot be empty")
        if any(match.video_id != self.video_id for match in self.matches):
            raise TemporalAlignmentError("Alignment matches must all share the video_id")

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(match.original_frame_id for match in self.matches)

    @property
    def timestamps_sec(self) -> tuple[float, ...]:
        return tuple(match.timestamp_sec for match in self.matches)

    def as_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "rank": self.rank,
            "ordered_frame_ids": list(self.frame_ids),
            "timestamps_sec": list(self.timestamps_sec),
            "event_scores": list(self.event_scores),
            "transition_penalty": self.transition_penalty,
            "total_alignment_score": self.total_score,
            "matches": [match.as_dict() for match in self.matches],
        }


@dataclass(frozen=True, slots=True)
class _PathState:
    matches: tuple[TemporalEventCandidate, ...]
    event_scores: tuple[float, ...]
    transition_penalty: float
    total_score: float


def build_candidate_matrix(
    video_id: str,
    events: Sequence[EventQuery],
    candidates: Sequence[TemporalEventCandidate],
) -> TemporalCandidateMatrix:
    """Build a stable event × candidate matrix for one candidate video."""

    ordered_events = tuple(events)
    event_indices = {event.index for event in ordered_events}
    if len(event_indices) != len(ordered_events):
        raise TemporalAlignmentError("Event indices must be unique")
    by_event: dict[int, list[TemporalEventCandidate]] = {event.index: [] for event in ordered_events}
    for candidate in candidates:
        if candidate.video_id != video_id:
            continue
        if candidate.event.index not in by_event:
            raise TemporalAlignmentError("Candidate belongs to an unknown TRAKE event")
        by_event[candidate.event.index].append(candidate)
    return TemporalCandidateMatrix(
        video_id=video_id,
        events=ordered_events,
        rows=tuple(_unique_ordered_row(by_event[event.index]) for event in ordered_events),
    )


class TemporalAligner:
    """Find k-best strictly monotonic event sequences with dynamic programming."""

    def __init__(self, config: TemporalAlignmentConfig) -> None:
        self._config = config

    @property
    def config(self) -> TemporalAlignmentConfig:
        return self._config

    def align(self, matrix: TemporalCandidateMatrix) -> tuple[TemporalAlignment, ...]:
        if not matrix.events or any(not row for row in matrix.rows):
            return ()
        paths_by_row: list[list[tuple[_PathState, ...]]] = []
        first_row = [
            (
                _PathState(
                    matches=(candidate,),
                    event_scores=(candidate.score,),
                    transition_penalty=0.0,
                    total_score=candidate.score,
                ),
            )
            for candidate in matrix.rows[0]
        ]
        paths_by_row.append(first_row)
        for row_index in range(1, len(matrix.rows)):
            previous_candidates = matrix.rows[row_index - 1]
            previous_paths = paths_by_row[-1]
            current_paths: list[tuple[_PathState, ...]] = []
            for current_candidate in matrix.rows[row_index]:
                states: list[_PathState] = []
                for previous_candidate, previous_states in zip(
                    previous_candidates,
                    previous_paths,
                    strict=True,
                ):
                    penalty = self._transition_penalty(previous_candidate, current_candidate)
                    if penalty is None:
                        continue
                    states.extend(
                        _PathState(
                            matches=state.matches + (current_candidate,),
                            event_scores=state.event_scores + (current_candidate.score,),
                            transition_penalty=state.transition_penalty + penalty,
                            total_score=state.total_score + current_candidate.score - penalty,
                        )
                        for state in previous_states
                    )
                current_paths.append(tuple(self._order_states(states)[: self._config.k_best_sequences]))
            paths_by_row.append(current_paths)
        complete_paths = [state for states in paths_by_row[-1] for state in states]
        ordered = self._order_states(complete_paths)
        alignments = tuple(
            TemporalAlignment(
                video_id=matrix.video_id,
                matches=state.matches,
                event_scores=state.event_scores,
                transition_penalty=state.transition_penalty,
                total_score=state.total_score,
            )
            for state in ordered
        )
        return self.deduplicate(alignments)[: self._config.k_best_sequences]

    def deduplicate(self, alignments: Sequence[TemporalAlignment]) -> tuple[TemporalAlignment, ...]:
        kept: list[TemporalAlignment] = []
        for alignment in sorted(alignments, key=_alignment_sort_key):
            if any(self.are_near_duplicates(alignment, existing) for existing in kept):
                continue
            kept.append(alignment)
        return tuple(replace(alignment, rank=rank) for rank, alignment in enumerate(kept, start=1))

    def are_near_duplicates(self, left: TemporalAlignment, right: TemporalAlignment) -> bool:
        return left.video_id == right.video_id and len(left.matches) == len(right.matches) and all(
            abs(left_timestamp - right_timestamp) <= self._config.sequence_dedup_window_sec
            for left_timestamp, right_timestamp in zip(left.timestamps_sec, right.timestamps_sec, strict=True)
        )

    def _transition_penalty(
        self,
        previous: TemporalEventCandidate,
        current: TemporalEventCandidate,
    ) -> float | None:
        if previous.original_frame_id >= current.original_frame_id:
            return None
        gap_sec = current.timestamp_sec - previous.timestamp_sec
        if gap_sec < self._config.min_temporal_gap_sec:
            return None
        if self._config.max_temporal_gap_sec is not None and gap_sec > self._config.max_temporal_gap_sec:
            return None
        return self._config.gap_penalty * max(0.0, gap_sec - self._config.min_temporal_gap_sec)

    @staticmethod
    def _order_states(states: Sequence[_PathState]) -> list[_PathState]:
        return sorted(
            states,
            key=lambda state: (
                -state.total_score,
                tuple(match.original_frame_id for match in state.matches),
                tuple(match.stable_id for match in state.matches),
            ),
        )


def _unique_ordered_row(
    candidates: Sequence[TemporalEventCandidate],
) -> tuple[TemporalEventCandidate, ...]:
    by_frame_id: dict[int, TemporalEventCandidate] = {}
    for candidate in candidates:
        previous = by_frame_id.get(candidate.original_frame_id)
        if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
            by_frame_id[candidate.original_frame_id] = candidate
    return tuple(sorted(by_frame_id.values(), key=lambda candidate: (candidate.original_frame_id, candidate.stable_id)))


def _candidate_sort_key(candidate: TemporalEventCandidate) -> tuple[float, str]:
    return (-candidate.score, candidate.stable_id)


def _alignment_sort_key(alignment: TemporalAlignment) -> tuple[float, str, tuple[int, ...], tuple[str, ...]]:
    return (
        -alignment.total_score,
        alignment.video_id,
        alignment.frame_ids,
        tuple(match.stable_id for match in alignment.matches),
    )
