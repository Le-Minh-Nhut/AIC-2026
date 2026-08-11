"""Candidate-video selection from the union of per-event retrieval results."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from trake.event_candidates import EventCandidate


class CandidateVideoSelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CandidateVideoEvidence:
    video_id: str
    matched_event_count: int
    event_indices: tuple[int, ...]
    evidence_score: float
    candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CandidateVideoSelector:
    """Ranks the union of videos without replacing sequence-aware final scoring."""

    def select(
        self,
        event_candidates: Sequence[EventCandidate],
        candidate_videos: int,
    ) -> tuple[CandidateVideoEvidence, ...]:
        if candidate_videos < 1:
            raise CandidateVideoSelectionError("candidate_videos must be at least 1")
        best_by_video_event: dict[tuple[str, int], EventCandidate] = {}
        counts: dict[str, int] = {}
        for event_candidate in event_candidates:
            if not math.isfinite(event_candidate.score):
                raise CandidateVideoSelectionError("Event candidate scores must be finite")
            key = (event_candidate.video_id, event_candidate.event.index)
            previous = best_by_video_event.get(key)
            if previous is None or _better(event_candidate, previous):
                best_by_video_event[key] = event_candidate
            counts[event_candidate.video_id] = counts.get(event_candidate.video_id, 0) + 1
        evidence: list[CandidateVideoEvidence] = []
        for video_id in sorted(counts):
            matches = tuple(
                candidate
                for (candidate_video_id, _), candidate in best_by_video_event.items()
                if candidate_video_id == video_id
            )
            event_indices = tuple(sorted(candidate.event.index for candidate in matches))
            evidence.append(
                CandidateVideoEvidence(
                    video_id=video_id,
                    matched_event_count=len(matches),
                    event_indices=event_indices,
                    evidence_score=sum(candidate.score for candidate in matches),
                    candidate_count=counts[video_id],
                )
            )
        ordered = sorted(
            evidence,
            key=lambda item: (-item.matched_event_count, -item.evidence_score, item.video_id),
        )
        return tuple(ordered[:candidate_videos])


def _better(left: EventCandidate, right: EventCandidate) -> bool:
    return (-left.score, left.original_frame_id, left.stable_id) < (
        -right.score,
        right.original_frame_id,
        right.stable_id,
    )
