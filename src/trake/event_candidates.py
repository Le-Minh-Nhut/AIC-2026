"""Typed event-to-frame evidence used by TRAKE temporal alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypeAlias

from domain.models import Candidate
from query.event_decomposer import EventQuery
from refinement.dense_frame_refiner import RefinedFrameCandidate


@dataclass(frozen=True, slots=True)
class EventCandidate:
    """A coarse keyframe retrieval result assigned to one ordered event."""

    event: EventQuery
    candidate: Candidate

    @property
    def video_id(self) -> str:
        return self.candidate.video_id

    @property
    def original_frame_id(self) -> int:
        return self.candidate.original_frame_id

    @property
    def timestamp_sec(self) -> float:
        return self.candidate.timestamp_sec

    @property
    def score(self) -> float:
        return self.candidate.score

    @property
    def stable_id(self) -> str:
        return self.candidate.keyframe_uid

    def as_dict(self) -> dict[str, object]:
        value = asdict(self.candidate)
        value["frame_id"] = self.original_frame_id
        value["event"] = self.event.as_dict()
        value["stage"] = "coarse"
        return value


@dataclass(frozen=True, slots=True)
class RefinedEventCandidate:
    """One dense original-video frame alternative for a TRAKE event."""

    event: EventQuery
    coarse_candidate: Candidate
    refined_candidate: RefinedFrameCandidate

    @property
    def video_id(self) -> str:
        return self.refined_candidate.video_id

    @property
    def original_frame_id(self) -> int:
        return self.refined_candidate.original_frame_id

    @property
    def timestamp_sec(self) -> float:
        return self.refined_candidate.timestamp_sec

    @property
    def score(self) -> float:
        return self.refined_candidate.score

    @property
    def stable_id(self) -> str:
        return f"{self.refined_candidate.source_keyframe_uid}:{self.original_frame_id}"

    def as_dict(self) -> dict[str, object]:
        value = asdict(self.refined_candidate)
        value["frame_id"] = self.original_frame_id
        value["coarse_frame_id"] = self.refined_candidate.coarse_original_frame_id
        value["sparse_frame_id"] = self.refined_candidate.sparse_original_frame_id
        value["event"] = self.event.as_dict()
        value["coarse_keyframe"] = asdict(self.coarse_candidate)
        value["stage"] = "refined"
        return value


TemporalEventCandidate: TypeAlias = EventCandidate | RefinedEventCandidate
