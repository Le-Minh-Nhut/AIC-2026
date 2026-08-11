"""Typed, task-neutral records for official BTC scoring and submissions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CompetitionContractError(ValueError):
    pass


class TaskType(StrEnum):
    KIS = "kis"
    QNA = "qna"
    TRAKE = "trake"


@dataclass(frozen=True, slots=True)
class FrameWindow:
    start_frame_id: int
    end_frame_id: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.start_frame_id, self.end_frame_id)):
            raise CompetitionContractError("Frame window bounds must be integers")
        if self.start_frame_id < 0 or self.end_frame_id < self.start_frame_id:
            raise CompetitionContractError("Frame window requires 0 <= start_frame_id <= end_frame_id")

    def contains(self, frame_id: int) -> bool:
        return self.start_frame_id <= frame_id <= self.end_frame_id


@dataclass(frozen=True, slots=True)
class GroundTruthQuery:
    query_id: str
    task: TaskType
    video_id: str
    frame_windows: tuple[FrameWindow, ...]
    answer: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.video_id.strip():
            raise CompetitionContractError("Ground truth query_id and video_id must be non-empty")
        if not self.frame_windows:
            raise CompetitionContractError("Ground truth needs at least one frame window")
        if self.task in {TaskType.KIS, TaskType.QNA} and len(self.frame_windows) != 1:
            raise CompetitionContractError(f"{self.task.value} ground truth needs exactly one frame window")
        if self.task is TaskType.QNA and (self.answer is None or not self.answer.strip()):
            raise CompetitionContractError("Q&A ground truth needs a non-empty answer")
        if self.task is not TaskType.QNA and self.answer is not None:
            raise CompetitionContractError(f"{self.task.value} ground truth cannot contain an answer")


@dataclass(frozen=True, slots=True)
class SubmissionCandidate:
    video_id: str
    frame_ids: tuple[int, ...]
    answer: str | None = None

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise CompetitionContractError("Submission candidate video_id must be non-empty")
        if not self.frame_ids or any(
            isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0
            for frame_id in self.frame_ids
        ):
            raise CompetitionContractError("Submission candidate needs non-negative frame IDs")
        if self.answer is not None and not self.answer.strip():
            raise CompetitionContractError("Submission answer cannot be blank")


@dataclass(frozen=True, slots=True)
class SubmissionQuery:
    query_id: str
    task: TaskType
    candidates: tuple[SubmissionCandidate, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise CompetitionContractError("Submission query_id must be non-empty")
