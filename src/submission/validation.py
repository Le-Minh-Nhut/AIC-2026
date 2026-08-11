"""Strict validation before a versioned AIC submission is written or evaluated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from domain.competition import GroundTruthQuery, SubmissionCandidate, SubmissionQuery, TaskType


class SubmissionValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SubmissionValidationSummary:
    query_count: int
    candidate_count: int


class SubmissionValidator:
    """Rejects malformed, duplicate, unknown-video, and invalid-frame answers."""

    def __init__(
        self,
        video_frame_counts: Mapping[str, int],
        ground_truths: Mapping[str, GroundTruthQuery] | None = None,
        max_results_per_query: int = 100,
    ) -> None:
        if not 1 <= max_results_per_query <= 100:
            raise SubmissionValidationError("max_results_per_query must be in [1, 100]")
        invalid_inventory = [
            video_id
            for video_id, frame_count in video_frame_counts.items()
            if not video_id or not isinstance(frame_count, int) or frame_count < 1
        ]
        if invalid_inventory:
            raise SubmissionValidationError("Video frame inventory needs non-empty IDs and positive frame counts")
        self._video_frame_counts = dict(video_frame_counts)
        self._ground_truths = dict(ground_truths or {})
        self._max_results = max_results_per_query

    def validate(self, queries: Sequence[SubmissionQuery]) -> SubmissionValidationSummary:
        seen_query_ids: set[str] = set()
        candidate_count = 0
        for query in queries:
            if query.query_id in seen_query_ids:
                raise SubmissionValidationError(f"Duplicate submission query_id: {query.query_id}")
            seen_query_ids.add(query.query_id)
            self._validate_query(query)
            candidate_count += len(query.candidates)
        return SubmissionValidationSummary(query_count=len(queries), candidate_count=candidate_count)

    def _validate_query(self, query: SubmissionQuery) -> None:
        if not query.candidates:
            raise SubmissionValidationError(f"Submission query has no candidates: {query.query_id}")
        if len(query.candidates) > self._max_results:
            raise SubmissionValidationError(
                f"Submission query {query.query_id} has {len(query.candidates)} results; maximum is {self._max_results}"
            )
        ground_truth = self._ground_truths.get(query.query_id)
        if self._ground_truths and ground_truth is None:
            raise SubmissionValidationError(f"Submission query is not present in ground truth: {query.query_id}")
        if ground_truth is not None and ground_truth.task is not query.task:
            raise SubmissionValidationError(f"Submission task differs from expected task for {query.query_id}")
        seen_candidates: set[tuple[object, ...]] = set()
        for candidate in query.candidates:
            self._validate_candidate(query, candidate, ground_truth)
            key = self._duplicate_key(query.task, candidate)
            if key in seen_candidates:
                raise SubmissionValidationError(f"Duplicate {query.task.value} candidate for {query.query_id}: {key}")
            seen_candidates.add(key)

    def _validate_candidate(
        self,
        query: SubmissionQuery,
        candidate: SubmissionCandidate,
        ground_truth: GroundTruthQuery | None,
    ) -> None:
        frame_count = self._video_frame_counts.get(candidate.video_id)
        if frame_count is None:
            raise SubmissionValidationError(f"Unknown video_id in submission: {candidate.video_id}")
        if any(frame_id >= frame_count for frame_id in candidate.frame_ids):
            raise SubmissionValidationError(f"Frame ID is outside video bounds for {candidate.video_id}")
        if query.task is TaskType.KIS:
            if len(candidate.frame_ids) != 1 or candidate.answer is not None:
                raise SubmissionValidationError("KIS result requires exactly video_id and frame_id")
        elif query.task is TaskType.QNA:
            if len(candidate.frame_ids) != 1 or candidate.answer is None:
                raise SubmissionValidationError("Q&A result requires video_id, frame_id, and non-empty answer")
        else:
            if candidate.answer is not None:
                raise SubmissionValidationError("TRAKE result cannot contain an answer")
            if any(left >= right for left, right in zip(candidate.frame_ids, candidate.frame_ids[1:])):
                raise SubmissionValidationError("TRAKE frame_ids must be strictly increasing")
            if ground_truth is not None and len(candidate.frame_ids) != len(ground_truth.frame_windows):
                raise SubmissionValidationError(
                    f"TRAKE result needs {len(ground_truth.frame_windows)} frame IDs for {query.query_id}"
                )

    @staticmethod
    def _duplicate_key(task: TaskType, candidate: SubmissionCandidate) -> tuple[object, ...]:
        if task in {TaskType.KIS, TaskType.QNA}:
            return candidate.video_id, candidate.frame_ids[0]
        return candidate.video_id, candidate.frame_ids
