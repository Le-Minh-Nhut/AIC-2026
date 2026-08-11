"""Per-candidate BTC R-Score implementations for KIS, Q&A, and TRAKE."""

from __future__ import annotations

from typing import Protocol

from domain.competition import GroundTruthQuery, SubmissionCandidate, TaskType
from qna.answer_normalizer import AnswerNormalizationError, AnswerNormalizer


class AnswerMatcher(Protocol):
    def matches(self, predicted: str, expected: str) -> bool: ...


class NormalizedAnswerMatcher:
    """Conservative answer matcher using the existing shared Q&A normalizer."""

    def __init__(self, normalizer: AnswerNormalizer | None = None) -> None:
        self._normalizer = normalizer or AnswerNormalizer()

    def matches(self, predicted: str, expected: str) -> bool:
        try:
            return self._normalizer.normalize(predicted) == self._normalizer.normalize(expected)
        except AnswerNormalizationError:
            return False


def r_score(
    ground_truth: GroundTruthQuery,
    candidate: SubmissionCandidate,
    answer_matcher: AnswerMatcher | None = None,
) -> float:
    """Return the official relevance score for one already-validated answer."""

    if candidate.video_id != ground_truth.video_id:
        return 0.0
    if ground_truth.task is TaskType.KIS:
        return float(
            len(candidate.frame_ids) == 1 and ground_truth.frame_windows[0].contains(candidate.frame_ids[0])
        )
    if ground_truth.task is TaskType.QNA:
        if len(candidate.frame_ids) != 1 or candidate.answer is None:
            return 0.0
        frame_matches = ground_truth.frame_windows[0].contains(candidate.frame_ids[0])
        matcher = answer_matcher or NormalizedAnswerMatcher()
        return float(frame_matches and matcher.matches(candidate.answer, ground_truth.answer or ""))
    if len(candidate.frame_ids) != len(ground_truth.frame_windows):
        return 0.0
    matched_events = sum(
        window.contains(frame_id)
        for window, frame_id in zip(ground_truth.frame_windows, candidate.frame_ids, strict=True)
    )
    return matched_events / len(ground_truth.frame_windows)
