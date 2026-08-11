"""Task-level evaluation with aggregate BTC R@K and Final Score metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from domain.competition import GroundTruthQuery, SubmissionQuery, TaskType
from evaluation.final_score import FINAL_SCORE_CUTOFFS, RankMetrics, calculate_rank_metrics
from evaluation.rscore import AnswerMatcher, r_score


class EvaluationValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    task: TaskType
    candidate_scores: tuple[float, ...]
    metrics: RankMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "task": self.task.value,
            "candidate_scores": list(self.candidate_scores),
            **self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    task: TaskType
    queries: tuple[QueryEvaluation, ...]
    aggregate_metrics: RankMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task.value,
            "query_count": len(self.queries),
            "aggregate": self.aggregate_metrics.as_dict(),
            "queries": [query.as_dict() for query in self.queries],
        }


class CompetitionEvaluator:
    """Evaluates one task at a time, preserving scores for every submitted rank."""

    def __init__(self, answer_matcher: AnswerMatcher | None = None) -> None:
        self._answer_matcher = answer_matcher

    def evaluate(
        self,
        ground_truths: Sequence[GroundTruthQuery],
        submissions: Mapping[str, SubmissionQuery],
    ) -> TaskEvaluation:
        if not ground_truths:
            raise EvaluationValidationError("Evaluation needs at least one ground-truth query")
        by_id = {ground_truth.query_id: ground_truth for ground_truth in ground_truths}
        if len(by_id) != len(ground_truths):
            raise EvaluationValidationError("Ground-truth query_id values must be unique")
        task = ground_truths[0].task
        if any(ground_truth.task is not task for ground_truth in ground_truths):
            raise EvaluationValidationError("Evaluate one task per call")
        unknown = sorted(set(submissions) - set(by_id))
        if unknown:
            raise EvaluationValidationError("Submission contains unknown query IDs: " + ", ".join(unknown))
        results = tuple(
            self._evaluate_query(ground_truth, submissions.get(ground_truth.query_id))
            for ground_truth in sorted(ground_truths, key=lambda value: value.query_id)
        )
        aggregate_scores = {
            cutoff: sum(result.metrics.recall_at[cutoff] for result in results) / len(results)
            for cutoff in FINAL_SCORE_CUTOFFS
        }
        aggregate = RankMetrics(
            recall_at=aggregate_scores,
            final_score=sum(aggregate_scores.values()) / len(FINAL_SCORE_CUTOFFS),
        )
        return TaskEvaluation(task=task, queries=results, aggregate_metrics=aggregate)

    def _evaluate_query(
        self,
        ground_truth: GroundTruthQuery,
        submission: SubmissionQuery | None,
    ) -> QueryEvaluation:
        if submission is not None and submission.task is not ground_truth.task:
            raise EvaluationValidationError(
                f"Task mismatch for query {ground_truth.query_id}: "
                f"{submission.task.value} != {ground_truth.task.value}"
            )
        candidates = submission.candidates if submission is not None else ()
        scores = tuple(r_score(ground_truth, candidate, self._answer_matcher) for candidate in candidates)
        return QueryEvaluation(
            query_id=ground_truth.query_id,
            task=ground_truth.task,
            candidate_scores=scores,
            metrics=calculate_rank_metrics(scores),
        )
