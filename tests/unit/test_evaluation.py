from __future__ import annotations

import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.competition import FrameWindow, GroundTruthQuery, SubmissionCandidate, SubmissionQuery, TaskType
from evaluation.evaluator import CompetitionEvaluator
from evaluation.final_score import FinalScoreError, calculate_rank_metrics


class EvaluationTests(unittest.TestCase):
    def test_kis_rank_cutoffs_and_final_score_follow_btc_formula(self) -> None:
        ground_truth = GroundTruthQuery("kis-1", TaskType.KIS, "V1", (FrameWindow(10, 12),))
        submission = SubmissionQuery(
            "kis-1",
            TaskType.KIS,
            (
                SubmissionCandidate("V2", (10,)),
                SubmissionCandidate("V1", (20,)),
                SubmissionCandidate("V1", (11,)),
            ),
        )

        result = CompetitionEvaluator().evaluate((ground_truth,), {submission.query_id: submission})

        self.assertEqual(result.queries[0].candidate_scores, (0.0, 0.0, 1.0))
        self.assertEqual(result.aggregate_metrics.recall_at, {1: 0.0, 5: 1.0, 20: 1.0, 50: 1.0, 100: 1.0})
        self.assertAlmostEqual(result.aggregate_metrics.final_score, 0.8)

    def test_qa_uses_injected_normalized_semantic_answer_matching(self) -> None:
        ground_truth = GroundTruthQuery("qna-1", TaskType.QNA, "V1", (FrameWindow(5, 5),), answer="Năm")
        submission = SubmissionQuery("qna-1", TaskType.QNA, (SubmissionCandidate("V1", (5,), "5 people"),))

        result = CompetitionEvaluator().evaluate((ground_truth,), {submission.query_id: submission})

        self.assertEqual(result.queries[0].candidate_scores, (1.0,))

    def test_trake_partial_evidence_and_wrong_video_follow_official_r_score(self) -> None:
        ground_truth = GroundTruthQuery(
            "trake-1",
            TaskType.TRAKE,
            "V1",
            (FrameWindow(10, 12), FrameWindow(20, 22), FrameWindow(30, 32), FrameWindow(40, 42)),
        )
        partial = SubmissionQuery("trake-1", TaskType.TRAKE, (SubmissionCandidate("V1", (11, 25, 31, 41)),))
        wrong_video = SubmissionQuery("trake-1", TaskType.TRAKE, (SubmissionCandidate("V2", (11, 21, 31, 41)),))

        partial_result = CompetitionEvaluator().evaluate((ground_truth,), {partial.query_id: partial})
        wrong_result = CompetitionEvaluator().evaluate((ground_truth,), {wrong_video.query_id: wrong_video})

        self.assertEqual(partial_result.queries[0].candidate_scores, (0.75,))
        self.assertEqual(wrong_result.queries[0].candidate_scores, (0.0,))

    def test_final_score_matches_official_rank_three_example_and_rejects_over_100(self) -> None:
        metrics = calculate_rank_metrics((0.5, 0.1, 0.8) + (0.6,) * 12)

        self.assertEqual(metrics.recall_at, {1: 0.5, 5: 0.8, 20: 0.8, 50: 0.8, 100: 0.8})
        self.assertAlmostEqual(metrics.final_score, 0.74)
        with self.assertRaises(FinalScoreError):
            calculate_rank_metrics((0.0,) * 101)
