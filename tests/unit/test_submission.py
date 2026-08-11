from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.competition import FrameWindow, GroundTruthQuery, SubmissionCandidate, SubmissionQuery, TaskType
from submission.ranker import (
    FrameDiversityConfig,
    RankedFrame,
    RankedSequence,
    SequenceDiversityConfig,
    diversify_ranked_frames,
    diversify_ranked_sequences,
)
from submission.validation import SubmissionValidationError, SubmissionValidator
from submission.writer import load_submission, submission_from_debug, write_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = {"V1": 100, "V2": 100}

    def test_validator_accepts_all_task_shapes_and_rejects_failure_cases(self) -> None:
        ground_truth = GroundTruthQuery(
            "trake-1",
            TaskType.TRAKE,
            "V1",
            (FrameWindow(1, 2), FrameWindow(3, 4), FrameWindow(5, 6)),
        )
        kis_ground_truth = GroundTruthQuery("kis-1", TaskType.KIS, "V1", (FrameWindow(1, 2),))
        qna_ground_truth = GroundTruthQuery(
            "qna-1",
            TaskType.QNA,
            "V1",
            (FrameWindow(2, 3),),
            answer="yes",
        )
        queries = (
            SubmissionQuery("kis-1", TaskType.KIS, (SubmissionCandidate("V1", (1,)),)),
            SubmissionQuery("qna-1", TaskType.QNA, (SubmissionCandidate("V1", (2,), "yes"),)),
            SubmissionQuery("trake-1", TaskType.TRAKE, (SubmissionCandidate("V1", (1, 3, 5)),)),
        )
        summary = SubmissionValidator(
            self.inventory,
            {"kis-1": kis_ground_truth, "qna-1": qna_ground_truth, "trake-1": ground_truth},
        ).validate(queries)
        self.assertEqual((summary.query_count, summary.candidate_count), (3, 3))

        with self.assertRaises(SubmissionValidationError):
            SubmissionValidator(self.inventory).validate(
                (SubmissionQuery("bad", TaskType.KIS, (SubmissionCandidate("V1", (1,)), SubmissionCandidate("V1", (1,)))),)
            )
        with self.assertRaises(SubmissionValidationError):
            SubmissionValidator(self.inventory).validate(
                (SubmissionQuery("bad", TaskType.KIS, (SubmissionCandidate("unknown", (1,)),)),)
            )
        with self.assertRaises(SubmissionValidationError):
            SubmissionValidator(self.inventory).validate(
                (SubmissionQuery("bad", TaskType.KIS, (SubmissionCandidate("V1", (100,)),)),)
            )
        with self.assertRaises(SubmissionValidationError):
            SubmissionValidator(self.inventory).validate(
                (SubmissionQuery("bad", TaskType.TRAKE, (SubmissionCandidate("V1", (5, 5)),)),)
            )

    def test_frame_and_sequence_diversity_are_deterministic(self) -> None:
        frames = diversify_ranked_frames(
            (
                RankedFrame("V1", 10, 1.0, 0.99),
                RankedFrame("V1", 11, 1.2, 0.98),
                RankedFrame("V1", 30, 3.0, 0.97),
                RankedFrame("V2", 10, 1.0, 0.96),
            ),
            FrameDiversityConfig(max_results=3, max_per_video=1, temporal_window_sec=1.0),
        )
        sequences = diversify_ranked_sequences(
            (
                RankedSequence("V1", (10, 20, 30), 0.99),
                RankedSequence("V1", (11, 21, 31), 0.98),
                RankedSequence("V2", (10, 20, 30), 0.97),
            ),
            SequenceDiversityConfig(max_results=3, max_per_video=2, near_duplicate_frame_window=2),
        )

        self.assertEqual([(item.video_id, item.frame_id) for item in frames], [("V1", 10), ("V2", 10)])
        self.assertEqual([(item.video_id, item.frame_ids) for item in sequences], [("V1", (10, 20, 30)), ("V2", (10, 20, 30))])

    def test_writer_converts_debug_with_diversity_and_round_trips(self) -> None:
        debug = {
            "candidates": [
                {"video_id": "V1", "frame_id": 10, "timestamp_sec": 1.0, "score": 0.9},
                {"video_id": "V1", "frame_id": 11, "timestamp_sec": 1.1, "score": 0.8},
                {"video_id": "V2", "frame_id": 5, "timestamp_sec": 0.5, "score": 0.7},
            ]
        }
        query = submission_from_debug(
            TaskType.KIS,
            "kis-1",
            debug,
            frame_config=FrameDiversityConfig(max_results=100, max_per_video=2, temporal_window_sec=1.0),
        )
        self.assertEqual([(item.video_id, item.frame_ids) for item in query.candidates], [("V1", (10,)), ("V2", (5,))])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.json"
            write_submission(path, (query,))
            self.assertEqual(load_submission(path), (query,))
