from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import Candidate
from query.event_decomposer import EventQuery, RuleBasedEventDecomposer
from refinement.dense_frame_refiner import RefinedFrameCandidate
from trake.event_candidates import EventCandidate
from trake.event_refiner import TrakeDenseEventRefiner
from trake.temporal_aligner import TemporalAligner, TemporalAlignmentConfig, build_candidate_matrix


def _event(index: int, text: str | None = None) -> EventQuery:
    return EventQuery(index=index, text=text or f"event {index}", context="athlete completes a high jump")


def _candidate(event: EventQuery, video_id: str, frame_id: int, score: float) -> EventCandidate:
    return EventCandidate(
        event=event,
        candidate=Candidate(
            keyframe_uid=f"{video_id}:{event.index}:{frame_id}",
            video_id=video_id,
            original_frame_id=frame_id,
            timestamp_sec=frame_id / 10.0,
            keyframe_path=f"raw/keyframes/{video_id}/{frame_id:06d}.jpg",
            score=score,
            rank=1,
            source="fgclip2",
        ),
    )


class _FakeFrameRefiner:
    def refine_candidate_frames(self, query: str, coarse: Candidate, top_k: int):
        del query
        options = {
            10: ((30, 0.99), (10, 0.80)),
            20: ((20, 0.99), (40, 0.80)),
        }[coarse.original_frame_id]
        return tuple(
            RefinedFrameCandidate(
                source_keyframe_uid=coarse.keyframe_uid,
                video_id=coarse.video_id,
                coarse_original_frame_id=coarse.original_frame_id,
                coarse_timestamp_sec=coarse.timestamp_sec,
                coarse_score=coarse.score,
                sparse_original_frame_id=frame_id,
                sparse_timestamp_sec=frame_id / 10.0,
                sparse_score=score,
                original_frame_id=frame_id,
                timestamp_sec=frame_id / 10.0,
                score=score,
                rank=rank,
                source="fgclip2",
                source_scores=(),
            )
            for rank, (frame_id, score) in enumerate(options[:top_k], start=1)
        )


class TrakeTemporalAlignmentTests(unittest.TestCase):
    def test_rule_decomposer_preserves_full_context_for_short_list_events(self) -> None:
        query = "chạy đà → giậm nhảy → qua xà → tiếp đất"

        events = RuleBasedEventDecomposer().decompose(query)

        self.assertEqual([event.text for event in events], ["chạy đà", "giậm nhảy", "qua xà", "tiếp đất"])
        self.assertIn(query, events[3].retrieval_text)
        self.assertIn("tiếp đất", events[3].retrieval_text)

    def test_dp_rejects_independent_argmax_when_it_breaks_temporal_order(self) -> None:
        events = (_event(0), _event(1))
        candidates = (
            _candidate(events[0], "L21_V001", 30, 0.99),
            _candidate(events[0], "L21_V001", 10, 0.80),
            _candidate(events[1], "L21_V001", 20, 0.99),
            _candidate(events[1], "L21_V001", 40, 0.70),
        )
        aligner = TemporalAligner(TemporalAlignmentConfig(k_best_sequences=3, sequence_dedup_window_sec=0.0))

        alignments = aligner.align(build_candidate_matrix("L21_V001", events, candidates))

        self.assertEqual(alignments[0].frame_ids, (10, 20))
        self.assertGreater(alignments[0].total_score, 1.5)
        self.assertNotIn((30, 20), [alignment.frame_ids for alignment in alignments])

    def test_k_best_removes_nearly_identical_sequences_but_keeps_another_basin(self) -> None:
        events = (_event(0), _event(1))
        candidates = (
            _candidate(events[0], "L21_V001", 10, 0.95),
            _candidate(events[0], "L21_V001", 11, 0.94),
            _candidate(events[0], "L21_V001", 50, 0.80),
            _candidate(events[1], "L21_V001", 30, 0.90),
            _candidate(events[1], "L21_V001", 31, 0.89),
            _candidate(events[1], "L21_V001", 70, 0.75),
        )
        aligner = TemporalAligner(TemporalAlignmentConfig(k_best_sequences=8, sequence_dedup_window_sec=0.2))

        alignments = aligner.align(build_candidate_matrix("L21_V001", events, candidates))

        self.assertEqual(alignments[0].frame_ids, (10, 30))
        self.assertIn((50, 70), [alignment.frame_ids for alignment in alignments])
        self.assertEqual(len({alignment.frame_ids for alignment in alignments}), len(alignments))
        self.assertTrue(
            all(
                not aligner.are_near_duplicates(left, right)
                for index, left in enumerate(alignments)
                for right in alignments[index + 1 :]
            )
        )

    def test_joint_dense_refinement_realigns_events_in_monotonic_order(self) -> None:
        events = (_event(0, "approach"), _event(1, "takeoff"))
        coarse_candidates = (
            _candidate(events[0], "L21_V001", 10, 0.8),
            _candidate(events[1], "L21_V001", 20, 0.8),
        )
        aligner = TemporalAligner(TemporalAlignmentConfig(k_best_sequences=2, sequence_dedup_window_sec=0.0))
        coarse = aligner.align(build_candidate_matrix("L21_V001", events, coarse_candidates))[0]
        refiner = TrakeDenseEventRefiner(_FakeFrameRefiner(), aligner, local_frame_candidates=2)

        run = refiner.refine(coarse)

        self.assertEqual(run.failures, ())
        self.assertEqual(run.alignments[0].frame_ids, (10, 20))
        self.assertTrue(all(left < right for left, right in zip(run.alignments[0].frame_ids, run.alignments[0].frame_ids[1:])))
