from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.competition import TaskType
from domain.models import Candidate
from refinement.dense_frame_refiner import (
    DenseRefinementRun,
    RefinedFrameCandidate,
    RefinementFailure,
)
from submission.ranker import FrameDiversityConfig
from submission.writer import submission_from_debug
from tasks.kis_service import KisCoarseResult, KisDenseRefinementService


def _coarse(rank: int, score: float, video_id: str | None = None) -> Candidate:
    video = video_id or f"V{rank:03d}"
    frame_id = rank * 10
    return Candidate(
        keyframe_uid=f"{video}:{frame_id:06d}",
        video_id=video,
        original_frame_id=frame_id,
        timestamp_sec=float(frame_id),
        keyframe_path=f"raw/keyframes/{video}/{frame_id:06d}.jpg",
        score=score,
        rank=rank,
        source="rrf_fusion",
    )


class _FakeCoarseService:
    def __init__(self, candidates: tuple[Candidate, ...]) -> None:
        self._candidates = candidates

    def search(self, query: str, top_k: int) -> KisCoarseResult:
        candidates = self._candidates[:top_k]
        return KisCoarseResult(
            query=query,
            candidates=candidates,
            video_candidates=(),
            initial_candidate_count=len(candidates),
            temporal_nms_enabled=False,
        )


class _FakeRefiner:
    def __init__(
        self,
        refine_count: int,
        local_scores: dict[str, float] | None = None,
        failed_uids: set[str] | None = None,
    ) -> None:
        self.refine_count = refine_count
        self.local_scores = local_scores or {}
        self.failed_uids = failed_uids or set()

    def refine(self, query: str, coarse_candidates: tuple[Candidate, ...]) -> DenseRefinementRun:
        del query
        successes: list[RefinedFrameCandidate] = []
        failures: list[RefinementFailure] = []
        for coarse in sorted(coarse_candidates, key=lambda candidate: candidate.rank)[: self.refine_count]:
            if coarse.keyframe_uid in self.failed_uids:
                failures.append(
                    RefinementFailure(
                        source_keyframe_uid=coarse.keyframe_uid,
                        video_id=coarse.video_id,
                        coarse_original_frame_id=coarse.original_frame_id,
                        coarse_score=coarse.score,
                        error="synthetic local decode failure",
                    )
                )
                continue
            refined = RefinedFrameCandidate.from_coarse(coarse)
            successes.append(
                replace(
                    refined,
                    original_frame_id=coarse.original_frame_id + 1,
                    timestamp_sec=coarse.timestamp_sec + 0.1,
                    score=self.local_scores.get(coarse.keyframe_uid, coarse.score),
                    rank=1,
                    refinement_status="refined",
                )
            )
        return DenseRefinementRun(tuple(successes), tuple(failures))


class RefinementRankingRegressionTests(unittest.TestCase):
    def test_short_coarse_list_does_not_crash_or_duplicate_candidates(self) -> None:
        coarse = tuple(_coarse(rank, score=0.5) for rank in range(1, 3))
        service = KisDenseRefinementService(
            _FakeCoarseService(coarse),
            _FakeRefiner(refine_count=10),
        )

        result = service.search("synthetic query", top_k=2)

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(
            len({candidate.source_keyframe_uid for candidate in result.candidates}),
            2,
        )

    def test_kis_top_100_retains_rank_15_outside_refinement_budget(self) -> None:
        coarse = tuple(_coarse(rank, score=1.0 - rank / 1000.0) for rank in range(1, 101))
        service = KisDenseRefinementService(
            _FakeCoarseService(coarse),
            _FakeRefiner(refine_count=10),
        )

        result = service.search("synthetic query", top_k=100)

        self.assertEqual(len(result.candidates), 100)
        self.assertEqual(
            result.candidates[14].source_keyframe_uid,
            coarse[14].keyframe_uid,
        )
        self.assertEqual(result.candidates[14].rank, 15)
        self.assertEqual(result.candidates[14].refinement_status, "coarse_fallback")
        self.assertEqual([candidate.rank for candidate in result.candidates], list(range(1, 101)))
        payload = result.as_dict()
        self.assertEqual(payload["refinement"]["refined_candidate_count"], 10)
        self.assertEqual(payload["refinement"]["fallback_candidate_count"], 90)

    def test_kis_local_scores_do_not_reorder_global_coarse_candidates_or_submission(self) -> None:
        candidate_a = _coarse(1, 0.99, "A")
        candidate_b = _coarse(2, 0.10, "B")
        service = KisDenseRefinementService(
            _FakeCoarseService((candidate_a, candidate_b)),
            _FakeRefiner(
                refine_count=2,
                local_scores={candidate_a.keyframe_uid: 0.01, candidate_b.keyframe_uid: 99.0},
            ),
        )

        result = service.search("synthetic query", top_k=2)

        self.assertEqual([candidate.video_id for candidate in result.candidates], ["A", "B"])
        self.assertEqual(
            [candidate.rank for candidate in result.candidates],
            [1, 2],
        )
        debug = result.as_dict()
        submission = submission_from_debug(
            TaskType.KIS,
            "kis-regression",
            debug,
            frame_config=FrameDiversityConfig(
                max_results=100,
                max_per_video=20,
                temporal_window_sec=0.0,
            ),
        )
        self.assertEqual([candidate.video_id for candidate in submission.candidates], ["A", "B"])

    def test_equal_local_scores_keep_deterministic_coarse_order(self) -> None:
        candidate_a = _coarse(1, 0.90, "A")
        candidate_b = _coarse(2, 0.80, "B")
        service = KisDenseRefinementService(
            _FakeCoarseService((candidate_a, candidate_b)),
            _FakeRefiner(
                refine_count=2,
                local_scores={candidate_a.keyframe_uid: 1.0, candidate_b.keyframe_uid: 1.0},
            ),
        )

        first = service.search("synthetic query", top_k=2)
        second = service.search("synthetic query", top_k=2)

        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual([candidate.video_id for candidate in first.candidates], ["A", "B"])

    def test_refinement_failure_keeps_coarse_fallback_and_is_deterministic(self) -> None:
        coarse = tuple(_coarse(rank, score=1.0 - rank / 10.0) for rank in range(1, 4))
        service = KisDenseRefinementService(
            _FakeCoarseService(coarse),
            _FakeRefiner(refine_count=2, failed_uids={coarse[0].keyframe_uid}),
        )

        first = service.search("synthetic query", top_k=3)
        second = service.search("synthetic query", top_k=3)

        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(
            [candidate.video_id for candidate in first.candidates],
            ["V001", "V002", "V003"],
        )
        self.assertEqual(first.candidates[0].refinement_status, "coarse_fallback")
        self.assertEqual(first.candidates[0].original_frame_id, coarse[0].original_frame_id)
        self.assertEqual(first.failures[0].source_keyframe_uid, coarse[0].keyframe_uid)
        self.assertEqual(len(first.candidates), 3)
