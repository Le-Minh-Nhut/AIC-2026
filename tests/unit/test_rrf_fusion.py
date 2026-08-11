from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import Candidate
from retrieval.fusion import FusionValidationError, WeightedRRFConfig, weighted_reciprocal_rank_fusion


def _candidate(uid: str, rank: int, score: float) -> Candidate:
    frame_id = {"A": 10, "B": 20, "C": 30}[uid]
    return Candidate(
        keyframe_uid=uid,
        video_id="L21_V001",
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        keyframe_path=f"raw/keyframes/L21_V001/{frame_id:06d}.jpg",
        score=score,
        rank=rank,
        source="fixture",
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_weighted_rrf_ranks_deterministically_and_retains_breakdown(self) -> None:
        fused = weighted_reciprocal_rank_fusion(
            {
                "fgclip2": (_candidate("A", 1, 0.95), _candidate("B", 2, 0.80)),
                "pecore": (_candidate("B", 1, 0.20), _candidate("C", 2, 0.10)),
            },
            WeightedRRFConfig(k=60, weights={"fgclip2": 1.0, "pecore": 2.0}),
        )

        self.assertEqual([candidate.keyframe_uid for candidate in fused], ["B", "C", "A"])
        best = fused[0]
        self.assertEqual(best.source, "rrf_fusion")
        self.assertEqual([(score.source, score.rank) for score in best.source_scores], [("fgclip2", 2), ("pecore", 1)])
        self.assertAlmostEqual(best.score, 1.0 / 62 + 2.0 / 61)
        self.assertAlmostEqual(sum(score.rrf_contribution or 0.0 for score in best.source_scores), best.score)

    def test_rrf_rejects_duplicate_source_rows_and_weight_mismatch(self) -> None:
        config = WeightedRRFConfig(k=60, weights={"fgclip2": 1.0})
        with self.assertRaises(FusionValidationError):
            weighted_reciprocal_rank_fusion({"fgclip2": (_candidate("A", 1, 1.0), _candidate("A", 2, 0.5))}, config)
        with self.assertRaises(FusionValidationError):
            weighted_reciprocal_rank_fusion(
                {"fgclip2": (_candidate("A", 1, 1.0),), "pecore": ()},
                config,
            )
