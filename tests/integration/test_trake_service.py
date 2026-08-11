from __future__ import annotations

import json
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

run_trake = importlib.import_module("run_trake")
Candidate = importlib.import_module("domain.models").Candidate
RuleBasedEventDecomposer = importlib.import_module("query.event_decomposer").RuleBasedEventDecomposer
KisCoarseResult = importlib.import_module("tasks.kis_service").KisCoarseResult
trake_service = importlib.import_module("tasks.trake_service")
TrakeService = trake_service.TrakeService
TrakeServiceConfig = trake_service.TrakeServiceConfig
temporal_aligner = importlib.import_module("trake.temporal_aligner")
TemporalAligner = temporal_aligner.TemporalAligner
TemporalAlignmentConfig = temporal_aligner.TemporalAlignmentConfig
CandidateVideoSelector = importlib.import_module("trake.video_selector").CandidateVideoSelector


def _candidate(video_id: str, event_name: str, frame_id: int, score: float) -> Candidate:
    return Candidate(
        keyframe_uid=f"{video_id}:{event_name}:{frame_id}",
        video_id=video_id,
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        keyframe_path=f"raw/keyframes/{video_id}/{frame_id:06d}.jpg",
        score=score,
        rank=1,
        source="fgclip2",
    )


class _FakeCoarseSearcher:
    def __init__(self) -> None:
        self._rankings = {
            "approach": (
                _candidate("L21_VBAD", "approach", 30, 0.99),
                _candidate("L21_VGOOD", "approach", 10, 0.80),
                _candidate("L21_VPARTIAL", "approach", 10, 0.98),
            ),
            "takeoff": (
                _candidate("L21_VBAD", "takeoff", 20, 0.99),
                _candidate("L21_VGOOD", "takeoff", 20, 0.80),
                _candidate("L21_VPARTIAL", "takeoff", 20, 0.97),
            ),
            "landing": (
                _candidate("L21_VBAD", "landing", 40, 0.99),
                _candidate("L21_VGOOD", "landing", 30, 0.80),
            ),
        }

    def search(self, query: str, top_k: int) -> KisCoarseResult:
        event_name = query.rsplit(":", maxsplit=1)[-1].strip()
        candidates = self._rankings[event_name][:top_k]
        return KisCoarseResult(
            query=query,
            candidates=candidates,
            video_candidates=(),
            initial_candidate_count=len(candidates),
            temporal_nms_enabled=True,
        )


class TrakeServiceIntegrationTests(unittest.TestCase):
    def test_end_to_end_service_keeps_only_video_with_complete_ordered_evidence(self) -> None:
        service = TrakeService(
            event_decomposer=RuleBasedEventDecomposer(),
            coarse_searcher=_FakeCoarseSearcher(),
            video_selector=CandidateVideoSelector(),
            temporal_aligner=TemporalAligner(
                TemporalAlignmentConfig(k_best_sequences=5, sequence_dedup_window_sec=0.0)
            ),
            config=TrakeServiceConfig(
                event_top_k=5,
                candidate_videos=3,
                k_best_sequences=5,
                sequences_to_refine=1,
            ),
        )

        result = service.search("approach -> takeoff -> landing")

        self.assertEqual(result.candidates[0].video_id, "L21_VGOOD")
        self.assertEqual(result.candidates[0].final_alignment.frame_ids, (10, 20, 30))
        self.assertEqual([video.video_id for video in result.candidate_videos], [
            "L21_VBAD",
            "L21_VGOOD",
            "L21_VPARTIAL",
        ])
        self.assertNotIn("L21_VBAD", [candidate.video_id for candidate in result.candidates])
        self.assertNotIn("L21_VPARTIAL", [candidate.video_id for candidate in result.candidates])
        payload = result.as_dict()
        self.assertEqual(payload["candidates"][0]["ordered_frame_ids"], [10, 20, 30])


class TrakeCliIntegrationTests(unittest.TestCase):
    def test_cli_uses_reusable_coarse_runtime_and_writes_debug_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "trake_debug.json"
            arguments = [
                "run_trake.py",
                "--coarse-only",
                "--query",
                "approach -> takeoff -> landing",
                "--debug-output",
                str(output_path),
            ]
            runtime = SimpleNamespace(
                service=_FakeCoarseSearcher(),
                runtime_metadata={"fixture": True},
                refinement_branches={},
                frame_fusion_config=None,
            )
            with patch.object(run_trake.run_kis, "load_keyframe_records_from_parquet", return_value=()), patch.object(
                run_trake.run_kis, "build_kis_coarse_runtime", return_value=runtime
            ), patch.object(sys, "argv", arguments):
                exit_code = run_trake.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["selected_encoder"], "fg_pe_fusion")
            self.assertFalse(payload["metadata"]["trake"]["refinement_enabled"])
            self.assertEqual(payload["candidates"][0]["ordered_frame_ids"], [10, 20, 30])
