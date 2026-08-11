from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

run_kis = importlib.import_module("run_kis")
domain_models = importlib.import_module("domain.models")
Candidate = domain_models.Candidate
CandidateSourceScore = domain_models.CandidateSourceScore
KeyframeRecord = domain_models.KeyframeRecord
retrieval_fusion = importlib.import_module("retrieval.fusion")
WeightedRRFConfig = retrieval_fusion.WeightedRRFConfig
kis_service = importlib.import_module("tasks.kis_service")
KisCoarseRetrievalService = kis_service.KisCoarseRetrievalService
KisMultiSourceRetrievalService = kis_service.KisMultiSourceRetrievalService


def _keyframe(uid: str, video_id: str, frame_id: int) -> KeyframeRecord:
    return KeyframeRecord(
        keyframe_uid=uid,
        video_id=video_id,
        keyframe_index=frame_id,
        keyframe_path=f"raw/keyframes/{video_id}/{frame_id:06d}.jpg",
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        width=2,
        height=2,
        file_size_bytes=1,
        is_readable=True,
        has_mapping=True,
    )


def _candidate(uid: str, video_id: str, frame_id: int, source: str, rank: int, score: float, evidence: str | None = None) -> Candidate:
    return Candidate(
        keyframe_uid=uid,
        video_id=video_id,
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        keyframe_path=f"raw/keyframes/{video_id}/{frame_id:06d}.jpg",
        score=score,
        rank=rank,
        source=source,
        source_scores=(
            CandidateSourceScore(
                source=source,
                rank=rank,
                score=score,
                evidence_id=f"{source}-{uid}",
                evidence_text=evidence,
            ),
        ),
    )


class _Branch:
    def __init__(self, source: str, candidates: tuple[Candidate, ...]) -> None:
        self.source = source
        self._candidates = candidates

    def retrieve(self, query: str, top_k: int) -> tuple[Candidate, ...]:
        return self._candidates[:top_k]


class _TextEncoder:
    embedding_dimension = 2

    def encode_texts(self, texts):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class _Retriever:
    def __init__(self, candidate: Candidate) -> None:
        self.candidate = candidate

    def retrieve(self, query_vector, top_k: int):
        return (self.candidate,)


class MultimodalRetrievalIntegrationTests(unittest.TestCase):
    def test_rrf_fuses_visual_ocr_asr_metadata_and_retains_text_evidence(self) -> None:
        candidate_a = "L21_V001:000010"
        candidate_b = "L21_V002:000010"
        branches = {
            "fgclip2": _Branch("fgclip2", (_candidate(candidate_a, "L21_V001", 10, "fgclip2", 1, 0.9),)),
            "pecore": _Branch("pecore", (_candidate(candidate_b, "L21_V002", 10, "pecore", 1, 0.9),)),
            "ocr": _Branch("ocr", (_candidate(candidate_a, "L21_V001", 10, "ocr", 1, 2.0, "EXIT"),)),
            "asr": _Branch("asr", (_candidate(candidate_b, "L21_V002", 10, "asr", 1, 2.0, "door open"),)),
            "metadata": _Branch("metadata", (_candidate(candidate_a, "L21_V001", 10, "metadata", 1, 1.0, "title: exit"),)),
        }
        service = KisMultiSourceRetrievalService(
            branches=branches,
            fusion_config=WeightedRRFConfig(
                k=60,
                weights={"fgclip2": 1.0, "pecore": 1.0, "ocr": 0.7, "asr": 0.7, "metadata": 0.5},
            ),
            temporal_nms_enabled=False,
            temporal_nms_window_sec=0.0,
            candidate_pool_multiplier=1,
        )

        result = service.search("exit", top_k=2)

        self.assertEqual(result.candidates[0].keyframe_uid, candidate_a)
        self.assertEqual(set(result.source_rankings), {"fgclip2", "pecore", "ocr", "asr", "metadata"})
        ocr_evidence = next(score for score in result.candidates[0].source_scores if score.source == "ocr")
        self.assertEqual(ocr_evidence.evidence_text, "EXIT")
        self.assertIsNotNone(ocr_evidence.rrf_contribution)

    def test_common_runtime_enables_or_disables_ocr_from_config(self) -> None:
        keyframe = _keyframe("L21_V001:000010", "L21_V001", 10)
        candidate = _candidate(keyframe.keyframe_uid, keyframe.video_id, 10, "fgclip2", 1, 0.9)
        retrieval_config = {
            "search": {"candidate_pool_multiplier": 1},
            "temporal_nms": {"enabled": False, "window_sec": 0.0},
            "video_aggregation": {"method": "max", "top_m": 1},
            "fusion": {"method": "rrf", "rrf_k": 60, "weights": {"fgclip2": 1.0, "ocr": 0.7}},
            "auxiliary_retrieval": {
                "ocr": {"enabled": False, "records_path": "unused.jsonl"},
                "asr": {"enabled": False, "records_path": "unused.jsonl"},
                "metadata": {"enabled": False, "records_path": "unused.jsonl", "fields": []},
            },
        }
        branch = run_kis.PreparedRetrieverBranch(
            source="fgclip2",
            text_encoder=_TextEncoder(),
            retriever=_Retriever(candidate),
            runtime_metadata={"fixture": True},
        )
        args = SimpleNamespace(no_temporal_nms=False)
        with patch.object(run_kis, "_fg_branch", return_value=branch):
            disabled = run_kis.build_kis_coarse_runtime(
                args=args,
                root=ROOT,
                data_root=ROOT,
                retrieval_config=retrieval_config,
                models_config={},
                keyframes=(keyframe,),
                encoder_name="fgclip2_large",
            )
        self.assertIsInstance(disabled.service, KisCoarseRetrievalService)
        self.assertFalse(disabled.runtime_metadata["auxiliary_retrieval"]["ocr"]["enabled"])

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "ocr.jsonl"
            artifact.write_text(
                json.dumps(
                    {
                        "keyframe_uid": keyframe.keyframe_uid,
                        "text": "exit sign",
                        "bbox": [0, 0, 1, 1],
                        "confidence": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            retrieval_config["auxiliary_retrieval"]["ocr"] = {
                "enabled": True,
                "records_path": str(artifact),
            }
            with patch.object(run_kis, "_fg_branch", return_value=branch):
                enabled = run_kis.build_kis_coarse_runtime(
                    args=args,
                    root=ROOT,
                    data_root=ROOT,
                    retrieval_config=retrieval_config,
                    models_config={},
                    keyframes=(keyframe,),
                    encoder_name="fgclip2_large",
                )

        self.assertIsInstance(enabled.service, KisMultiSourceRetrievalService)
        self.assertTrue(enabled.runtime_metadata["auxiliary_retrieval"]["ocr"]["enabled"])
        self.assertEqual(set(enabled.service.search("exit", 1).source_rankings), {"fgclip2", "ocr"})
