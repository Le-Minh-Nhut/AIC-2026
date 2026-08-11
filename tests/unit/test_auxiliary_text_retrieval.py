from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.text_artifacts import (
    TextArtifactValidationError,
    load_asr_records_jsonl,
    load_metadata_records_jsonl,
    load_ocr_records_jsonl,
)
from domain.models import KeyframeRecord
from retrieval.text_retriever import (
    ASRTextRetriever,
    KeyframeCandidateMapper,
    MetadataTextRetriever,
    OCRTextRetriever,
)


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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AuxiliaryTextRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keyframes = (
            _keyframe("L21_V001:000000", "L21_V001", 0),
            _keyframe("L21_V001:000010", "L21_V001", 10),
            _keyframe("L21_V002:000000", "L21_V002", 0),
        )
        self.mapper = KeyframeCandidateMapper(self.keyframes)

    def test_ocr_bm25_returns_keyframe_candidate_and_text_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ocr.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "record_id": "ocr-1",
                        "keyframe_uid": "L21_V001:000010",
                        "text": "Emergency exit sign",
                        "bbox": [0, 0, 10, 10],
                        "confidence": 0.98,
                    },
                    {
                        "record_id": "ocr-2",
                        "keyframe_uid": "L21_V002:000000",
                        "text": "Welcome home",
                        "bbox": [0, 0, 10, 10],
                        "confidence": 0.9,
                    },
                ],
            )

            records = load_ocr_records_jsonl(path)
            candidates = OCRTextRetriever(records, self.mapper).retrieve("exit", top_k=2)

            self.assertEqual(records[0].bbox, (0.0, 0.0, 10.0, 10.0))
            self.assertEqual(candidates[0].keyframe_uid, "L21_V001:000010")
            self.assertEqual(candidates[0].source_scores[0].evidence_id, "ocr-1")
            self.assertEqual(candidates[0].source_scores[0].evidence_text, "Emergency exit sign")

    def test_asr_hit_maps_segment_midpoint_to_nearest_keyframe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asr.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "segment_id": "asr-1",
                        "video_id": "L21_V001",
                        "start_sec": 0.7,
                        "end_sec": 1.1,
                        "text": "the door is open",
                    }
                ],
            )

            candidates = ASRTextRetriever(load_asr_records_jsonl(path), self.mapper).retrieve("door", top_k=1)

            self.assertEqual(candidates[0].video_id, "L21_V001")
            self.assertEqual(candidates[0].original_frame_id, 10)
            self.assertEqual(candidates[0].timestamp_sec, 1.0)
            self.assertEqual(candidates[0].source_scores[0].evidence_id, "asr-1")

    def test_metadata_indexes_only_declared_present_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "video_id": "L21_V002",
                        "fields": {"title": "High jump final", "tags": ["athletics", "bar"]},
                    }
                ],
            )

            records = load_metadata_records_jsonl(path, ["title", "tags"])
            candidates = MetadataTextRetriever(records, self.mapper).retrieve("jump", top_k=1)

            self.assertEqual(candidates[0].video_id, "L21_V002")
            self.assertEqual(candidates[0].source_scores[0].evidence_text, "tags: athletics bar title: High jump final")
            with self.assertRaises(TextArtifactValidationError):
                load_metadata_records_jsonl(path, ["description"])
