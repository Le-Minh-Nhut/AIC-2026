from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.keyframe_mapping import validate_mapping
from domain.models import KeyframeRecord, MappingRecord, VideoRecord


def keyframe(index: int) -> KeyframeRecord:
    return KeyframeRecord(
        keyframe_uid=f"L21_V001:{index:06d}",
        video_id="L21_V001",
        keyframe_index=index,
        keyframe_path=f"raw/keyframes/L21_V001/{index:06d}.jpg",
        original_frame_id=None,
        timestamp_sec=None,
        width=10,
        height=10,
        file_size_bytes=10,
        is_readable=True,
        has_mapping=False,
    )


def video() -> VideoRecord:
    return VideoRecord(
        video_id="L21_V001",
        video_path="raw/videos/L21/L21_V001.mp4",
        group_id="L21",
        fps=25.0,
        frame_count=5,
        duration_sec=0.2,
        width=10,
        height=10,
        video_codec="h264",
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
        has_audio=False,
        container="mp4",
        file_size_bytes=10,
        is_readable=True,
    )


class MappingValidationTests(unittest.TestCase):
    def test_detects_non_monotonic_and_out_of_bounds_mapping(self) -> None:
        report = validate_mapping(
            [keyframe(0), keyframe(1)],
            [video()],
            [
                MappingRecord("L21_V001", 0, 5, "map.json"),
                MappingRecord("L21_V001", 1, 4, "map.json"),
            ],
        )

        self.assertEqual(report.invalid_frame_count, 1)
        self.assertEqual(report.non_monotonic_count, 1)
        self.assertTrue(any(issue.code == "mapping_frame_out_of_bounds" for issue in report.issues))
