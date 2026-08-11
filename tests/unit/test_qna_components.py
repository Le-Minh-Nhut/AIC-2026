from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import VideoRecord
from qna.answer_normalizer import normalize_answer
from qna.frame_selector import CandidateClipSelector, ClipSelectorConfig
from refinement.frame_sampler import FrameSampler
from refinement.video_decoder import DecodedFrame, DecodedVideoInfo


class _SyntheticDecoder:
    def inspect(self, video_path: Path) -> DecodedVideoInfo:
        return DecodedVideoInfo(fps=10.0, frame_count=41)

    def decode_frames(self, video_path: Path, frame_ids: tuple[int, ...]) -> tuple[DecodedFrame, ...]:
        return tuple(
            DecodedFrame(
                frame_id=frame_id,
                timestamp_sec=frame_id / 10.0,
                image=Image.new("RGB", (2, 2), color=(frame_id, 0, 0)),
            )
            for frame_id in frame_ids
        )


def _video() -> VideoRecord:
    return VideoRecord(
        video_id="L21_V001",
        video_path="raw/videos/L21_V001.mp4",
        group_id="L21",
        fps=10.0,
        frame_count=41,
        duration_sec=4.1,
        width=2,
        height=2,
        video_codec="fixture",
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
        has_audio=False,
        container="fixture",
        file_size_bytes=1,
        is_readable=True,
    )


class QnaComponentTests(unittest.TestCase):
    def test_clip_selector_samples_ordered_multi_frame_context_around_anchor(self) -> None:
        selector = CandidateClipSelector(
            decoder=_SyntheticDecoder(),
            sampler=FrameSampler(),
            video_records=(_video(),),
            data_root=Path("/fixture"),
            config=ClipSelectorConfig(window_sec=1.0, multi_frame_count=5),
        )

        clip = selector.select("L21_V001", 20)

        self.assertEqual(clip.frame_ids, (10, 15, 20, 25, 30))
        self.assertEqual(clip.timestamps_sec, (1.0, 1.5, 2.0, 2.5, 3.0))
        self.assertEqual(clip.frames[2].image.getpixel((0, 0))[0], 20)

    def test_answer_normalizer_handles_counts_boolean_colors_and_keeps_names_conservative(self) -> None:
        self.assertEqual(normalize_answer(" Five people! "), "5")
        self.assertEqual(normalize_answer("Năm"), "5")
        self.assertEqual(normalize_answer("5 people"), "5")
        self.assertEqual(normalize_answer("Có."), "yes")
        self.assertEqual(normalize_answer("KHÔNG"), "no")
        self.assertEqual(normalize_answer("màu Xanh Dương"), "blue")
        self.assertEqual(normalize_answer("It is blue."), "blue")
        self.assertEqual(normalize_answer("năm người chơi"), "5")
        self.assertEqual(normalize_answer("Nguyen Van A"), "nguyen van a")
