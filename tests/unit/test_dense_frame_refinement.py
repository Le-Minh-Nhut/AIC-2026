from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import Candidate, VideoRecord
from refinement.dense_frame_refiner import (
    DenseFrameRefiner,
    FrameScoringBranch,
    RefinementConfig,
    VisualFrameScorer,
)
from refinement.frame_sampler import FrameSampler
from refinement.video_decoder import DecodedFrame, DecodedVideoInfo
from retrieval.fusion import WeightedRRFConfig


class _SyntheticDecoder:
    def __init__(self, frame_count: int = 31) -> None:
        self.frame_count = frame_count
        self.requests: list[tuple[int, ...]] = []

    def inspect(self, video_path: Path) -> DecodedVideoInfo:
        return DecodedVideoInfo(fps=10.0, frame_count=self.frame_count)

    def decode_frames(self, video_path: Path, frame_ids: tuple[int, ...]) -> tuple[DecodedFrame, ...]:
        self.requests.append(frame_ids)
        return tuple(
            DecodedFrame(
                frame_id=frame_id,
                timestamp_sec=frame_id / 10.0,
                image=Image.new("RGB", (2, 2), color=(frame_id, 0, 0)),
            )
            for frame_id in frame_ids
        )


class _PeakEncoder:
    embedding_dimension = 2

    def __init__(self, peak: int) -> None:
        self.peak = peak
        self.image_calls = 0

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        vectors = np.array(
            [[max(1.0, 100.0 - abs(image.getpixel((0, 0))[0] - self.peak) * 10), 1.0] for image in images],
            dtype=np.float32,
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _video(video_id: str = "L21_V001") -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        video_path=f"raw/videos/{video_id}.mp4",
        group_id="L21",
        fps=10.0,
        frame_count=31,
        duration_sec=3.1,
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


def _coarse(video_id: str = "L21_V001") -> Candidate:
    return Candidate(
        keyframe_uid=f"{video_id}:000001",
        video_id=video_id,
        original_frame_id=10,
        timestamp_sec=1.0,
        keyframe_path=f"raw/keyframes/{video_id}/000001.jpg",
        score=0.5,
        rank=1,
        source="fgclip2",
    )


class DenseFrameRefinementTests(unittest.TestCase):
    def test_sampler_clamps_bounds_and_keeps_center(self) -> None:
        sampler = FrameSampler()

        sparse = sampler.sparse_frame_ids(1, fps=10.0, window_sec=1.0, sample_fps=2.0, frame_count=20)
        dense = sampler.dense_frame_ids(18, fps=10.0, window_sec=1.0, frame_count=20)

        self.assertEqual(sparse, (0, 1, 5, 10, 11))
        self.assertEqual(dense, tuple(range(8, 20)))

    def test_sparse_then_dense_refinement_selects_best_original_frame(self) -> None:
        decoder = _SyntheticDecoder()
        encoder = _PeakEncoder(peak=14)
        refiner = DenseFrameRefiner(
            decoder=decoder,
            sampler=FrameSampler(),
            scorer=VisualFrameScorer((FrameScoringBranch("fgclip2", encoder),)),
            video_records=(_video(),),
            data_root=Path("/fixture"),
            config=RefinementConfig(
                coarse_window_sec=1.0,
                sparse_fps=2.0,
                dense_window_sec=0.5,
                candidate_count=1,
            ),
        )

        result = refiner.refine("find fixture", (_coarse(),))

        self.assertEqual(result.failures, ())
        self.assertEqual(result.candidates[0].coarse_original_frame_id, 10)
        self.assertEqual(result.candidates[0].sparse_original_frame_id, 15)
        self.assertEqual(result.candidates[0].original_frame_id, 14)
        self.assertEqual(decoder.requests[0], (0, 5, 10, 15, 20))
        self.assertEqual(decoder.requests[1], tuple(range(10, 21)))
        self.assertGreaterEqual(encoder.image_calls, 2)

    def test_fusion_scores_frames_with_rrf_and_reports_missing_video(self) -> None:
        scorer = VisualFrameScorer(
            (
                FrameScoringBranch("fgclip2", _PeakEncoder(peak=14)),
                FrameScoringBranch("pecore", _PeakEncoder(peak=16)),
            ),
            fusion_config=WeightedRRFConfig(k=60, weights={"fgclip2": 1.0, "pecore": 1.0}),
        )
        scored = scorer.prepare("find fixture").score(
            _SyntheticDecoder().decode_frames(Path("fixture.mp4"), (14, 15, 16))
        )
        self.assertEqual(scored[0].source, "rrf_fusion")
        self.assertEqual(len(scored[0].source_scores), 2)

        refiner = DenseFrameRefiner(
            decoder=_SyntheticDecoder(),
            sampler=FrameSampler(),
            scorer=VisualFrameScorer((FrameScoringBranch("fgclip2", _PeakEncoder(14)),)),
            video_records=(_video(),),
            data_root=Path("/fixture"),
            config=RefinementConfig(1.0, 2.0, 0.5, 1),
        )
        result = refiner.refine("find fixture", (_coarse("L21_V404"),))
        self.assertEqual(result.candidates, ())
        self.assertIn("No video-manifest record", result.failures[0].error)
