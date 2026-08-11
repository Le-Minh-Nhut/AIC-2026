from __future__ import annotations

import sys
import unittest
from typing import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import Candidate, KeyframeRecord
from encoders.btc_clip import BtcClipTextEncoder
from indexing.exact_index import ExactCosineIndex
from retrieval.temporal_nms import temporal_nms
from retrieval.video_aggregation import aggregate_video_candidates


class _ArrayStore:
    def __init__(self) -> None:
        self.dimension = 2
        self._vectors = np.array([[2.0, 0.0], [0.0, 5.0], [1.0, 1.0]], dtype=np.float32)
        self._ids = ("one", "two", "three")

    @property
    def count(self) -> int:
        return len(self._ids)

    def iter_batches(self, batch_size: int):
        yield self._ids, self._vectors

    def metadata_for_id(self, item_id: str) -> KeyframeRecord:
        raise KeyError(item_id)


class _TextBackend:
    embedding_dimension = None

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([[3.0, 4.0] for _ in texts], dtype=np.float32)


def candidate(video_id: str, frame_id: int, score: float, rank: int) -> Candidate:
    return Candidate(
        keyframe_uid=f"{video_id}:{frame_id:06d}",
        video_id=video_id,
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        keyframe_path=f"raw/keyframes/{video_id}/{frame_id:06d}.jpg",
        score=score,
        rank=rank,
        source="fixture",
    )


class RetrievalCoreTests(unittest.TestCase):
    def test_btc_text_adapter_normalizes_without_loading_a_model(self) -> None:
        encoder = BtcClipTextEncoder(_TextBackend())

        result = encoder.encode_texts(["a fixture query"])

        np.testing.assert_allclose(result, np.array([[0.6, 0.8]], dtype=np.float32), rtol=1e-6)
        self.assertEqual(encoder.embedding_dimension, 2)

    def test_exact_cosine_returns_stable_best_result(self) -> None:
        index = ExactCosineIndex(_ArrayStore(), batch_size=2)

        hits = index.search(np.array([0.0, 2.0], dtype=np.float32), top_k=2)

        self.assertEqual([hit.item_id for hit in hits], ["two", "three"])
        self.assertAlmostEqual(hits[0].score, 1.0)

    def test_temporal_nms_and_video_aggregation_preserve_diversity(self) -> None:
        candidates = [
            candidate("L21_V001", 10, 0.95, 1),
            candidate("L21_V001", 15, 0.90, 2),
            candidate("L21_V002", 10, 0.85, 3),
            candidate("L21_V001", 50, 0.70, 4),
        ]

        selected = temporal_nms(candidates, window_sec=2.0)
        videos = aggregate_video_candidates(selected, method="mean_top_m", top_m=2)

        self.assertEqual([item.keyframe_uid for item in selected], ["L21_V001:000010", "L21_V002:000010", "L21_V001:000050"])
        self.assertEqual([item.rank for item in selected], [1, 2, 3])
        self.assertEqual(videos[0].video_id, "L21_V002")
        self.assertAlmostEqual(videos[1].score, 0.825)
