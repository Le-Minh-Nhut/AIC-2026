from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import KeyframeRecord
from encoders.pecore import PECoreEncoder
from indexing.embedding_pipeline import EmbeddingRunConfig, OfflineKeyframeEmbedder
from indexing.faiss_index import build_faiss_flat_ip_index, load_faiss_flat_ip_index
from indexing.sharded_feature_store import ShardedNpyFeatureStore


class _PEBackend:
    embedding_dimension = None
    preprocessing_config = {"fixture": "pecore"}

    def __init__(self) -> None:
        self.image_calls = 0

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        return np.array(
            [[float(image.getpixel((0, 0))[0]), 1.0, 2.0] for image in images],
            dtype=np.float32,
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[4.0, 5.0, 6.0]], dtype=np.float32), (len(texts), 1))


class _FakeFaissIndex:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack((self.vectors, vectors))

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = vectors @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :top_k]
        return np.take_along_axis(scores, order, axis=1), order.astype(np.int64)


class _FakeFaiss:
    IndexFlatIP = _FakeFaissIndex

    @staticmethod
    def write_index(index: _FakeFaissIndex, path: str) -> None:
        with Path(path).open("wb") as handle:
            np.save(handle, index.vectors)

    @staticmethod
    def read_index(path: str) -> _FakeFaissIndex:
        with Path(path).open("rb") as handle:
            vectors = np.load(handle)
        index = _FakeFaissIndex(int(vectors.shape[1]))
        index.add(vectors)
        return index


def _record(root: Path, uid: str, index: int, red: int) -> KeyframeRecord:
    video_id = "L21_V001"
    image_path = root / "raw" / "keyframes" / video_id / f"{index:06d}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(red, 0, 0)).save(image_path)
    return KeyframeRecord(
        keyframe_uid=uid,
        video_id=video_id,
        keyframe_index=index,
        keyframe_path=str(image_path.relative_to(root)),
        original_frame_id=index * 10,
        timestamp_sec=float(index),
        width=4,
        height=4,
        file_size_bytes=image_path.stat().st_size,
        is_readable=True,
        has_mapping=True,
    )


class PECorePipelineTests(unittest.TestCase):
    def _run_fixture(self, root: Path) -> tuple[tuple[KeyframeRecord, ...], Path, _PEBackend]:
        records = (
            _record(root, "L21_V001:000001", 1, 20),
            _record(root, "L21_V001:000000", 0, 10),
        )
        backend = _PEBackend()
        encoder = PECoreEncoder(backend, batch_size=1)
        output = root / "embeddings"
        result = OfflineKeyframeEmbedder(
            encoder,
            EmbeddingRunConfig(
                encoder_name="pecore_g14_448",
                model_id="facebook/PE-Core-G14-448",
                model_revision=None,
                preprocessing_config=encoder.preprocessing_config,
                output_dir=output,
                data_root=root,
                shard_size=1,
                batch_size=1,
            ),
        ).run(records)
        self.assertEqual(result.dimension, 3)
        return records, result.manifest_path, backend

    def test_encoder_normalizes_images_and_texts_at_runtime_dimension(self) -> None:
        backend = _PEBackend()
        encoder = PECoreEncoder(backend, batch_size=1)
        image = Image.new("RGB", (4, 4), color=(7, 0, 0))

        images = encoder.encode_images([image, image])
        texts = encoder.encode_texts(["fixture text"])

        np.testing.assert_allclose(np.linalg.norm(images, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(np.linalg.norm(texts, axis=1), [1.0])
        self.assertEqual(encoder.embedding_dimension, 3)
        self.assertEqual(backend.image_calls, 2)

    def test_shards_resume_and_faiss_keep_stable_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            records, manifest_path, backend = self._run_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["encoder"]["name"], "pecore_g14_448")
            self.assertEqual(backend.image_calls, 2)

            resumed_backend = _PEBackend()
            resumed_encoder = PECoreEncoder(resumed_backend, batch_size=1)
            resumed = OfflineKeyframeEmbedder(
                resumed_encoder,
                EmbeddingRunConfig(
                    encoder_name="pecore_g14_448",
                    model_id="facebook/PE-Core-G14-448",
                    model_revision=None,
                    preprocessing_config=resumed_encoder.preprocessing_config,
                    output_dir=manifest_path.parent,
                    data_root=root,
                    shard_size=1,
                    batch_size=1,
                ),
            ).run(records, resume=True)
            store = ShardedNpyFeatureStore(manifest_path, records)
            self.assertEqual(resumed.encoded_shards, 0)
            self.assertEqual(resumed_backend.image_calls, 0)
            self.assertEqual(store.ordered_uids, tuple(sorted(record.keyframe_uid for record in records)))

            index_dir = root / "index"
            with patch("indexing.faiss_index._load_faiss", return_value=_FakeFaiss):
                build_faiss_flat_ip_index(store, index_dir, batch_size=1)
                index = load_faiss_flat_ip_index(index_dir)
            index.validate_feature_store(store)
            self.assertEqual(index.item_ids, store.ordered_uids)
