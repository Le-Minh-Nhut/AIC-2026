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
from encoders.fgclip2 import FGCLIP2Encoder
from indexing.embedding_pipeline import EmbeddingRunConfig, OfflineKeyframeEmbedder
from indexing.faiss_index import FaissFlatIPIndex, build_faiss_flat_ip_index, load_faiss_flat_ip_index
from indexing.sharded_feature_store import ShardedNpyFeatureStore


class _FGBackend:
    embedding_dimension = None
    preprocessing_config = {"fixture": True}

    def __init__(self) -> None:
        self.image_calls = 0

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        return np.array([[float(image.getpixel((0, 0))[0]), 1.0] for image in images], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([[2.0, 3.0] for _ in texts], dtype=np.float32)


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


def _keyframe(root: Path, uid: str, index: int, red: int) -> KeyframeRecord:
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


class FGCLIP2PipelineTests(unittest.TestCase):
    def _encode_fixture(self, root: Path) -> tuple[tuple[KeyframeRecord, ...], Path, _FGBackend]:
        records = (
            _keyframe(root, "L21_V001:000002", 2, 30),
            _keyframe(root, "L21_V001:000000", 0, 10),
            _keyframe(root, "L21_V001:000001", 1, 20),
        )
        backend = _FGBackend()
        encoder = FGCLIP2Encoder(backend, batch_size=1)
        output = root / "embeddings"
        config = EmbeddingRunConfig(
            encoder_name="fgclip2_large",
            model_id="fixture-model",
            model_revision="fixture-revision",
            preprocessing_config=encoder.preprocessing_config,
            output_dir=output,
            data_root=root,
            shard_size=2,
            batch_size=1,
        )
        result = OfflineKeyframeEmbedder(encoder, config).run(records)
        self.assertEqual(result.count, 3)
        self.assertEqual(result.dimension, 2)
        return records, result.manifest_path, backend

    def test_encoder_normalizes_images_and_texts_with_runtime_dimension(self) -> None:
        backend = _FGBackend()
        encoder = FGCLIP2Encoder(backend, batch_size=1)
        image = Image.new("RGB", (4, 4), color=(3, 0, 0))

        image_vectors = encoder.encode_images([image, image])
        text_vectors = encoder.encode_texts(["fixture query"])

        np.testing.assert_allclose(np.linalg.norm(image_vectors, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(np.linalg.norm(text_vectors, axis=1), [1.0])
        self.assertEqual(encoder.embedding_dimension, 2)
        self.assertEqual(backend.image_calls, 2)

    def test_shards_resume_without_reencoding_and_preserve_uid_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            records, manifest_path, backend = self._encode_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["keyframe_uid_order"], sorted(record.keyframe_uid for record in records))
            self.assertEqual(len(manifest["shards"]), 2)
            self.assertEqual(backend.image_calls, 3)

            resumed_backend = _FGBackend()
            resumed_encoder = FGCLIP2Encoder(resumed_backend, batch_size=1)
            config = EmbeddingRunConfig(
                encoder_name="fgclip2_large",
                model_id="fixture-model",
                model_revision="fixture-revision",
                preprocessing_config=resumed_encoder.preprocessing_config,
                output_dir=manifest_path.parent,
                data_root=root,
                shard_size=2,
                batch_size=1,
            )
            resumed = OfflineKeyframeEmbedder(resumed_encoder, config).run(records, resume=True)
            store = ShardedNpyFeatureStore(manifest_path, records)

            self.assertEqual(resumed.encoded_shards, 0)
            self.assertEqual(resumed_backend.image_calls, 0)
            self.assertEqual(store.ordered_uids, tuple(sorted(record.keyframe_uid for record in records)))
            self.assertTrue(store.validation.vectors_are_l2_normalized)

    def test_builds_faiss_with_stable_keyframe_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            records, manifest_path, _ = self._encode_fixture(root)
            store = ShardedNpyFeatureStore(manifest_path, records)
            output = root / "index"
            with patch("indexing.faiss_index._load_faiss", return_value=_FakeFaiss):
                result = build_faiss_flat_ip_index(store, output, batch_size=2)
                loaded_index = load_faiss_flat_ip_index(output)

            ids = [json.loads(line) for line in (output / "index_ids.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result.dimension, store.dimension)
            self.assertEqual(ids, list(store.ordered_uids))
            loaded_index.validate_feature_store(store)
            fake_index = _FakeFaissIndex(store.dimension)
            for _, vectors in store.iter_batches(2):
                fake_index.add(vectors)
            index = FaissFlatIPIndex(fake_index, ids, store.dimension)
            index.validate_feature_store(store)
            self.assertEqual(index.search(np.array([1.0, 0.0], dtype=np.float32), 1)[0].item_id, ids[-1])
            self.assertEqual(loaded_index.search(np.array([1.0, 0.0], dtype=np.float32), 1)[0].item_id, ids[-1])
