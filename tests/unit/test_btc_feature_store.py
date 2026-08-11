from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import KeyframeRecord
from indexing.feature_store import (
    BtcClipFeatureStore,
    FeatureMappingVerificationError,
    FeatureStoreValidationError,
)


def keyframe(uid: str, video_id: str, index: int, frame_id: int) -> KeyframeRecord:
    return KeyframeRecord(
        keyframe_uid=uid,
        video_id=video_id,
        keyframe_index=index,
        keyframe_path=f"raw/keyframes/{video_id}/{index:06d}.jpg",
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 25.0,
        width=10,
        height=10,
        file_size_bytes=10,
        is_readable=True,
        has_mapping=True,
    )


def write_order(path: Path, file_name: str, uids: list[str], verified: bool = True) -> Path:
    order_path = path / "order.json"
    order_path.write_text(
        json.dumps(
            {
                "feature_files": [file_name],
                "keyframe_uids": uids,
                "mapping_verified": verified,
                "verification_method": "fixture explicit UID order",
            }
        ),
        encoding="utf-8",
    )
    return order_path


class BtcClipFeatureStoreTests(unittest.TestCase):
    def test_uses_mmap_and_validates_verified_row_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_path = root / "features.npy"
            np.save(feature_path, np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
            records = [keyframe("L21_V001:000000", "L21_V001", 0, 0), keyframe("L21_V001:000001", "L21_V001", 1, 25)]
            store = BtcClipFeatureStore(
                [feature_path], records, write_order(root, feature_path.name, [record.keyframe_uid for record in records])
            )

            self.assertEqual(store.count, 2)
            self.assertEqual(store.dimension, 2)
            self.assertTrue(store.validation.uses_mmap)
            self.assertFalse(store.validation.vectors_are_l2_normalized)
            batch_ids, batch_vectors = next(store.iter_batches(10))
            self.assertEqual(batch_ids, ("L21_V001:000000", "L21_V001:000001"))
            self.assertEqual(batch_vectors.shape, (2, 2))

    def test_rejects_unverified_order_or_invalid_feature_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_path = root / "features.npy"
            np.save(feature_path, np.array([[np.nan, 1.0]], dtype=np.float32))
            records = [keyframe("L21_V001:000000", "L21_V001", 0, 0)]
            with self.assertRaises(FeatureMappingVerificationError):
                BtcClipFeatureStore(
                    [feature_path],
                    records,
                    write_order(root, feature_path.name, [records[0].keyframe_uid], verified=False),
                )
            with self.assertRaises(FeatureStoreValidationError):
                BtcClipFeatureStore(
                    [feature_path],
                    records,
                    write_order(root, feature_path.name, [records[0].keyframe_uid]),
                )

