"""Validated, memory-mapped BTC CLIP feature storage with explicit row mappings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from domain.models import KeyframeRecord
from indexing.base import FeatureValidationReport


class FeatureStoreValidationError(ValueError):
    pass


class FeatureMappingVerificationError(FeatureStoreValidationError):
    pass


@dataclass(frozen=True, slots=True)
class FeatureOrderManifest:
    feature_files: tuple[Path, ...]
    keyframe_uids: tuple[str, ...]
    mapping_verified: bool
    verification_method: str


def _keyframe_record_from_dict(value: dict[str, object]) -> KeyframeRecord:
    required = {
        "keyframe_uid",
        "video_id",
        "keyframe_index",
        "keyframe_path",
        "original_frame_id",
        "timestamp_sec",
        "width",
        "height",
        "file_size_bytes",
        "is_readable",
        "has_mapping",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise FeatureStoreValidationError(
            f"Keyframe manifest row is missing retrieval metadata fields: {', '.join(missing)}"
        )
    return KeyframeRecord(
        keyframe_uid=str(value["keyframe_uid"]),
        video_id=str(value["video_id"]),
        keyframe_index=int(value["keyframe_index"]) if value["keyframe_index"] is not None else None,
        keyframe_path=str(value["keyframe_path"]),
        original_frame_id=(
            int(value["original_frame_id"]) if value["original_frame_id"] is not None else None
        ),
        timestamp_sec=float(value["timestamp_sec"]) if value["timestamp_sec"] is not None else None,
        width=int(value["width"]) if value["width"] is not None else None,
        height=int(value["height"]) if value["height"] is not None else None,
        file_size_bytes=int(value["file_size_bytes"]),
        is_readable=bool(value["is_readable"]),
        has_mapping=bool(value["has_mapping"]),
        image_mode=str(value["image_mode"]) if value.get("image_mode") is not None else None,
        read_error=str(value["read_error"]) if value.get("read_error") is not None else None,
    )


def load_keyframe_records_from_parquet(path: Path) -> tuple[KeyframeRecord, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise FeatureStoreValidationError(
            "Reading keyframes_manifest.parquet requires pyarrow. Install project dependencies first."
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Keyframe manifest does not exist: {path}")
    rows = pq.read_table(path).to_pylist()
    return tuple(_keyframe_record_from_dict(row) for row in rows)


def load_feature_order_manifest(path: Path) -> FeatureOrderManifest:
    if not path.is_file():
        raise FileNotFoundError(f"BTC feature-order manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureMappingVerificationError(f"Invalid BTC feature-order manifest: {error}") from error
    if not isinstance(value, dict):
        raise FeatureMappingVerificationError("BTC feature-order manifest must be a JSON object")
    files = value.get("feature_files")
    uids = value.get("keyframe_uids")
    verified = value.get("mapping_verified")
    verification_method = value.get("verification_method")
    if not isinstance(files, list) or not all(isinstance(item, str) and item for item in files):
        raise FeatureMappingVerificationError("feature_files must be a non-empty string list")
    if not isinstance(uids, list) or not all(isinstance(item, str) and item for item in uids):
        raise FeatureMappingVerificationError("keyframe_uids must be a non-empty string list")
    if verified is not True:
        raise FeatureMappingVerificationError(
            "Row mapping is unresolved: mapping_verified must be explicitly true"
        )
    if not isinstance(verification_method, str) or not verification_method.strip():
        raise FeatureMappingVerificationError(
            "A non-empty verification_method is required before BTC retrieval can run"
        )
    return FeatureOrderManifest(
        feature_files=tuple((path.parent / item).resolve() for item in files),
        keyframe_uids=tuple(uids),
        mapping_verified=True,
        verification_method=verification_method.strip(),
    )


class BtcClipFeatureStore:
    """Read-only BTC feature store that never infers row-to-keyframe ordering."""

    def __init__(
        self,
        feature_paths: Sequence[Path],
        keyframes: Sequence[KeyframeRecord],
        feature_order_manifest: Path,
        mmap: bool = True,
        validation_batch_size: int = 8192,
    ) -> None:
        if not feature_paths:
            raise FeatureStoreValidationError("At least one BTC CLIP .npy file is required")
        if validation_batch_size < 1:
            raise ValueError("validation_batch_size must be at least 1")
        self._feature_paths = tuple(path.resolve() for path in feature_paths)
        self._order = load_feature_order_manifest(feature_order_manifest)
        if self._order.feature_files != self._feature_paths:
            raise FeatureMappingVerificationError(
                "Feature files do not exactly match the verified feature-order manifest; "
                "refusing to assume row order"
            )
        metadata_by_uid = {record.keyframe_uid: record for record in keyframes}
        if len(metadata_by_uid) != len(keyframes):
            raise FeatureMappingVerificationError("Keyframe manifest contains duplicate keyframe_uid values")
        if len(set(self._order.keyframe_uids)) != len(self._order.keyframe_uids):
            raise FeatureMappingVerificationError("Feature-order manifest contains duplicate keyframe_uids")
        unknown_uids = sorted(set(self._order.keyframe_uids) - set(metadata_by_uid))
        if unknown_uids:
            raise FeatureMappingVerificationError(
                "Verified row mapping references keyframes absent from metadata: "
                + ", ".join(unknown_uids[:10])
            )
        self._metadata_by_uid = {uid: metadata_by_uid[uid] for uid in self._order.keyframe_uids}
        incomplete = [
            uid
            for uid, record in self._metadata_by_uid.items()
            if not record.has_mapping
            or record.original_frame_id is None
            or record.timestamp_sec is None
        ]
        if incomplete:
            raise FeatureMappingVerificationError(
                "BTC retrieval requires each ordered keyframe to map to video/frame/timestamp: "
                + ", ".join(incomplete[:10])
            )
        self._arrays = tuple(self._load_array(path, mmap) for path in self._feature_paths)
        self._offsets = self._build_offsets()
        if self.count != len(self._order.keyframe_uids):
            raise FeatureMappingVerificationError(
                f"Feature rows ({self.count}) do not equal verified keyframe UID count "
                f"({len(self._order.keyframe_uids)})"
            )
        self._validation = self._validate(validation_batch_size, mmap)

    @property
    def count(self) -> int:
        return sum(array.shape[0] for array in self._arrays)

    @property
    def dimension(self) -> int:
        return int(self._arrays[0].shape[1])

    @property
    def validation(self) -> FeatureValidationReport:
        return self._validation

    @property
    def verification_method(self) -> str:
        return self._order.verification_method

    def metadata_for_id(self, item_id: str) -> KeyframeRecord:
        try:
            return self._metadata_by_uid[item_id]
        except KeyError as error:
            raise KeyError(f"Unknown BTC feature item ID: {item_id}") from error

    def iter_batches(self, batch_size: int) -> Iterator[tuple[tuple[str, ...], np.ndarray]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        global_offset = 0
        for array in self._arrays:
            for start in range(0, array.shape[0], batch_size):
                end = min(start + batch_size, array.shape[0])
                ids = self._order.keyframe_uids[global_offset + start : global_offset + end]
                yield ids, np.asarray(array[start:end], dtype=np.float32)
            global_offset += array.shape[0]

    @staticmethod
    def _load_array(path: Path, mmap: bool) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"BTC CLIP feature file does not exist: {path}")
        try:
            array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise FeatureStoreValidationError(f"Cannot load BTC CLIP features from {path}: {error}") from error
        if not isinstance(array, np.ndarray):
            raise FeatureStoreValidationError(
                f"BTC CLIP feature file must contain one .npy matrix, received {type(array).__name__}"
            )
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise FeatureStoreValidationError(
                f"BTC CLIP features must have non-empty shape [rows, dimension], received {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise FeatureStoreValidationError(
                f"BTC CLIP features must use floating dtype, received {array.dtype} in {path}"
            )
        return array

    def _build_offsets(self) -> tuple[int, ...]:
        dimensions = {array.shape[1] for array in self._arrays}
        if len(dimensions) != 1:
            raise FeatureStoreValidationError(
                f"BTC CLIP feature shards have inconsistent dimensions: {sorted(dimensions)}"
            )
        offsets = [0]
        for array in self._arrays:
            offsets.append(offsets[-1] + array.shape[0])
        return tuple(offsets)

    def _validate(self, batch_size: int, mmap: bool) -> FeatureValidationReport:
        nan_count = 0
        inf_count = 0
        zero_vector_count = 0
        min_norm = math.inf
        max_norm = 0.0
        for _, vectors in self.iter_batches(batch_size):
            finite = np.isfinite(vectors)
            nan_count += int(np.isnan(vectors).sum())
            inf_count += int(np.isinf(vectors).sum())
            if not finite.all():
                continue
            norms = np.linalg.norm(vectors, axis=1)
            zero_vector_count += int(np.count_nonzero(norms == 0))
            if norms.size:
                min_norm = min(min_norm, float(norms.min()))
                max_norm = max(max_norm, float(norms.max()))
        if nan_count or inf_count:
            raise FeatureStoreValidationError(
                f"BTC CLIP features contain NaN ({nan_count}) or Inf ({inf_count}) values"
            )
        if zero_vector_count:
            raise FeatureStoreValidationError(
                f"BTC CLIP features contain {zero_vector_count} zero-norm vectors"
            )
        return FeatureValidationReport(
            count=self.count,
            dimension=self.dimension,
            dtypes=tuple(sorted({str(array.dtype) for array in self._arrays})),
            nan_count=nan_count,
            inf_count=inf_count,
            zero_vector_count=zero_vector_count,
            min_norm=min_norm,
            max_norm=max_norm,
            vectors_are_l2_normalized=bool(max_norm - 1.0 < 1e-3 and 1.0 - min_norm < 1e-3),
            uses_mmap=mmap,
        )
