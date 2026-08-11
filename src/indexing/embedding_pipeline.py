"""Resumable offline keyframe embedding pipeline with stable shard metadata."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from domain.models import KeyframeRecord
from domain.protocols import ImageTextEncoder
from encoders.base import l2_normalize


class EmbeddingPipelineError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EmbeddingRunConfig:
    encoder_name: str
    model_id: str
    model_revision: str | None
    preprocessing_config: dict[str, object]
    output_dir: Path
    data_root: Path
    shard_size: int
    batch_size: int
    storage_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.shard_size < 1 or self.batch_size < 1:
            raise ValueError("shard_size and batch_size must be at least 1")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be float16 or float32")

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.storage_dtype)

    def immutable_config(self) -> dict[str, object]:
        return {
            "encoder_name": self.encoder_name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "preprocessing_config": self.preprocessing_config,
            "storage_dtype": self.storage_dtype,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    manifest_path: Path
    count: int
    dimension: int | None
    completed_shards: int
    encoded_shards: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _git_commit(data_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(data_root.parent), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


class OfflineKeyframeEmbedder:
    """Encodes each keyframe once, writing recoverable .npy shards atomically."""

    def __init__(self, encoder: ImageTextEncoder, config: EmbeddingRunConfig) -> None:
        self._encoder = encoder
        self._config = config

    def run(self, keyframes: Sequence[KeyframeRecord], resume: bool = True) -> EmbeddingResult:
        ordered = tuple(sorted(keyframes, key=lambda record: record.keyframe_uid))
        self._validate_keyframes(ordered)
        manifest_path = self._config.output_dir / "manifest.json"
        expected_uids = [record.keyframe_uid for record in ordered]
        expected_config_hash = _canonical_json_hash(self._config.immutable_config())
        if manifest_path.exists():
            if not resume:
                raise EmbeddingPipelineError(
                    f"Embedding manifest already exists: {manifest_path}. Use resume or a new output directory."
                )
            manifest = self._load_manifest(manifest_path)
            self._validate_existing_manifest(manifest, expected_uids, expected_config_hash)
        else:
            manifest = self._new_manifest(expected_uids, expected_config_hash)
            self._write_json_atomic(manifest_path, manifest)
        completed_rows = self._validate_completed_shards(manifest, ordered)
        encoded_shards = 0
        for start in range(completed_rows, len(ordered), self._config.shard_size):
            records = ordered[start : start + self._config.shard_size]
            shard = self._encode_shard(records, start, len(manifest["shards"]))
            manifest["shards"].append(shard)
            manifest["count"] = start + len(records)
            manifest["embedding_dimension"] = shard["dimension"]
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            self._write_json_atomic(manifest_path, manifest)
            encoded_shards += 1
        return EmbeddingResult(
            manifest_path=manifest_path,
            count=int(manifest["count"]),
            dimension=manifest["embedding_dimension"],
            completed_shards=len(manifest["shards"]),
            encoded_shards=encoded_shards,
        )

    def _new_manifest(self, ordered_uids: list[str], config_hash: str) -> dict[str, Any]:
        timestamp = datetime.now(UTC).isoformat()
        return {
            "schema_version": "1.0",
            "created_at": timestamp,
            "updated_at": timestamp,
            "git_commit": _git_commit(self._config.data_root),
            "encoder": {
                "name": self._config.encoder_name,
                "model_id": self._config.model_id,
                "model_revision": self._config.model_revision,
                "preprocessing_config": self._config.preprocessing_config,
                "config_hash": config_hash,
            },
            "storage_dtype": self._config.storage_dtype,
            "normalized": True,
            "embedding_dimension": None,
            "count": 0,
            "source": {
                "keyframe_uid_order": ordered_uids,
                "keyframe_uid_order_sha256": _canonical_json_hash(ordered_uids),
            },
            "shards": [],
        }

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EmbeddingPipelineError(f"Invalid embedding manifest {path}: {error}") from error
        if not isinstance(value, dict):
            raise EmbeddingPipelineError("Embedding manifest must be a JSON object")
        return value

    def _validate_existing_manifest(
        self,
        manifest: dict[str, Any],
        expected_uids: list[str],
        expected_config_hash: str,
    ) -> None:
        if manifest.get("schema_version") != "1.0":
            raise EmbeddingPipelineError("Unsupported embedding manifest schema version")
        if manifest.get("encoder", {}).get("config_hash") != expected_config_hash:
            raise EmbeddingPipelineError(
                "Embedding manifest encoder/preprocessing/storage configuration differs; refusing unsafe resume"
            )
        source = manifest.get("source", {})
        if source.get("keyframe_uid_order") != expected_uids:
            raise EmbeddingPipelineError(
                "Keyframe UID order changed since encoding began; refusing unsafe resume"
            )
        if source.get("keyframe_uid_order_sha256") != _canonical_json_hash(expected_uids):
            raise EmbeddingPipelineError("Embedding manifest keyframe UID hash is invalid")

    def _validate_completed_shards(
        self,
        manifest: dict[str, Any],
        ordered: Sequence[KeyframeRecord],
    ) -> int:
        completed_rows = 0
        expected_dimension = manifest.get("embedding_dimension")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise EmbeddingPipelineError("Embedding manifest shards must be a list")
        for shard_index, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise EmbeddingPipelineError(f"Shard {shard_index} metadata must be an object")
            start = shard.get("row_start")
            end = shard.get("row_end")
            uids = shard.get("keyframe_uids")
            if start != completed_rows or not isinstance(end, int) or end <= start:
                raise EmbeddingPipelineError(f"Shard {shard_index} has non-contiguous row range")
            if uids != [record.keyframe_uid for record in ordered[start:end]]:
                raise EmbeddingPipelineError(f"Shard {shard_index} keyframe UID mapping no longer matches source")
            file_path = self._config.output_dir / str(shard.get("file", ""))
            metadata_path = self._config.output_dir / str(shard.get("metadata_file", ""))
            if not file_path.is_file() or not metadata_path.is_file():
                raise EmbeddingPipelineError(f"Completed shard {shard_index} is missing file or metadata")
            if shard.get("sha256") != _sha256_file(file_path):
                raise EmbeddingPipelineError(f"Completed shard {shard_index} checksum does not match manifest")
            try:
                array = np.load(file_path, mmap_mode="r", allow_pickle=False)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise EmbeddingPipelineError(f"Completed shard {shard_index} cannot be read: {error}") from error
            if array.shape[0] != end - start or array.ndim != 2:
                raise EmbeddingPipelineError(f"Completed shard {shard_index} has invalid array shape {array.shape}")
            if str(array.dtype) != self._config.storage_dtype:
                raise EmbeddingPipelineError(f"Completed shard {shard_index} has dtype {array.dtype}")
            if metadata.get("keyframe_uids") != uids:
                raise EmbeddingPipelineError(f"Completed shard {shard_index} metadata UID list is inconsistent")
            if expected_dimension is not None and array.shape[1] != expected_dimension:
                raise EmbeddingPipelineError(f"Completed shard {shard_index} has inconsistent embedding dimension")
            completed_rows = end
        if completed_rows != manifest.get("count"):
            raise EmbeddingPipelineError("Embedding manifest count does not match completed shard ranges")
        return completed_rows

    def _encode_shard(
        self,
        records: Sequence[KeyframeRecord],
        row_start: int,
        shard_index: int,
    ) -> dict[str, Any]:
        shard_name = f"shard_{shard_index:05d}.npy"
        metadata_name = f"shard_{shard_index:05d}.json"
        output_path = self._config.output_dir / shard_name
        metadata_path = self._config.output_dir / metadata_name
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        matrix: np.memmap | None = None
        dimension: int | None = None
        try:
            for batch_start in range(0, len(records), self._config.batch_size):
                batch_records = records[batch_start : batch_start + self._config.batch_size]
                images = [self._load_image(record) for record in batch_records]
                vectors = l2_normalize(self._encoder.encode_images(images))
                if vectors.shape[0] != len(batch_records):
                    raise EmbeddingPipelineError("Encoder returned a row count different from input image batch")
                if dimension is None:
                    dimension = int(vectors.shape[1])
                    matrix = np.lib.format.open_memmap(
                        temporary_path,
                        mode="w+",
                        dtype=self._config.numpy_dtype,
                        shape=(len(records), dimension),
                    )
                if vectors.shape[1] != dimension or matrix is None:
                    raise EmbeddingPipelineError("Encoder embedding dimension changed within a shard")
                matrix[batch_start : batch_start + len(batch_records)] = vectors.astype(
                    self._config.numpy_dtype,
                    copy=False,
                )
            if matrix is None or dimension is None:
                raise EmbeddingPipelineError("Cannot write an empty embedding shard")
            matrix.flush()
            del matrix
            os.replace(temporary_path, output_path)
            shard = {
                "file": shard_name,
                "metadata_file": metadata_name,
                "row_start": row_start,
                "row_end": row_start + len(records),
                "count": len(records),
                "dimension": dimension,
                "dtype": self._config.storage_dtype,
                "normalized": True,
                "keyframe_uids": [record.keyframe_uid for record in records],
                "sha256": _sha256_file(output_path),
            }
            self._write_json_atomic(metadata_path, shard)
            return shard
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _load_image(self, record: KeyframeRecord) -> Image.Image:
        if not record.is_readable:
            raise EmbeddingPipelineError(f"Keyframe is marked unreadable: {record.keyframe_uid}")
        if not record.has_mapping or record.original_frame_id is None or record.timestamp_sec is None:
            raise EmbeddingPipelineError(
                f"Keyframe lacks required video/frame/timestamp mapping: {record.keyframe_uid}"
            )
        path = self._config.data_root / record.keyframe_path
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise EmbeddingPipelineError(f"Cannot load keyframe {record.keyframe_uid}: {error}") from error

    @staticmethod
    def _validate_keyframes(keyframes: Sequence[KeyframeRecord]) -> None:
        if not keyframes:
            raise EmbeddingPipelineError("No keyframes were provided for encoding")
        uids = [record.keyframe_uid for record in keyframes]
        if len(set(uids)) != len(uids):
            raise EmbeddingPipelineError("Keyframe UIDs must be unique before embedding")

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, path)
