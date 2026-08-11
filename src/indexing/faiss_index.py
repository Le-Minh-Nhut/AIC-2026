"""Persistent FAISS IndexFlatIP adapter with stable keyframe UID mapping."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from domain.models import SearchHit
from domain.protocols import FeatureStore
from encoders.base import l2_normalize


class FaissUnavailableError(RuntimeError):
    pass


class FaissIndexValidationError(ValueError):
    pass


def _load_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise FaissUnavailableError(
            "FAISS IndexFlatIP requires optional dependency faiss-cpu or faiss-gpu"
        ) from error
    return faiss


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, path)


@dataclass(frozen=True, slots=True)
class FaissBuildResult:
    index_path: Path
    manifest_path: Path
    count: int
    dimension: int


class FaissFlatIPIndex:
    def __init__(self, index: Any, item_ids: Sequence[str], dimension: int) -> None:
        if dimension < 1:
            raise ValueError("FAISS index dimension must be positive")
        self._index = index
        self._item_ids = tuple(item_ids)
        self._dimension = dimension
        if len(self._item_ids) != int(index.ntotal):
            raise FaissIndexValidationError("FAISS vector count does not match persisted keyframe UID count")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def item_ids(self) -> tuple[str, ...]:
        return self._item_ids

    def search(self, query: np.ndarray, top_k: int) -> Sequence[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, self._dimension):
            raise ValueError(
                f"FAISS query must have shape [1, {self._dimension}], received {vector.shape}"
            )
        scores, rows = self._index.search(l2_normalize(vector), min(top_k, len(self._item_ids)))
        hits = [
            SearchHit(self._item_ids[int(row)], float(score))
            for score, row in zip(scores[0], rows[0])
            if int(row) >= 0
        ]
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.item_id)))

    def validate_feature_store(self, feature_store: FeatureStore) -> None:
        ordered_uids = getattr(feature_store, "ordered_uids", None)
        if feature_store.dimension != self._dimension or feature_store.count != len(self._item_ids):
            raise FaissIndexValidationError("FAISS index does not match feature-store count or dimension")
        if ordered_uids is not None and tuple(ordered_uids) != self._item_ids:
            raise FaissIndexValidationError("FAISS row-to-keyframe UID order does not match feature store")


def build_faiss_flat_ip_index(
    feature_store: FeatureStore,
    output_dir: Path,
    batch_size: int,
    overwrite: bool = False,
) -> FaissBuildResult:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    ids_path = output_dir / "index_ids.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not overwrite and any(path.exists() for path in (index_path, ids_path, manifest_path)):
        raise FileExistsError("FAISS output already exists; pass overwrite=True to replace it")
    faiss = _load_faiss()
    index = faiss.IndexFlatIP(feature_store.dimension)
    temporary_ids = ids_path.with_suffix(ids_path.suffix + ".tmp")
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    count = 0
    try:
        with temporary_ids.open("w", encoding="utf-8", newline="\n") as handle:
            for item_ids, vectors in feature_store.iter_batches(batch_size):
                normalized = l2_normalize(vectors).astype(np.float32, copy=False)
                index.add(normalized)
                for item_id in item_ids:
                    handle.write(json.dumps(item_id, ensure_ascii=False))
                    handle.write("\n")
                count += len(item_ids)
        if count != feature_store.count or int(index.ntotal) != count:
            raise FaissIndexValidationError("FAISS build count does not match feature store")
        faiss.write_index(index, str(temporary_index))
        os.replace(temporary_index, index_path)
        os.replace(temporary_ids, ids_path)
        manifest = {
            "schema_version": "1.0",
            "index_type": "IndexFlatIP",
            "dimension": feature_store.dimension,
            "count": count,
            "normalized": True,
            "index_file": index_path.name,
            "index_sha256": _sha256_file(index_path),
            "ids_file": ids_path.name,
            "ids_sha256": _sha256_file(ids_path),
            "feature_manifest_path": str(getattr(feature_store, "manifest_path", "NOT PRESENT")),
            "feature_manifest_sha256": (
                _sha256_file(feature_store.manifest_path)
                if getattr(feature_store, "manifest_path", None) is not None
                else "NOT PRESENT"
            ),
        }
        _write_json_atomic(manifest_path, manifest)
    except Exception:
        temporary_ids.unlink(missing_ok=True)
        temporary_index.unlink(missing_ok=True)
        raise
    return FaissBuildResult(index_path, manifest_path, count, feature_store.dimension)


def load_faiss_flat_ip_index(output_dir: Path) -> FaissFlatIPIndex:
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FaissIndexValidationError(f"Invalid FAISS manifest {manifest_path}: {error}") from error
    if manifest.get("schema_version") != "1.0" or manifest.get("index_type") != "IndexFlatIP":
        raise FaissIndexValidationError("Unsupported FAISS index manifest")
    index_path = output_dir / str(manifest.get("index_file", ""))
    ids_path = output_dir / str(manifest.get("ids_file", ""))
    if not index_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError("FAISS index or ID mapping file is missing")
    if manifest.get("index_sha256") != _sha256_file(index_path):
        raise FaissIndexValidationError("FAISS index checksum does not match manifest")
    if manifest.get("ids_sha256") != _sha256_file(ids_path):
        raise FaissIndexValidationError("FAISS ID mapping checksum does not match manifest")
    try:
        item_ids = tuple(json.loads(line) for line in ids_path.read_text(encoding="utf-8").splitlines() if line)
    except json.JSONDecodeError as error:
        raise FaissIndexValidationError(f"Invalid FAISS ID mapping: {error}") from error
    if len(item_ids) != len(set(item_ids)) or not all(isinstance(item, str) for item in item_ids):
        raise FaissIndexValidationError("FAISS ID mapping must contain unique keyframe UID strings")
    index = _load_faiss().read_index(str(index_path))
    if int(manifest.get("count", -1)) != len(item_ids) or int(index.ntotal) != len(item_ids):
        raise FaissIndexValidationError("FAISS manifest count, index count, and UID mapping count differ")
    if int(getattr(index, "d", -1)) != int(manifest["dimension"]):
        raise FaissIndexValidationError("FAISS index dimension does not match manifest")
    return FaissFlatIPIndex(index, item_ids, int(manifest["dimension"]))
