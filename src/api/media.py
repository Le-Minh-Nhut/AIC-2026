"""Manifest-backed media access for the browser; never accept filesystem paths."""

from __future__ import annotations

from pathlib import Path

from data.video_repository import load_video_records_from_parquet
from domain.models import KeyframeRecord, VideoRecord
from indexing.feature_store import load_keyframe_records_from_parquet


class MediaLookupError(ValueError):
    """Raised when a requested manifest item cannot be served safely."""


class ManifestMediaRepository:
    """Resolve media by stable IDs using M1 manifests, not arbitrary client paths."""

    def __init__(self, data_root: Path, keyframe_manifest: Path, video_manifest: Path) -> None:
        self._data_root = data_root.resolve()
        self._keyframe_manifest = keyframe_manifest
        self._video_manifest = video_manifest
        self._keyframes: dict[str, KeyframeRecord] | None = None
        self._videos: dict[str, VideoRecord] | None = None

    def keyframe_path(self, keyframe_uid: str) -> Path:
        if self._keyframes is None:
            records = load_keyframe_records_from_parquet(self._keyframe_manifest)
            self._keyframes = {record.keyframe_uid: record for record in records}
        record = self._keyframes.get(keyframe_uid)
        if record is None:
            raise MediaLookupError(f"Unknown keyframe_uid: {keyframe_uid}")
        return self._safe_path(record.keyframe_path, "keyframe")

    def video_path(self, video_id: str) -> Path:
        if self._videos is None:
            records = load_video_records_from_parquet(self._video_manifest)
            self._videos = {record.video_id: record for record in records}
        record = self._videos.get(video_id)
        if record is None:
            raise MediaLookupError(f"Unknown video_id: {video_id}")
        return self._safe_path(record.video_path, "video")

    def _safe_path(self, manifest_path: str, label: str) -> Path:
        candidate = Path(manifest_path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self._data_root / candidate).resolve()
        )
        if not resolved.is_relative_to(self._data_root):
            raise MediaLookupError(f"Refusing {label} path outside data root")
        if not resolved.is_file():
            raise MediaLookupError(f"{label.capitalize()} file is missing: {manifest_path}")
        return resolved
