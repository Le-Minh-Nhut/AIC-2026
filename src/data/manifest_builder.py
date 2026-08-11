"""Build stable video and keyframe Parquet manifests from extracted data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data.keyframe_manifest import scan_keyframe_records
from data.keyframe_mapping import MappingLoadResult, load_mapping_records, validate_mapping
from data.repositories import parquet_schema, write_parquet_records
from data.video_manifest import scan_video_records
from domain.models import KeyframeRecord, MappingValidationReport, VideoRecord


VIDEO_SCHEMA = [
    ("video_id", "string"),
    ("video_path", "string"),
    ("group_id", "string"),
    ("fps", "float64"),
    ("frame_count", "int64"),
    ("duration_sec", "float64"),
    ("width", "int64"),
    ("height", "int64"),
    ("video_codec", "string"),
    ("audio_codec", "string"),
    ("audio_sample_rate", "int64"),
    ("audio_channels", "int64"),
    ("has_audio", "bool"),
    ("container", "string"),
    ("file_size_bytes", "int64"),
    ("is_readable", "bool"),
    ("bitrate", "int64"),
    ("probe_error", "string"),
]

KEYFRAME_SCHEMA = [
    ("keyframe_uid", "string"),
    ("video_id", "string"),
    ("keyframe_index", "int64"),
    ("keyframe_path", "string"),
    ("original_frame_id", "int64"),
    ("timestamp_sec", "float64"),
    ("width", "int64"),
    ("height", "int64"),
    ("file_size_bytes", "int64"),
    ("is_readable", "bool"),
    ("has_mapping", "bool"),
    ("image_mode", "string"),
    ("read_error", "string"),
]


@dataclass(frozen=True, slots=True)
class ManifestBuildResult:
    videos: tuple[VideoRecord, ...]
    keyframes: tuple[KeyframeRecord, ...]
    mapping_load: MappingLoadResult
    mapping_validation: MappingValidationReport


def collect_manifest_records(data_root: Path, timestamp_tolerance_seconds: float = 1.0) -> ManifestBuildResult:
    videos = tuple(scan_video_records(data_root))
    mapping_load = load_mapping_records(data_root / "raw" / "map_keyframes")
    keyframes = tuple(scan_keyframe_records(data_root, videos, mapping_load.records))
    validation = validate_mapping(
        keyframes,
        videos,
        mapping_load.records,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    return ManifestBuildResult(videos, keyframes, mapping_load, validation)


def build_manifests(data_root: Path, timestamp_tolerance_seconds: float = 1.0) -> ManifestBuildResult:
    result = collect_manifest_records(data_root, timestamp_tolerance_seconds)
    manifests_root = data_root / "manifests"
    write_parquet_records(
        manifests_root / "videos_manifest.parquet",
        (record.as_dict() for record in result.videos),
        parquet_schema(VIDEO_SCHEMA),
    )
    write_parquet_records(
        manifests_root / "keyframes_manifest.parquet",
        (record.as_dict() for record in result.keyframes),
        parquet_schema(KEYFRAME_SCHEMA),
    )
    mapping_audit = {
        "mapping_load": {
            "record_count": len(result.mapping_load.records),
            "unsupported_files": list(result.mapping_load.unsupported_files),
            "malformed_rows": list(result.mapping_load.malformed_rows),
        },
        "validation": result.mapping_validation.as_dict(),
    }
    (manifests_root / "mapping_validation.json").write_text(
        json.dumps(mapping_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
