"""Schema-aware keyframe to original-frame mapping discovery and validation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from domain.models import (
    KeyframeRecord,
    MappingRecord,
    MappingValidationReport,
    ValidationIssue,
    VideoRecord,
)


VIDEO_ID_PATTERN = re.compile(r"L\d+_V\d+", re.IGNORECASE)
KEYFRAME_INDEX_KEYS = ("keyframe_index", "keyframe_id", "keyframe", "index", "image_index")
FRAME_ID_KEYS = ("original_frame_id", "frame_id", "frame_index", "frame_idx", "frame")
VIDEO_ID_KEYS = ("video_id", "video", "video_name", "videoid")


@dataclass(frozen=True, slots=True)
class MappingLoadResult:
    records: tuple[MappingRecord, ...]
    unsupported_files: tuple[str, ...]
    malformed_rows: tuple[str, ...]


def infer_video_id(value: str) -> str | None:
    match = VIDEO_ID_PATTERN.search(value)
    return match.group(0).upper() if match else None


def infer_keyframe_index(value: str) -> int | None:
    match = re.search(r"(\d+)(?!.*\d)", value)
    return int(match.group(1)) if match else None


def _first_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    normalized = {str(key).casefold(): value for key, value in mapping.items()}
    for key in keys:
        if key in normalized:
            return normalized[key]
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(str(value))
    except (TypeError, ValueError):
        return None
    return converted


def _record_from_mapping_row(row: dict[str, Any], source_path: Path) -> MappingRecord | None:
    video_value = _first_value(row, VIDEO_ID_KEYS)
    index_value = _first_value(row, KEYFRAME_INDEX_KEYS)
    frame_value = _first_value(row, FRAME_ID_KEYS)
    if video_value is None or index_value is None or frame_value is None:
        return None
    video_id = infer_video_id(str(video_value)) or str(video_value).strip()
    keyframe_index = _as_int(index_value)
    original_frame_id = _as_int(frame_value)
    if not video_id or keyframe_index is None or original_frame_id is None:
        return None
    return MappingRecord(video_id.upper(), keyframe_index, original_frame_id, str(source_path))


def _records_from_json(value: Any, source_path: Path) -> list[MappingRecord] | None:
    if isinstance(value, list):
        records = [_record_from_mapping_row(row, source_path) for row in value if isinstance(row, dict)]
        return [record for record in records if record is not None] or None
    if not isinstance(value, dict):
        return None
    direct_record = _record_from_mapping_row(value, source_path)
    if direct_record is not None:
        return [direct_record]
    for nested_key in ("keyframes", "mappings", "data", "items", "frames"):
        nested = value.get(nested_key)
        nested_records = _records_from_json(nested, source_path)
        if nested_records is not None:
            return nested_records
    video_id = infer_video_id(source_path.stem)
    if video_id is not None and all(_as_int(key) is not None for key in value):
        records: list[MappingRecord] = []
        for index_value, frame_value in value.items():
            keyframe_index = _as_int(index_value)
            original_frame_id = _as_int(frame_value)
            if keyframe_index is None or original_frame_id is None:
                return None
            records.append(MappingRecord(video_id, keyframe_index, original_frame_id, str(source_path)))
        return records
    return None


def _load_csv_mapping(path: Path) -> tuple[list[MappingRecord], list[str]]:
    records: list[MappingRecord] = []
    malformed: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return records, [f"{path}: no header"]
        for line_number, row in enumerate(reader, start=2):
            record = _record_from_mapping_row(row, path)
            if record is None:
                malformed.append(f"{path}:{line_number}")
            else:
                records.append(record)
    return records, malformed


def load_mapping_records(root: Path) -> MappingLoadResult:
    if not root.exists():
        return MappingLoadResult((), (), ())
    records: list[MappingRecord] = []
    unsupported: list[str] = []
    malformed: list[str] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            parsed, errors = _load_csv_mapping(path)
            records.extend(parsed)
            malformed.extend(errors)
            continue
        if suffix in {".json", ".jsonl"}:
            try:
                if suffix == ".jsonl":
                    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                else:
                    values = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                malformed.append(f"{path}: {error}")
                continue
            parsed = _records_from_json(values, path)
            if parsed is None:
                unsupported.append(str(path))
            else:
                records.extend(parsed)
            continue
        unsupported.append(str(path))
    return MappingLoadResult(tuple(records), tuple(unsupported), tuple(malformed))


def mapping_lookup(records: Iterable[MappingRecord]) -> dict[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    for record in records:
        lookup[(record.video_id, record.keyframe_index)] = record.original_frame_id
    return lookup


def validate_mapping(
    keyframes: Iterable[KeyframeRecord],
    videos: Iterable[VideoRecord],
    mappings: Iterable[MappingRecord],
    timestamp_tolerance_seconds: float = 1.0,
) -> MappingValidationReport:
    keyframe_list = list(keyframes)
    video_map = {video.video_id: video for video in videos}
    mapping_list = list(mappings)
    report = MappingValidationReport()
    keyframe_keys = {
        (keyframe.video_id, keyframe.keyframe_index)
        for keyframe in keyframe_list
        if keyframe.keyframe_index is not None
    }
    mapping_keys = [(mapping.video_id, mapping.keyframe_index) for mapping in mapping_list]
    mapping_counts = Counter(mapping_keys)
    invalid_mapping_keys: set[tuple[str, int]] = set()
    duplicate_mapping_keys = [key for key, count in mapping_counts.items() if count > 1]
    if duplicate_mapping_keys:
        report.duplicate_keyframe_index_count = len(duplicate_mapping_keys)
        report.issues.append(
            ValidationIssue(
                "HIGH",
                "duplicate_mapping_keyframe_index",
                "Multiple mapping rows share a video/keyframe index",
                tuple(f"{video_id}:{index}" for video_id, index in duplicate_mapping_keys[:50]),
            )
        )
    for keyframe in keyframe_list:
        if keyframe.keyframe_index is not None and (keyframe.video_id, keyframe.keyframe_index) in mapping_counts:
            report.matched_count += 1
        else:
            report.missing_mapping_count += 1
    for mapping in mapping_list:
        mapping_key = (mapping.video_id, mapping.keyframe_index)
        if mapping_key not in keyframe_keys:
            report.missing_image_count += 1
        video = video_map.get(mapping.video_id)
        if video is None:
            report.unknown_video_count += 1
            continue
        if mapping.original_frame_id < 0 or (
            video.frame_count is not None and mapping.original_frame_id >= video.frame_count
        ):
            report.invalid_frame_count += 1
            invalid_mapping_keys.add(mapping_key)
    if report.unknown_video_count:
        report.issues.append(
            ValidationIssue("HIGH", "mapping_unknown_video", "Mappings reference missing videos")
        )
    if report.missing_mapping_count:
        report.issues.append(
            ValidationIssue("HIGH", "keyframe_mapping_missing", "Keyframes without mapping were found")
        )
    grouped = defaultdict(list)
    for mapping in mapping_list:
        grouped[mapping.video_id].append(mapping)
    duplicate_frame_mappings: list[str] = []
    non_monotonic: list[str] = []
    for video_id, values in grouped.items():
        ordered = sorted(values, key=lambda value: value.keyframe_index)
        frame_counts = Counter(value.original_frame_id for value in ordered)
        duplicate_frame_mappings.extend(
            f"{video_id}:{frame_id}" for frame_id, count in frame_counts.items() if count > 1
        )
        for previous, current in zip(ordered, ordered[1:]):
            if current.original_frame_id < previous.original_frame_id:
                non_monotonic.append(video_id)
                break
        video = video_map.get(video_id)
        if video and video.fps and video.duration_sec is not None:
            for mapping in ordered:
                mapping_key = (mapping.video_id, mapping.keyframe_index)
                if (
                    mapping_key not in invalid_mapping_keys
                    and mapping.original_frame_id / video.fps
                    > video.duration_sec + timestamp_tolerance_seconds
                ):
                    report.invalid_frame_count += 1
                    invalid_mapping_keys.add(mapping_key)
                    break
    if report.invalid_frame_count and not any(
        issue.code == "mapping_frame_out_of_bounds" for issue in report.issues
    ):
        report.issues.append(
            ValidationIssue("HIGH", "mapping_frame_out_of_bounds", "Mappings contain invalid frame IDs")
        )
    report.duplicate_frame_mapping_count = len(duplicate_frame_mappings)
    report.non_monotonic_count = len(non_monotonic)
    if duplicate_frame_mappings:
        report.issues.append(
            ValidationIssue(
                "MEDIUM",
                "duplicate_original_frame_mapping",
                "Multiple keyframes map to the same original frame",
                tuple(duplicate_frame_mappings[:50]),
            )
        )
    if non_monotonic:
        report.issues.append(
            ValidationIssue(
                "HIGH",
                "non_monotonic_mapping",
                "Original frame IDs decrease as keyframe indices increase",
                tuple(non_monotonic[:50]),
            )
        )
    return report
