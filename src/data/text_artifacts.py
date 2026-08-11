"""Validated JSONL artifacts for OCR, ASR, and discovered metadata text."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TextArtifactValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OCRTextRecord:
    record_id: str
    keyframe_uid: str
    text: str
    bbox: tuple[float, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class ASRTranscriptRecord:
    segment_id: str
    video_id: str
    start_sec: float
    end_sec: float
    text: str

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2


@dataclass(frozen=True, slots=True)
class MetadataTextRecord:
    video_id: str
    fields: dict[str, str]


def load_ocr_records_jsonl(path: Path) -> tuple[OCRTextRecord, ...]:
    records: list[OCRTextRecord] = []
    seen_ids: set[str] = set()
    for line_number, value in _jsonl_values(path):
        keyframe_uid = _required_text(value, "keyframe_uid", path, line_number)
        text = _required_text(value, "text", path, line_number)
        bbox = _flatten_bbox(value.get("bbox"), path, line_number)
        confidence = _finite_float(value.get("confidence"), "confidence", path, line_number)
        if confidence < 0:
            raise TextArtifactValidationError(f"OCR confidence must be non-negative: {path}:{line_number}")
        record_id = str(value.get("record_id") or f"{keyframe_uid}:{line_number}").strip()
        if not record_id or record_id in seen_ids:
            raise TextArtifactValidationError(f"OCR record_id must be unique and non-empty: {path}:{line_number}")
        seen_ids.add(record_id)
        records.append(OCRTextRecord(record_id, keyframe_uid, text, bbox, confidence))
    return tuple(records)


def load_asr_records_jsonl(path: Path) -> tuple[ASRTranscriptRecord, ...]:
    records: list[ASRTranscriptRecord] = []
    seen_ids: set[str] = set()
    for line_number, value in _jsonl_values(path):
        video_id = _required_text(value, "video_id", path, line_number)
        start_sec = _finite_float(value.get("start_sec"), "start_sec", path, line_number)
        end_sec = _finite_float(value.get("end_sec"), "end_sec", path, line_number)
        if start_sec < 0 or end_sec < start_sec:
            raise TextArtifactValidationError(
                f"ASR segment requires 0 <= start_sec <= end_sec: {path}:{line_number}"
            )
        text = _required_text(value, "text", path, line_number)
        segment_id = str(value.get("segment_id") or f"{video_id}:{line_number}").strip()
        if not segment_id or segment_id in seen_ids:
            raise TextArtifactValidationError(f"ASR segment_id must be unique and non-empty: {path}:{line_number}")
        seen_ids.add(segment_id)
        records.append(ASRTranscriptRecord(segment_id, video_id, start_sec, end_sec, text))
    return tuple(records)


def load_metadata_records_jsonl(
    path: Path,
    allowed_fields: Sequence[str],
) -> tuple[MetadataTextRecord, ...]:
    configured_fields = tuple(sorted({field.strip() for field in allowed_fields if field.strip()}))
    if not configured_fields:
        raise TextArtifactValidationError(
            "Metadata retrieval needs explicitly configured fields discovered during data analysis"
        )
    records: list[MetadataTextRecord] = []
    seen_video_ids: set[str] = set()
    found_fields: set[str] = set()
    for line_number, value in _jsonl_values(path):
        video_id = _required_text(value, "video_id", path, line_number)
        if video_id in seen_video_ids:
            raise TextArtifactValidationError(f"Metadata video_id must be unique: {path}:{line_number}")
        seen_video_ids.add(video_id)
        raw_fields = value.get("fields")
        if not isinstance(raw_fields, Mapping):
            raise TextArtifactValidationError(f"Metadata fields must be an object: {path}:{line_number}")
        fields = {
            field: text
            for field in configured_fields
            if (text := _metadata_text(raw_fields.get(field))) is not None
        }
        found_fields.update(fields)
        if fields:
            records.append(MetadataTextRecord(video_id=video_id, fields=fields))
    missing_fields = sorted(set(configured_fields) - found_fields)
    if missing_fields:
        raise TextArtifactValidationError(
            "Configured metadata fields are not present in the artifact: " + ", ".join(missing_fields)
        )
    return tuple(records)


def _jsonl_values(path: Path) -> tuple[tuple[int, dict[str, Any]], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Text artifact does not exist: {path}")
    values: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise TextArtifactValidationError(f"Invalid JSONL at {path}:{line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise TextArtifactValidationError(f"JSONL row must be an object: {path}:{line_number}")
        values.append((line_number, value))
    return tuple(values)


def _required_text(value: Mapping[str, Any], field: str, path: Path, line_number: int) -> str:
    text = str(value.get(field) or "").strip()
    if not text:
        raise TextArtifactValidationError(f"Missing non-empty {field}: {path}:{line_number}")
    return text


def _finite_float(value: object, field: str, path: Path, line_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TextArtifactValidationError(f"Invalid {field}: {path}:{line_number}") from error
    if not math.isfinite(number):
        raise TextArtifactValidationError(f"Non-finite {field}: {path}:{line_number}")
    return number


def _flatten_bbox(value: object, path: Path, line_number: int) -> tuple[float, ...]:
    flattened: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for nested in item:
                visit(nested)
            return
        flattened.append(_finite_float(item, "bbox", path, line_number))

    visit(value)
    if len(flattened) < 4 or len(flattened) % 2:
        raise TextArtifactValidationError(f"OCR bbox needs an even number of at least four values: {path}:{line_number}")
    return tuple(flattened)


def _metadata_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return " ".join(parts) if parts else None
    return None
