"""Versioned ground-truth and evaluation-report JSON I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from domain.competition import CompetitionContractError, FrameWindow, GroundTruthQuery, TaskType


GROUND_TRUTH_SCHEMA_VERSION = "1.0"


class EvaluationFormatError(ValueError):
    pass


def load_ground_truth(path: Path) -> tuple[GroundTruthQuery, ...]:
    value = _read_json_object(path, "ground truth")
    if value.get("schema_version") != GROUND_TRUTH_SCHEMA_VERSION:
        raise EvaluationFormatError("Unsupported ground-truth schema_version")
    raw_queries = value.get("queries")
    if not isinstance(raw_queries, list):
        raise EvaluationFormatError("Ground-truth queries must be a list")
    try:
        queries = tuple(_ground_truth_from_dict(item) for item in raw_queries)
    except CompetitionContractError as error:
        raise EvaluationFormatError(str(error)) from error
    if len({query.query_id for query in queries}) != len(queries):
        raise EvaluationFormatError("Ground-truth query_id values must be unique")
    return queries


def write_evaluation_report(path: Path, report: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _ground_truth_from_dict(value: object) -> GroundTruthQuery:
    if not isinstance(value, dict):
        raise EvaluationFormatError("Ground-truth query must be an object")
    try:
        task = TaskType(str(value.get("task")))
    except ValueError as error:
        raise EvaluationFormatError("Ground-truth task must be kis, qna, or trake") from error
    raw_windows = value.get("frame_windows")
    if not isinstance(raw_windows, list):
        raise EvaluationFormatError("Ground-truth frame_windows must be a list")
    windows = tuple(_window_from_value(item) for item in raw_windows)
    answer = value.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise EvaluationFormatError("Ground-truth answer must be a string")
    return GroundTruthQuery(
        query_id=str(value.get("query_id") or "").strip(),
        task=task,
        video_id=str(value.get("video_id") or "").strip(),
        frame_windows=windows,
        answer=answer,
    )


def _window_from_value(value: object) -> FrameWindow:
    if not isinstance(value, list) or len(value) != 2:
        raise EvaluationFormatError("Each frame window must be [start_frame_id, end_frame_id]")
    start, end = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)):
        raise EvaluationFormatError("Frame window bounds must be integers")
    return FrameWindow(start, end)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label.capitalize()} file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationFormatError(f"Invalid {label} JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationFormatError(f"{label.capitalize()} root must be an object")
    return value
