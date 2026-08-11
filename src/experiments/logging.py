"""Atomic JSONL experiment logs with reproducible provenance fields."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from evaluation.final_score import FINAL_SCORE_CUTOFFS


class ExperimentLogError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    task: str
    dataset_snapshot: str
    encoder: str | None
    sources: tuple[str, ...]
    model_revisions: Mapping[str, str | None]
    config: Mapping[str, object]
    metrics: Mapping[str, float]
    latency_ms: Mapping[str, float]
    git_commit: str | None = None
    notes: str = ""
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id.strip() or not self.task.strip() or not self.dataset_snapshot.strip():
            raise ExperimentLogError("experiment_id, task, and dataset_snapshot must be non-empty")
        required_metrics = {f"R@{cutoff}" for cutoff in FINAL_SCORE_CUTOFFS} | {"final_score"}
        missing = sorted(required_metrics - set(self.metrics))
        if missing:
            raise ExperimentLogError("Experiment metrics are missing: " + ", ".join(missing))
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in self.metrics.values()
        ):
            raise ExperimentLogError("Experiment metrics must be finite values in [0, 1]")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0
            for value in self.latency_ms.values()
        ):
            raise ExperimentLogError("Experiment latency values must be finite and non-negative")

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["sources"] = list(self.sources)
        value["model_revisions"] = dict(sorted(self.model_revisions.items()))
        value["config"] = dict(self.config)
        value["metrics"] = dict(self.metrics)
        value["latency_ms"] = dict(self.latency_ms)
        value["created_at"] = self.created_at or datetime.now(UTC).isoformat()
        return value


class ExperimentLogger:
    """Appends one immutable experiment record per ID without implicit overwrites."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: ExperimentRecord) -> Path:
        existing = self._load_existing()
        if any(value.get("experiment_id") == record.experiment_id for value in existing):
            raise ExperimentLogError(f"Experiment ID already exists: {record.experiment_id}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existing.append(record.as_dict())
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in existing),
            encoding="utf-8",
        )
        os.replace(temporary, self._path)
        return self._path

    def _load_existing(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        if not self._path.is_file():
            raise ExperimentLogError(f"Experiment log is not a file: {self._path}")
        values: list[dict[str, object]] = []
        for line_number, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExperimentLogError(f"Invalid experiment JSONL at {self._path}:{line_number}") from error
            if not isinstance(value, dict) or not isinstance(value.get("experiment_id"), str):
                raise ExperimentLogError(f"Invalid experiment record at {self._path}:{line_number}")
            values.append(value)
        return values


def dataset_snapshot_descriptor(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset snapshot/report does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path.name}:sha256:{digest}"


def repository_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None
