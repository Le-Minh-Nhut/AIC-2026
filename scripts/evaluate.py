#!/usr/bin/env python3
"""Evaluate versioned KIS, Q&A, or TRAKE submissions with official BTC metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_yaml_config, repository_root
from domain.competition import TaskType
from evaluation.evaluator import CompetitionEvaluator, EvaluationValidationError
from evaluation.io import EvaluationFormatError, load_ground_truth, write_evaluation_report
from experiments.logging import ExperimentLogger, ExperimentRecord, dataset_snapshot_descriptor, repository_git_commit
from submission.writer import SubmissionFormatError, load_submission


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path, help="Versioned local ground-truth JSON")
    parser.add_argument("--submission", required=True, type=Path, help="Versioned submission JSON")
    parser.add_argument("--task", choices=tuple(task.value for task in TaskType), help="Evaluate one task")
    parser.add_argument("--output", type=Path, help="Evaluation report JSON")
    parser.add_argument("--experiment-log", type=Path, help="Append experiment JSONL at this path")
    parser.add_argument("--experiment-id", help="Unique ID required when --experiment-log is used")
    parser.add_argument("--dataset-snapshot", type=Path, help="Data report/snapshot file recorded with experiment")
    parser.add_argument("--config", action="append", type=Path, default=[], help="Config file recorded by SHA256")
    parser.add_argument("--runtime-metadata", type=Path, help="JSON runtime metadata from a benchmark run")
    parser.add_argument("--pipeline-latency-ms", type=float, help="Optional full-pipeline latency recorded in experiment")
    parser.add_argument("--notes", default="", help="Experiment notes")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.pipeline_latency_ms is not None and args.pipeline_latency_ms < 0:
        raise SystemExit("--pipeline-latency-ms must be non-negative")
    root = repository_root()
    try:
        ground_truths = load_ground_truth(args.ground_truth)
        submissions = load_submission(args.submission)
        task = TaskType(args.task) if args.task else _infer_task(ground_truths)
        selected_ground_truths = tuple(query for query in ground_truths if query.task is task)
        selected_submissions = {
            query.query_id: query for query in submissions if query.task is task
        }
        started = time.perf_counter()
        result = CompetitionEvaluator().evaluate(selected_ground_truths, selected_submissions)
        evaluation_latency_ms = (time.perf_counter() - started) * 1000
        report = {
            "schema_version": "1.0",
            "ground_truth": str(args.ground_truth),
            "submission": str(args.submission),
            "evaluation_latency_ms": evaluation_latency_ms,
            **result.as_dict(),
        }
        output = args.output or root / "outputs" / "evaluations" / f"{task.value}.json"
        write_evaluation_report(output, report)
        if args.experiment_log is not None:
            _log_experiment(args, root, result.as_dict()["aggregate"], evaluation_latency_ms, task)
    except (
        EvaluationFormatError,
        EvaluationValidationError,
        SubmissionFormatError,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    aggregate = result.aggregate_metrics.as_dict()
    print("  ".join(f"{name}={value:.6f}" for name, value in aggregate.items()))
    print(f"Evaluation JSON: {output}")
    return 0


def _infer_task(ground_truths: tuple[object, ...]) -> TaskType:
    tasks = {query.task for query in ground_truths}
    if len(tasks) != 1:
        raise EvaluationValidationError("--task is required when ground truth contains multiple tasks")
    return next(iter(tasks))


def _log_experiment(
    args: argparse.Namespace,
    root: Path,
    metrics: object,
    evaluation_latency_ms: float,
    task: TaskType,
) -> None:
    if not args.experiment_id or args.dataset_snapshot is None:
        raise EvaluationValidationError("--experiment-log requires --experiment-id and --dataset-snapshot")
    if not isinstance(metrics, dict):
        raise EvaluationValidationError("Evaluation aggregate metrics are invalid")
    runtime_payload = _load_json_object(args.runtime_metadata) if args.runtime_metadata else {}
    runtime = runtime_payload.get("metadata") if isinstance(runtime_payload.get("metadata"), dict) else runtime_payload
    config = {
        str(path): {"sha256": _file_digest(path), "content": load_yaml_config(path)}
        for path in args.config
    }
    latency = {"evaluation": evaluation_latency_ms}
    if args.pipeline_latency_ms is not None:
        latency["pipeline"] = args.pipeline_latency_ms
    ExperimentLogger(args.experiment_log).append(
        ExperimentRecord(
            experiment_id=args.experiment_id,
            task=task.value,
            dataset_snapshot=dataset_snapshot_descriptor(args.dataset_snapshot),
            encoder=_nested_text(runtime, "selected_encoder"),
            sources=tuple(sorted(_runtime_sources(runtime))),
            model_revisions=_model_revisions(runtime),
            config=config,
            metrics={str(name): float(value) for name, value in metrics.items()},
            latency_ms=latency,
            git_commit=repository_git_commit(root),
            notes=args.notes,
        )
    )


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationValidationError(f"Invalid runtime metadata JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvaluationValidationError("Runtime metadata must be a JSON object")
    return value


def _file_digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_text(runtime: dict[str, object], key: str) -> str | None:
    value = runtime.get(key)
    return value if isinstance(value, str) else None


def _runtime_sources(runtime: dict[str, object]) -> set[str]:
    fusion = runtime.get("multimodal_fusion") or runtime.get("fusion")
    if isinstance(fusion, dict) and isinstance(fusion.get("weights"), dict):
        return {str(source) for source in fusion["weights"]}
    encoder = _nested_text(runtime, "selected_encoder")
    return {encoder} if encoder is not None else set()


def _model_revisions(runtime: dict[str, object]) -> dict[str, str | None]:
    branches = runtime.get("branches")
    if not isinstance(branches, dict):
        return {}
    values: dict[str, str | None] = {}
    for source, branch in branches.items():
        if not isinstance(branch, dict) or not isinstance(branch.get("encoder"), dict):
            continue
        revision = branch["encoder"].get("revision")
        values[str(source)] = str(revision) if revision is not None else None
    top_encoder = runtime.get("encoder")
    selected_encoder = _nested_text(runtime, "selected_encoder")
    if isinstance(top_encoder, dict) and selected_encoder is not None:
        revision = top_encoder.get("revision")
        values.setdefault(selected_encoder, str(revision) if revision is not None else None)
    return values


if __name__ == "__main__":
    raise SystemExit(main())
