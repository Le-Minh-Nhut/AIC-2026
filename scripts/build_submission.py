#!/usr/bin/env python3
"""Convert a KIS, Q&A, or TRAKE debug JSON into a diverse versioned submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_yaml_config, repository_root
from domain.competition import TaskType
from submission.ranker import FrameDiversityConfig, SequenceDiversityConfig
from submission.writer import SubmissionFormatError, load_submission, submission_from_debug, write_submission


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(task.value for task in TaskType))
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--debug-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--append", action="store_true", help="Add a new query to an existing submission")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing submission")
    parser.add_argument("--config", type=Path, help="Hardening config path")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.append and args.overwrite:
        raise SystemExit("Choose at most one of --append and --overwrite")
    root = repository_root()
    config = load_yaml_config(args.config or root / "configs" / "hardening.yaml")
    try:
        debug_payload = _load_json(args.debug_json)
        task = TaskType(args.task)
        diversity = config["submission"]["diversity"]
        query = submission_from_debug(
            task=task,
            query_id=args.query_id,
            debug_payload=debug_payload,
            frame_config=FrameDiversityConfig(
                max_results=int(config["submission"]["max_results_per_query"]),
                max_per_video=int(diversity["max_per_video"]),
                temporal_window_sec=float(diversity["temporal_window_sec"]),
            ),
            sequence_config=SequenceDiversityConfig(
                max_results=int(config["submission"]["max_results_per_query"]),
                max_per_video=int(diversity["max_sequences_per_video"]),
                near_duplicate_frame_window=int(diversity["trake_near_duplicate_frame_window"]),
            ),
        )
        existing = load_submission(args.output) if args.append and args.output.exists() else ()
        if args.output.exists() and not args.append and not args.overwrite:
            raise SubmissionFormatError("Output exists; use --append or --overwrite explicitly")
        if any(item.query_id == query.query_id for item in existing):
            raise SubmissionFormatError(f"Query ID already exists in submission: {query.query_id}")
        write_submission(args.output, (*existing, query))
    except (FileNotFoundError, KeyError, SubmissionFormatError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Submission JSON: {args.output}")
    return 0


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Debug JSON does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubmissionFormatError(f"Invalid debug JSON: {path}") from error
    if not isinstance(value, dict):
        raise SubmissionFormatError("Debug JSON root must be an object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
