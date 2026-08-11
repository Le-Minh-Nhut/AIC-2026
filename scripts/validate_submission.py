#!/usr/bin/env python3
"""Validate a versioned submission against the video manifest and optional ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config, repository_root
from data.video_repository import VideoManifestValidationError, load_video_records_from_parquet
from evaluation.io import EvaluationFormatError, load_ground_truth
from submission.validation import SubmissionValidationError, SubmissionValidator
from submission.writer import SubmissionFormatError, load_submission


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--video-manifest", type=Path, help="Verified videos_manifest.parquet")
    parser.add_argument("--ground-truth", type=Path, help="Optional local GT for task/event-count checks")
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--config", type=Path, help="Hardening config path")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    hardening_config = load_yaml_config(args.config or root / "configs" / "hardening.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    video_manifest = args.video_manifest or data_root / "manifests" / "videos_manifest.parquet"
    try:
        records = load_video_records_from_parquet(video_manifest)
        inventory = {
            record.video_id: record.frame_count
            for record in records
            if record.is_readable and record.frame_count is not None
        }
        if len(inventory) != len(records):
            raise SubmissionValidationError(
                "Video manifest has unreadable videos or missing frame_count; strict submission validation cannot continue"
            )
        ground_truths = load_ground_truth(args.ground_truth) if args.ground_truth else ()
        summary = SubmissionValidator(
            video_frame_counts={video_id: int(frame_count) for video_id, frame_count in inventory.items()},
            ground_truths={query.query_id: query for query in ground_truths},
            max_results_per_query=int(hardening_config["submission"]["max_results_per_query"]),
        ).validate(load_submission(args.submission))
    except (
        EvaluationFormatError,
        FileNotFoundError,
        SubmissionFormatError,
        SubmissionValidationError,
        VideoManifestValidationError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Valid submission: queries={summary.query_count} candidates={summary.candidate_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
