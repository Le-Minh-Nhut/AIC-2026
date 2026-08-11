#!/usr/bin/env python3
"""Generate deterministic AIC 2026 data audit reports from downloaded data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config
from data.analyzer import AnalysisOptions, analyze_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--report", type=Path, default=Path("docs/DATA_ANALYSIS.md"))
    parser.add_argument("--sample-decode-videos", type=int, default=20)
    parser.add_argument("--sample-images", type=int, default=100)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on configured BLOCKER/HIGH issues")
    args = parser.parse_args()
    if args.sample_decode_videos < 0 or args.sample_images < 0:
        raise SystemExit("Sample sizes must not be negative")
    config = load_yaml_config()
    data_root = configured_data_root(config, args.data_root)
    analysis_config = config["analysis"]
    report = analyze_data(
        AnalysisOptions(
            data_root=data_root,
            report_path=args.report,
            sample_decode_videos=args.sample_decode_videos,
            sample_images=args.sample_images,
            random_seed=int(analysis_config["random_seed"]),
            timestamp_tolerance_seconds=float(analysis_config["mapping_timestamp_tolerance_seconds"]),
            duration_tolerance_seconds=float(analysis_config["duration_consistency_tolerance_seconds"]),
        )
    )
    print(f"JSON report: {data_root / 'reports' / 'data_analysis.json'}")
    print(f"Markdown report: {args.report}")
    configured_severities = set(analysis_config["strict_fail_severities"])
    failed_issues = [issue for issue in report["issues"] if issue["severity"] in configured_severities]
    if args.strict and failed_issues:
        print(f"Strict mode failed: {len(failed_issues)} configured severity issue(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
