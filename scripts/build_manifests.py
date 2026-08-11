#!/usr/bin/env python3
"""Build video and keyframe Parquet manifests from extracted AIC data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config
from data.manifest_builder import build_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    args = parser.parse_args()
    config = load_yaml_config()
    data_root = configured_data_root(config, args.data_root)
    result = build_manifests(
        data_root,
        timestamp_tolerance_seconds=float(config["analysis"]["mapping_timestamp_tolerance_seconds"]),
    )
    print(f"videos_manifest.parquet: {len(result.videos)} row(s)")
    print(f"keyframes_manifest.parquet: {len(result.keyframes)} row(s)")
    print(f"mapping records: {len(result.mapping_load.records)}")
    print(f"mapping issues: {len(result.mapping_validation.issues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
