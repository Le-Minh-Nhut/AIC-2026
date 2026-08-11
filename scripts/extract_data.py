#!/usr/bin/env python3
"""Safely extract downloaded AIC data archives without deleting them by default."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config
from data.archive_manifest import ArchiveManifestStore
from data.source_sheet import classify_archive
from domain.models import ArchiveCategory
from download.extractor import extract_archive
from download.integrity import inspect_zip_archive


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--only", choices=("keyframes", "videos", "support", "all"), default="all")
    parser.add_argument("--archive", action="append", help="Extract one named archive; may be repeated")
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete an archive only after a successful explicit extraction",
    )
    return parser.parse_args()


def matches_selection(category: ArchiveCategory, selection: str) -> bool:
    return selection == "all" or (selection == "support" and category.is_support) or category.value == selection


def main() -> int:
    args = parse_arguments()
    config = load_yaml_config()
    data_root = configured_data_root(config, args.data_root)
    archive_root = data_root / "raw" / "archives"
    store = ArchiveManifestStore(data_root / "manifests" / "archives_manifest.jsonl")
    records_by_name = {record.filename: record for record in store.load().values()}
    archive_paths = sorted(archive_root.glob("*.zip")) if archive_root.exists() else []
    selected_paths = [path for path in archive_paths if args.archive is None or path.name in args.archive]
    failures = 0
    for archive_path in selected_paths:
        record = records_by_name.get(archive_path.name)
        category = record.category if record else classify_archive(archive_path.name)
        if not matches_selection(category, args.only):
            continue
        inspection = inspect_zip_archive(archive_path)
        if not inspection.zip_valid:
            print(f"ERROR: refusing invalid archive {archive_path}: {inspection.error}", file=sys.stderr)
            failures += 1
            continue
        try:
            result = extract_archive(
                archive_path,
                category,
                data_root,
                reject_symlinks=bool(config["extraction"]["reject_symlinks"]),
            )
            print(
                f"{archive_path.name}: extracted {len(result.extracted_files)} file(s), "
                f"skipped {len(result.skipped_files)} identical file(s)"
            )
            if record:
                record.extracted = True
                record.status = "extracted"
                store.upsert(record)
            if args.delete_archives:
                archive_path.unlink()
        except Exception as error:
            print(f"ERROR: extraction failed for {archive_path}: {error}", file=sys.stderr)
            failures += 1
    if args.archive and not selected_paths:
        print("ERROR: none of the requested archives were found", file=sys.stderr)
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
