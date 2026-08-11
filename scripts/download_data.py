#!/usr/bin/env python3
"""Fetch the official source sheet and download verified AIC data archives."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config
from data.archive_manifest import ArchiveManifestStore
from data.source_sheet import (
    DEFAULT_SOURCE_SHEET_URL,
    SourceArchive,
    SourceSheetClient,
    fallback_archives,
    parse_source_sheet_csv,
    render_source_sheet_csv,
    save_source_snapshot,
)
from domain.models import ArchiveCategory
from download.downloader import DownloadSettings, download_archives
from download.extractor import extract_archive


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List archives parsed from the source sheet")
    parser.add_argument("--dry-run", action="store_true", help="Parse and display archives without downloading")
    parser.add_argument(
        "--only",
        choices=("keyframes", "videos", "support", "all"),
        default="all",
        help="Archive category selection",
    )
    parser.add_argument("--workers", type=int, help="Number of concurrent archive downloads")
    parser.add_argument("--extract", action="store_true", help="Safely extract verified archives after download")
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--source-url", help="Override official source-sheet CSV URL")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, help="Number of retries after the first attempt")
    return parser.parse_args()


def select_archives(archives: tuple[SourceArchive, ...], selection: str) -> tuple[SourceArchive, ...]:
    if selection == "all":
        return archives
    if selection == "support":
        return tuple(archive for archive in archives if archive.category.is_support)
    category = ArchiveCategory(selection)
    return tuple(archive for archive in archives if archive.category is category)


def fetch_sources(
    source_url: str,
    snapshot_path: Path,
    timeout_seconds: float,
) -> tuple[tuple[SourceArchive, ...], str | None]:
    try:
        client = SourceSheetClient(source_url, timeout_seconds=timeout_seconds)
        csv_text = client.fetch_csv()
        result = parse_source_sheet_csv(csv_text)
        save_source_snapshot(csv_text, snapshot_path)
        warning = "; ".join(result.dropped_rows) if result.dropped_rows else None
        return result.archives, warning
    except Exception as error:
        if snapshot_path.exists():
            try:
                result = parse_source_sheet_csv(snapshot_path.read_text(encoding="utf-8"))
                return result.archives, f"Source fetch failed ({error}); using existing snapshot"
            except Exception as snapshot_error:
                error = RuntimeError(f"{error}; existing snapshot is invalid: {snapshot_error}")
        archives = fallback_archives()
        save_source_snapshot(render_source_sheet_csv(archives), snapshot_path)
        return archives, f"Source fetch failed ({error}); using audited fallback snapshot"


def display_archives(archives: tuple[SourceArchive, ...]) -> None:
    for archive in archives:
        print(f"{archive.category.value:18} {archive.filename:38} {archive.download_url}")
    print(f"Total: {len(archives)} archive(s)")


def main() -> int:
    args = parse_arguments()
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries is not None and args.retries < 0:
        raise SystemExit("--retries must be zero or greater")
    config = load_yaml_config()
    data_root = configured_data_root(config, args.data_root)
    download_config = config["download"]
    timeout_seconds = args.timeout or float(download_config["timeout_seconds"])
    source_url = args.source_url or os.environ.get("AIC2026_SOURCE_SHEET_URL") or config["source_sheet"].get(
        "csv_url", DEFAULT_SOURCE_SHEET_URL
    )
    snapshot_path = data_root / "manifests" / "source_sheet_snapshot.csv"
    archives, warning = fetch_sources(source_url, snapshot_path, timeout_seconds)
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)
    selected = select_archives(archives, args.only)
    if args.list or args.dry_run:
        display_archives(selected)
    if args.list or args.dry_run:
        return 0
    if not selected:
        print("No archives match the selected category.", file=sys.stderr)
        return 1
    settings = DownloadSettings(
        timeout_seconds=timeout_seconds,
        retries=args.retries if args.retries is not None else int(download_config["retries"]),
        backoff_seconds=float(download_config["backoff_seconds"]),
        chunk_size_bytes=int(download_config["chunk_size_bytes"]),
        low_space_warning_fraction=float(download_config["low_space_warning_fraction"]),
        reserved_free_space_bytes=int(download_config["reserved_free_space_bytes"]),
    )
    manifest_store = ArchiveManifestStore(data_root / "manifests" / "archives_manifest.jsonl")
    records = download_archives(
        selected,
        data_root / "raw" / "archives",
        manifest_store,
        settings,
        workers=args.workers or int(download_config["workers"]),
    )
    failures = [record for record in records if not record.downloaded]
    for record in records:
        print(f"{record.filename}: {record.status}")
    if args.extract:
        for record in records:
            if not record.downloaded or not record.archive_path:
                continue
            result = extract_archive(
                Path(record.archive_path),
                record.category,
                data_root,
                reject_symlinks=bool(config["extraction"]["reject_symlinks"]),
            )
            record.extracted = True
            record.status = "extracted"
            manifest_store.upsert(record)
            print(
                f"{record.filename}: extracted {len(result.extracted_files)} file(s), "
                f"skipped {len(result.skipped_files)} identical file(s)"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
