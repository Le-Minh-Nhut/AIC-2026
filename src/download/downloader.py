"""Resumable archive downloader with integrity verification."""

from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data.archive_manifest import ArchiveManifestStore
from data.source_sheet import SourceArchive
from domain.models import ArchiveInspection, ArchiveRecord
from download.integrity import inspect_zip_archive, sha256_file


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DownloadSettings:
    timeout_seconds: float = 30.0
    retries: int = 3
    backoff_seconds: float = 1.0
    chunk_size_bytes: int = 1024 * 1024
    low_space_warning_fraction: float = 0.15
    reserved_free_space_bytes: int = 1024 * 1024 * 1024


class ProgressReporter:
    def __init__(self, filename: str, total_bytes: int | None) -> None:
        self.filename = filename
        self.total_bytes = total_bytes
        self.last_percent = -5

    def update(self, downloaded_bytes: int) -> None:
        if self.total_bytes is None:
            return
        percent = int(downloaded_bytes * 100 / max(self.total_bytes, 1))
        if percent >= self.last_percent + 5 or percent == 100:
            self.last_percent = percent
            print(
                f"{self.filename}: {downloaded_bytes}/{self.total_bytes} bytes ({percent}%)",
                file=sys.stderr,
            )


class ArchiveDownloader:
    def __init__(self, settings: DownloadSettings) -> None:
        self.settings = settings

    def content_length(self, url: str) -> int | None:
        request = Request(url, method="HEAD", headers={"User-Agent": "aic2026-data-pipeline/0.1"})
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                header = response.headers.get("Content-Length")
        except (HTTPError, URLError, TimeoutError, OSError):
            return None
        try:
            return int(header) if header is not None else None
        except ValueError:
            return None

    def download(self, source: SourceArchive, destination: Path) -> ArchiveRecord:
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing = self._verified_existing(source, destination)
        if existing is not None:
            return existing
        partial = destination.with_suffix(destination.suffix + ".part")
        error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                self._download_once(source, destination, partial)
                inspection = inspect_zip_archive(destination)
                if not inspection.zip_valid:
                    destination.unlink(missing_ok=True)
                    raise DownloadError(inspection.error or "Downloaded file is not a valid ZIP archive")
                return self._record_from_file(source, destination, inspection, "downloaded")
            except (DownloadError, HTTPError, URLError, TimeoutError, OSError) as caught:
                error = caught
                if attempt >= self.settings.retries:
                    break
                time.sleep(self.settings.backoff_seconds * (2**attempt))
        return ArchiveRecord(
            filename=source.filename,
            download_url=source.download_url,
            category=source.category,
            archive_path=str(destination),
            status="failed",
            error=str(error),
            updated_at=datetime.now(UTC).isoformat(),
        )

    def _verified_existing(self, source: SourceArchive, destination: Path) -> ArchiveRecord | None:
        if not destination.exists():
            return None
        inspection = inspect_zip_archive(destination)
        if not inspection.zip_valid:
            return None
        return self._record_from_file(source, destination, inspection, "skipped_existing")

    def _download_once(self, source: SourceArchive, destination: Path, partial: Path) -> None:
        resume_from = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "aic2026-data-pipeline/0.1"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        request = Request(source.download_url, headers=headers)
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range")
            can_resume = resume_from > 0 and status == 206 and bool(content_range)
            write_mode = "ab" if can_resume else "wb"
            if not can_resume:
                resume_from = 0
            content_length = response.headers.get("Content-Length")
            try:
                received_length = int(content_length) if content_length is not None else None
            except ValueError:
                received_length = None
            total_length = resume_from + received_length if received_length is not None else None
            progress = ProgressReporter(source.filename, total_length)
            downloaded = resume_from
            with partial.open(write_mode) as handle:
                while True:
                    chunk = response.read(self.settings.chunk_size_bytes)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    progress.update(downloaded)
            progress.update(downloaded)
        if not partial.exists() or partial.stat().st_size == 0:
            raise DownloadError("Server returned an empty download")
        partial.replace(destination)

    def _record_from_file(
        self,
        source: SourceArchive,
        path: Path,
        inspection: ArchiveInspection,
        status: str,
    ) -> ArchiveRecord:
        return ArchiveRecord(
            filename=source.filename,
            download_url=source.download_url,
            category=source.category,
            downloaded=True,
            archive_path=str(path),
            file_size_bytes=path.stat().st_size,
            sha256_local=sha256_file(path),
            zip_valid=inspection.zip_valid,
            entry_count=inspection.entry_count,
            compressed_size=inspection.compressed_size,
            estimated_uncompressed_size=inspection.uncompressed_size,
            status=status,
            error=inspection.error,
            updated_at=datetime.now(UTC).isoformat(),
        )


def ensure_disk_capacity(
    sources: Iterable[SourceArchive],
    downloader: ArchiveDownloader,
    archive_root: Path,
) -> list[str]:
    source_list = list(sources)
    known_sizes = [size for source in source_list if (size := downloader.content_length(source.download_url))]
    free_bytes = shutil.disk_usage(archive_root).free
    warnings: list[str] = []
    if known_sizes and sum(known_sizes) > free_bytes - downloader.settings.reserved_free_space_bytes:
        raise DownloadError(
            "Insufficient free disk space for known archive Content-Length values: "
            f"need {sum(known_sizes)} bytes with reserve, free {free_bytes} bytes"
        )
    if free_bytes / max(shutil.disk_usage(archive_root).total, 1) < downloader.settings.low_space_warning_fraction:
        warnings.append(f"Low free disk space: {free_bytes} bytes available")
    if len(known_sizes) == 0:
        warnings.append("No archive Content-Length values available; extracted size is not estimated")
    elif len(known_sizes) < len(source_list):
        warnings.append("Some archive Content-Length values are unavailable; total download is partial")
    return warnings


def download_archives(
    sources: Iterable[SourceArchive],
    archive_root: Path,
    manifest_store: ArchiveManifestStore,
    settings: DownloadSettings,
    workers: int = 1,
) -> list[ArchiveRecord]:
    selected = list(sources)
    archive_root.mkdir(parents=True, exist_ok=True)
    downloader = ArchiveDownloader(settings)
    for warning in ensure_disk_capacity(selected, downloader, archive_root):
        print(f"WARNING: {warning}", file=sys.stderr)
    manifest_lock = Lock()

    def work(source: SourceArchive) -> ArchiveRecord:
        record = downloader.download(source, archive_root / source.filename)
        with manifest_lock:
            manifest_store.upsert(record)
        return record

    if workers <= 1:
        return [work(source) for source in selected]
    records: list[ArchiveRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work, source) for source in selected]
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda record: record.filename)
