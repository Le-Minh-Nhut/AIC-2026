"""Official source-sheet parsing with a maintained audit fallback."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from domain.models import ArchiveCategory


DEFAULT_SOURCE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/export?format=csv&gid=0"
)


@dataclass(frozen=True, slots=True)
class SourceArchive:
    filename: str
    download_url: str
    category: ArchiveCategory


@dataclass(frozen=True, slots=True)
class SheetParseResult:
    archives: tuple[SourceArchive, ...]
    dropped_rows: tuple[str, ...]


def classify_archive(filename: str) -> ArchiveCategory:
    normalized = filename.strip().lower()
    if re.fullmatch(r"keyframes_.+\.zip", normalized):
        return ArchiveCategory.KEYFRAMES
    if re.fullmatch(r"videos_.+\.zip", normalized):
        return ArchiveCategory.VIDEOS
    if normalized.startswith("clip-features-") and normalized.endswith(".zip"):
        return ArchiveCategory.BTC_CLIP_FEATURES
    if normalized.startswith("map-keyframes-") and normalized.endswith(".zip"):
        return ArchiveCategory.MAP_KEYFRAMES
    if normalized.startswith("media-info-") and normalized.endswith(".zip"):
        return ArchiveCategory.MEDIA_INFO
    if normalized.startswith("objects-") and normalized.endswith(".zip"):
        return ArchiveCategory.OBJECTS
    return ArchiveCategory.UNKNOWN


def validate_download_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def parse_source_sheet_csv(
    csv_text: str,
    filename_column: str = "Filenames",
    url_column: str = "Download link",
) -> SheetParseResult:
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    if reader.fieldnames is None:
        raise ValueError("Source sheet CSV has no header row")
    columns = {name.strip().casefold(): name for name in reader.fieldnames if name}
    filename_key = columns.get(filename_column.casefold())
    url_key = columns.get(url_column.casefold())
    if filename_key is None or url_key is None:
        raise ValueError(
            f"Source sheet must contain '{filename_column}' and '{url_column}' columns; "
            f"found {reader.fieldnames!r}"
        )

    archives: list[SourceArchive] = []
    dropped_rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row_number, row in enumerate(reader, start=2):
        filename = (row.get(filename_key) or "").strip()
        download_url = (row.get(url_key) or "").strip()
        if not filename and not download_url:
            continue
        if not filename or not download_url:
            dropped_rows.append(f"row {row_number}: filename or URL is empty")
            continue
        if not validate_download_url(download_url):
            dropped_rows.append(f"row {row_number}: invalid URL for {filename!r}")
            continue
        identity = (filename, download_url)
        if identity in seen:
            dropped_rows.append(f"row {row_number}: duplicate archive {filename!r}")
            continue
        seen.add(identity)
        archives.append(SourceArchive(filename, download_url, classify_archive(filename)))
    return SheetParseResult(tuple(archives), tuple(dropped_rows))


class SourceSheetClient:
    def __init__(self, url: str = DEFAULT_SOURCE_SHEET_URL, timeout_seconds: float = 30.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def fetch_csv(self) -> str:
        request = Request(self.url, headers={"User-Agent": "aic2026-data-pipeline/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read()
        return payload.decode("utf-8-sig")

    def fetch(self) -> SheetParseResult:
        return parse_source_sheet_csv(self.fetch_csv())


def save_source_snapshot(csv_text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(csv_text, encoding="utf-8", newline="")


def fallback_archives() -> tuple[SourceArchive, ...]:
    names = [
        *(f"Keyframes_L{group}.zip" for group in range(21, 26)),
        *(f"Keyframes_L26_{part}.zip" for part in "abcde"),
        *(f"Keyframes_L{group}.zip" for group in range(27, 31)),
        *(f"Videos_L{group}_a.zip" for group in range(21, 26)),
        *(f"Videos_L26_{part}.zip" for part in "abcde"),
        *(f"Videos_L{group}_a.zip" for group in range(27, 31)),
        "clip-features-32-aic25-b1.zip",
        "map-keyframes-aic25-b1.zip",
        "media-info-aic25-b1.zip",
        "objects-aic25-b1.zip",
    ]
    base_url = "https://aic-data.ledo.io.vn"
    return tuple(
        SourceArchive(name, f"{base_url}/{name}", classify_archive(name)) for name in names
    )


def render_source_sheet_csv(archives: Iterable[SourceArchive]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["Filenames", "Download link"])
    writer.writeheader()
    for archive in archives:
        writer.writerow({"Filenames": archive.filename, "Download link": archive.download_url})
    return buffer.getvalue()
