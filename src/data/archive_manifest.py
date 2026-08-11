"""Atomic JSONL archive-manifest storage."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from domain.models import ArchiveRecord


class ArchiveManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[tuple[str, str], ArchiveRecord]:
        if not self.path.exists():
            return {}
        records: dict[tuple[str, str], ArchiveRecord] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    record = ArchiveRecord.from_dict(value)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid archive manifest at line {line_number}: {error}") from error
                records[(record.filename, record.download_url)] = record
        return records

    def upsert(self, record: ArchiveRecord) -> None:
        records = self.load()
        record.updated_at = datetime.now(UTC).isoformat()
        records[(record.filename, record.download_url)] = record
        self.write(records.values())

    def write(self, records: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        ordered = sorted(
            list(records), key=lambda record: (record.category.value, record.filename, record.download_url)
        )
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in ordered:
                handle.write(json.dumps(record.as_dict(), sort_keys=True, ensure_ascii=False))
                handle.write("\n")
        os.replace(temporary_path, self.path)
