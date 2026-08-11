"""ZIP extraction with zip-slip and symlink protection."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from domain.models import ArchiveCategory


class UnsafeArchiveMemberError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExtractedMember:
    archive_path: str
    archive_member: str
    destination_path: str
    category: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    extracted_files: tuple[ExtractedMember, ...]
    skipped_files: tuple[ExtractedMember, ...]


def _member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveMemberError(f"Unsafe archive member path: {name!r}")
    if re.match(r"^[A-Za-z]:", normalized):
        raise UnsafeArchiveMemberError(f"Unsafe drive-qualified member path: {name!r}")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def validate_archive_member(info: zipfile.ZipInfo, reject_symlinks: bool = True) -> PurePosixPath:
    path = _member_path(info.filename)
    if reject_symlinks and _is_symlink(info):
        raise UnsafeArchiveMemberError(f"Symlink archive member is rejected: {info.filename!r}")
    return path


def _find_video_id(member: PurePosixPath) -> str | None:
    for component in reversed(member.parts[:-1]):
        match = re.search(r"L\d+_V\d+", component, re.IGNORECASE)
        if match:
            return match.group(0).upper()
    match = re.search(r"L\d+_V\d+", member.stem, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _find_group_id(member: PurePosixPath) -> str | None:
    for component in member.parts:
        match = re.search(r"L\d+", component, re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return None


def normalized_destination(
    category: ArchiveCategory,
    member: PurePosixPath,
    raw_root: Path,
) -> Path:
    filename = member.name
    if category is ArchiveCategory.KEYFRAMES:
        video_id = _find_video_id(member)
        relative = Path(video_id or "unresolved") / filename
        return raw_root / "keyframes" / relative
    if category is ArchiveCategory.VIDEOS:
        group_id = _find_group_id(member)
        relative = Path(group_id or "unresolved") / filename
        return raw_root / "videos" / relative
    category_directory = category.value
    if category is ArchiveCategory.UNKNOWN:
        category_directory = "unknown"
    return raw_root / category_directory / Path(*member.parts)


def _same_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _append_extraction_records(manifest_path: Path, records: tuple[ExtractedMember, ...]) -> None:
    if not records:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True))
            handle.write("\n")


def extract_archive(
    archive_path: Path,
    category: ArchiveCategory,
    data_root: Path,
    reject_symlinks: bool = True,
) -> ExtractionResult:
    raw_root = data_root / "raw"
    extracted: list[ExtractedMember] = []
    skipped: list[ExtractedMember] = []
    with zipfile.ZipFile(archive_path) as archive:
        plans: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in archive.infolist():
            member = validate_archive_member(info, reject_symlinks=reject_symlinks)
            if info.is_dir():
                continue
            destination = normalized_destination(category, member, raw_root)
            destination.resolve().relative_to(raw_root.resolve())
            plans.append((info, destination))
        for info, destination in plans:
            record = ExtractedMember(
                archive_path=str(archive_path),
                archive_member=info.filename,
                destination_path=str(destination),
                category=category.value,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                with archive.open(info) as source:
                    temporary_path = destination.with_suffix(destination.suffix + ".compare")
                    try:
                        with temporary_path.open("wb") as temporary_handle:
                            shutil.copyfileobj(source, temporary_handle)
                        if _same_content(destination, temporary_path):
                            skipped.append(record)
                            continue
                        raise FileExistsError(
                            f"Extraction collision with different content: {destination}"
                        )
                    finally:
                        temporary_path.unlink(missing_ok=True)
            temporary_path = destination.with_suffix(destination.suffix + ".extracting")
            try:
                with archive.open(info) as source, temporary_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            extracted.append(record)
    records = tuple(extracted + skipped)
    _append_extraction_records(data_root / "manifests" / "extraction_manifest.jsonl", records)
    return ExtractionResult(tuple(extracted), tuple(skipped))
