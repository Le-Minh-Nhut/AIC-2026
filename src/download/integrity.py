"""Archive integrity checks used before extraction."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from domain.models import ArchiveInspection


ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def has_zip_signature(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in ZIP_SIGNATURES
    except OSError:
        return False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_zip_archive(path: Path) -> ArchiveInspection:
    if not has_zip_signature(path):
        return ArchiveInspection(False, False, None, None, None, "ZIP signature is missing")
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            infos = archive.infolist()
            compressed_size = sum(info.compress_size for info in infos)
            uncompressed_size = sum(info.file_size for info in infos)
        if bad_member is not None:
            return ArchiveInspection(
                True,
                False,
                len(infos),
                compressed_size,
                uncompressed_size,
                f"CRC validation failed for {bad_member}",
            )
        return ArchiveInspection(True, True, len(infos), compressed_size, uncompressed_size)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        return ArchiveInspection(True, False, None, None, None, str(error))
