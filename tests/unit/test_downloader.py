from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.source_sheet import SourceArchive
from domain.models import ArchiveCategory
from download.downloader import ArchiveDownloader, DownloadSettings


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0
        self.status = 206
        self.headers = {"Content-Range": "bytes 2-5/6", "Content-Length": str(len(payload))}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, size: int) -> bytes:
        result = self.payload[self.offset : self.offset + size]
        self.offset += len(result)
        return result


class DownloaderTests(unittest.TestCase):
    def test_range_response_resumes_partial_file(self) -> None:
        source = SourceArchive("archive.zip", "https://example.test/archive.zip", ArchiveCategory.KEYFRAMES)
        downloader = ArchiveDownloader(DownloadSettings(chunk_size_bytes=2))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / source.filename
            partial = destination.with_suffix(".zip.part")
            partial.write_bytes(b"PK")
            with patch("download.downloader.urlopen", return_value=_FakeResponse(b"DATA")):
                downloader._download_once(source, destination, partial)

            self.assertEqual(destination.read_bytes(), b"PKDATA")
            self.assertFalse(partial.exists())
