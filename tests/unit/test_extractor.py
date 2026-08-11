from __future__ import annotations

import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from domain.models import ArchiveCategory
from download.extractor import UnsafeArchiveMemberError, extract_archive


class ExtractorTests(unittest.TestCase):
    def test_extract_normalizes_keyframes_into_logical_video_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "Keyframes_L21.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("nested/L21_V001/000001.jpg", b"image-bytes")

            result = extract_archive(archive, ArchiveCategory.KEYFRAMES, root)

            expected = root / "raw" / "keyframes" / "L21_V001" / "000001.jpg"
            self.assertTrue(expected.exists())
            self.assertEqual(len(result.extracted_files), 1)

    def test_zip_slip_member_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", b"blocked")

            with self.assertRaises(UnsafeArchiveMemberError):
                extract_archive(archive, ArchiveCategory.KEYFRAMES, root)

            self.assertFalse((root / "outside.txt").exists())

    def test_symlink_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "symlink.zip"
            info = zipfile.ZipInfo("L21_V001/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(info, "target")

            with self.assertRaises(UnsafeArchiveMemberError):
                extract_archive(archive, ArchiveCategory.KEYFRAMES, root)
