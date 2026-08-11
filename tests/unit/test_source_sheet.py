from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.source_sheet import (
    classify_archive,
    fallback_archives,
    parse_source_sheet_csv,
    validate_download_url,
)
from domain.models import ArchiveCategory


class SourceSheetTests(unittest.TestCase):
    def test_parser_normalizes_headers_and_drops_invalid_rows(self) -> None:
        result = parse_source_sheet_csv(
            " Filenames , Download link \n"
            "Keyframes_L21.zip,https://example.com/Keyframes_L21.zip\n"
            "Videos_L21_a.zip,not-a-url\n"
            ",https://example.com/missing.zip\n"
        )

        self.assertEqual(len(result.archives), 1)
        self.assertEqual(result.archives[0].category, ArchiveCategory.KEYFRAMES)
        self.assertEqual(len(result.dropped_rows), 2)

    def test_fallback_snapshot_has_audited_archive_counts(self) -> None:
        archives = fallback_archives()

        self.assertEqual(len(archives), 32)
        self.assertEqual(sum(item.category is ArchiveCategory.KEYFRAMES for item in archives), 14)
        self.assertEqual(sum(item.category is ArchiveCategory.VIDEOS for item in archives), 14)
        self.assertEqual(sum(item.category.is_support for item in archives), 4)

    def test_url_and_category_validation(self) -> None:
        self.assertTrue(validate_download_url("https://aic-data.ledo.io.vn/data.zip"))
        self.assertFalse(validate_download_url("file:///tmp/data.zip"))
        self.assertEqual(classify_archive("objects-aic25-b1.zip"), ArchiveCategory.OBJECTS)
        self.assertEqual(classify_archive("unexpected.zip"), ArchiveCategory.UNKNOWN)
