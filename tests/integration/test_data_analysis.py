from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.analyzer import AnalysisOptions, analyze_data
from data.source_sheet import fallback_archives, render_source_sheet_csv


class DataAnalysisTests(unittest.TestCase):
    def test_analysis_generates_json_markdown_and_tables_for_empty_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            snapshot = data_root / "manifests" / "source_sheet_snapshot.csv"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text(render_source_sheet_csv(fallback_archives()), encoding="utf-8")
            report_path = root / "docs" / "DATA_ANALYSIS.md"

            report = analyze_data(AnalysisOptions(data_root=data_root, report_path=report_path))

            self.assertEqual(report["dataset"]["archives"]["expected_count"], 32)
            self.assertTrue((data_root / "reports" / "data_analysis.json").exists())
            self.assertTrue((data_root / "reports" / "data_analysis_tables" / "mapping_summary.csv").exists())
            self.assertTrue(report_path.exists())
