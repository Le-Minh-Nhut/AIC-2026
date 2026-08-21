from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.keyframe_mapping import MappingLoadResult
from data.manifest_builder import (
    ManifestBuildError,
    ManifestBuildResult,
    build_manifests,
    collect_manifest_records,
)
from domain.models import MappingValidationReport, ValidationIssue


class ManifestPipelineTests(unittest.TestCase):
    def test_build_manifests_fails_before_writing_on_high_mapping_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            invalid_result = ManifestBuildResult(
                videos=(),
                keyframes=(),
                mapping_load=MappingLoadResult((), (), ()),
                mapping_validation=MappingValidationReport(
                    issues=[ValidationIssue("HIGH", "mapping_frame_out_of_bounds", "invalid frame")]
                ),
            )
            with patch("data.manifest_builder.collect_manifest_records", return_value=invalid_result):
                with self.assertRaises(ManifestBuildError):
                    build_manifests(data_root)

            self.assertFalse((data_root / "manifests" / "videos_manifest.parquet").exists())
            self.assertFalse((data_root / "manifests" / "keyframes_manifest.parquet").exists())

    def test_discovers_keyframes_and_json_mapping_without_guessing_video_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            keyframe_root = data_root / "raw" / "keyframes" / "L21_V001"
            mapping_root = data_root / "raw" / "map_keyframes"
            keyframe_root.mkdir(parents=True)
            mapping_root.mkdir(parents=True)
            for index in range(5):
                Image.new("RGB", (8, 6), color=(index, 0, 0)).save(keyframe_root / f"{index:06d}.jpg")
            (mapping_root / "L21_V001.json").write_text(
                json.dumps({str(index): index * 25 for index in range(5)}), encoding="utf-8"
            )

            result = collect_manifest_records(data_root)

            self.assertEqual(len(result.keyframes), 5)
            self.assertEqual(len(result.mapping_load.records), 5)
            self.assertTrue(all(record.has_mapping for record in result.keyframes))
