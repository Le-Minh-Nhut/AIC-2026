from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.manifest_builder import collect_manifest_records


class ManifestPipelineTests(unittest.TestCase):
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
