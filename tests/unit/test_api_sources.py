from __future__ import annotations

import unittest

from api.sources import apply_source_selection, encoder_for_sources, normalize_sources


class ApiSourceSelectionTests(unittest.TestCase):
    def test_source_selection_maps_existing_encoder_modes(self) -> None:
        self.assertEqual(encoder_for_sources(("fgclip2",)), "fgclip2_large")
        self.assertEqual(encoder_for_sources(("pecore",)), "pecore_g14_448")
        self.assertEqual(encoder_for_sources(("pecore", "fgclip2", "ocr")), "fg_pe_fusion")


    def test_source_selection_enables_only_requested_auxiliary_branches(self) -> None:
        config = {
            "auxiliary_retrieval": {
                "ocr": {"enabled": False},
                "asr": {"enabled": True},
                "metadata": {"enabled": True},
            }
        }
        selected = apply_source_selection(config, ("fgclip2", "ocr"))

        self.assertTrue(selected["auxiliary_retrieval"]["ocr"]["enabled"])
        self.assertFalse(selected["auxiliary_retrieval"]["asr"]["enabled"])
        self.assertFalse(selected["auxiliary_retrieval"]["metadata"]["enabled"])
        self.assertTrue(config["auxiliary_retrieval"]["asr"]["enabled"])


    def test_text_only_or_unknown_sources_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "FG-CLIP2 or PE-Core"):
            normalize_sources(("ocr",))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            normalize_sources(("fgclip2", "unknown"))
