from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_kis
from domain.models import KeyframeRecord
from domain.models import VideoRecord
from encoders.btc_clip import BtcClipTextEncoder
from encoders.fgclip2 import FGCLIP2Encoder
from encoders.pecore import PECoreEncoder
from indexing.base import FeatureValidationReport
from indexing.exact_index import ExactCosineIndex
from indexing.feature_store import BtcClipFeatureStore
from retrieval.visual_retriever import VectorRetriever
from tasks.kis_service import KisCoarseRetrievalService, write_kis_debug


def keyframe(uid: str, video_id: str, index: int, frame_id: int) -> KeyframeRecord:
    return KeyframeRecord(
        keyframe_uid=uid,
        video_id=video_id,
        keyframe_index=index,
        keyframe_path=f"raw/keyframes/{video_id}/{index:06d}.jpg",
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        width=10,
        height=10,
        file_size_bytes=10,
        is_readable=True,
        has_mapping=True,
    )


class _QueryBackend:
    embedding_dimension = 2

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[0.0, 4.0]], dtype=np.float32), (len(texts), 1))


class _FGQueryBackend:
    embedding_dimension = 2
    preprocessing_config = {"fixture": True}

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.zeros((len(images), 2), dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class _PEQueryBackend:
    embedding_dimension = 2
    preprocessing_config = {"fixture": "pecore"}

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.zeros((len(images), 2), dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(texts), 1))


class _RefinementFGBackend:
    embedding_dimension = 2
    preprocessing_config = {"fixture": "refinement"}

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        values = np.array(
            [[max(1.0, 100.0 - abs(image.getpixel((0, 0))[0] - 14) * 10), 1.0] for image in images],
            dtype=np.float32,
        )
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class _RefinementDecoder:
    def inspect(self, video_path: Path):
        from refinement.video_decoder import DecodedVideoInfo

        return DecodedVideoInfo(fps=10.0, frame_count=31)

    def decode_frames(self, video_path: Path, frame_ids: tuple[int, ...]):
        from refinement.video_decoder import DecodedFrame

        return tuple(
            DecodedFrame(
                frame_id=frame_id,
                timestamp_sec=frame_id / 10.0,
                image=Image.new("RGB", (2, 2), color=(frame_id, 0, 0)),
            )
            for frame_id in frame_ids
        )


def video_record(video_id: str) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        video_path=f"raw/videos/{video_id}.mp4",
        group_id="L21",
        fps=10.0,
        frame_count=31,
        duration_sec=3.1,
        width=2,
        height=2,
        video_codec="fixture",
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
        has_audio=False,
        container="fixture",
        file_size_bytes=1,
        is_readable=True,
    )


class _FGStore:
    def __init__(self, records: tuple[KeyframeRecord, ...]) -> None:
        self._records = {record.keyframe_uid: record for record in records}
        self.ordered_uids = tuple(self._records)
        self.count = len(records)
        self.dimension = 2
        self.manifest_path = Path("fixture_fg_manifest.json")
        self.validation = FeatureValidationReport(
            count=self.count,
            dimension=self.dimension,
            dtypes=("float32",),
            nan_count=0,
            inf_count=0,
            zero_vector_count=0,
            min_norm=1.0,
            max_norm=1.0,
            vectors_are_l2_normalized=True,
            uses_mmap=True,
        )

    def metadata_for_id(self, item_id: str) -> KeyframeRecord:
        return self._records[item_id]


class _FGIndex:
    dimension = 2

    def __init__(self, item_ids: tuple[str, ...]) -> None:
        self._item_ids = item_ids

    def validate_feature_store(self, store: _FGStore) -> None:
        if store.ordered_uids != self._item_ids:
            raise ValueError("fixture mapping mismatch")

    def search(self, query: np.ndarray, top_k: int):
        from domain.models import SearchHit

        return tuple(SearchHit(item_id, 1.0 - rank * 0.1) for rank, item_id in enumerate(self._item_ids[:top_k]))


class KisBaselineIntegrationTests(unittest.TestCase):
    def _fixture_store(self, root: Path) -> tuple[BtcClipFeatureStore, tuple[KeyframeRecord, ...], Path]:
        feature_path = root / "btc_features.npy"
        np.save(
            feature_path,
            np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 3.0], [0.0, 1.0]], dtype=np.float32),
        )
        records = (
            keyframe("L21_V001:000000", "L21_V001", 0, 0),
            keyframe("L21_V001:000001", "L21_V001", 1, 10),
            keyframe("L21_V001:000002", "L21_V001", 2, 15),
            keyframe("L21_V002:000000", "L21_V002", 0, 10),
        )
        order_path = root / "btc_clip_feature_order.json"
        order_path.write_text(
            json.dumps(
                {
                    "feature_files": [feature_path.name],
                    "keyframe_uids": [record.keyframe_uid for record in records],
                    "mapping_verified": True,
                    "verification_method": "small fixture verified against explicit UID list",
                }
            ),
            encoding="utf-8",
        )
        return BtcClipFeatureStore([feature_path], records, order_path), records, order_path

    def test_kis_service_maps_frames_applies_nms_and_writes_debug_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, _, _ = self._fixture_store(root)
            service = KisCoarseRetrievalService(
                text_encoder=BtcClipTextEncoder(_QueryBackend()),
                retriever=VectorRetriever(ExactCosineIndex(store, batch_size=2), store, source="btc_clip"),
                temporal_nms_enabled=True,
                temporal_nms_window_sec=1.0,
                candidate_pool_multiplier=3,
            )

            result = service.search("find a blue event", top_k=3)
            debug_path = write_kis_debug(result, root / "debug.json", {"fixture": True})
            payload = json.loads(debug_path.read_text(encoding="utf-8"))

            self.assertEqual([candidate.original_frame_id for candidate in result.candidates], [10, 10])
            self.assertEqual([candidate.video_id for candidate in result.candidates], ["L21_V001", "L21_V002"])
            self.assertEqual(payload["candidates"][0]["frame_id"], 10)
            self.assertEqual(payload["candidates"][0]["keyframe_uid"], "L21_V001:000001")

    def test_cli_runs_end_to_end_with_fake_encoder_and_writes_debug_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, records, order_path = self._fixture_store(root)
            feature_path = root / "btc_features.npy"
            output_path = root / "cli_debug.json"
            arguments = [
                "run_kis.py",
                "--encoder",
                "btc_clip",
                "--coarse-only",
                "--query",
                "fixture query",
                "--top-k",
                "2",
                "--feature-file",
                str(feature_path),
                "--feature-order-manifest",
                str(order_path),
                "--keyframe-manifest",
                str(root / "keyframes_manifest.parquet"),
                "--debug-output",
                str(output_path),
            ]
            fake_encoder = BtcClipTextEncoder(_QueryBackend())
            with patch.object(run_kis, "load_keyframe_records_from_parquet", return_value=records), patch.object(
                run_kis, "load_btc_text_encoder", return_value=fake_encoder
            ), patch.object(sys, "argv", arguments):
                exit_code = run_kis.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["feature_store"]["dimension"], store.dimension)
            self.assertEqual(len(payload["candidates"]), 2)

    def test_cli_selects_fgclip2_without_reusing_btc_retrieval_logic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = (
                keyframe("L21_V001:000000", "L21_V001", 0, 0),
                keyframe("L21_V002:000000", "L21_V002", 0, 10),
            )
            store = _FGStore(records)
            output_path = root / "fg_debug.json"
            arguments = [
                "run_kis.py",
                "--encoder",
                "fgclip2_large",
                "--coarse-only",
                "--query",
                "fixture FG query",
                "--top-k",
                "2",
                "--fg-embedding-manifest",
                str(root / "manifest.json"),
                "--fg-index-dir",
                str(root / "index"),
                "--debug-output",
                str(output_path),
            ]
            encoder = FGCLIP2Encoder(_FGQueryBackend(), batch_size=2)
            with patch.object(run_kis, "load_keyframe_records_from_parquet", return_value=records), patch.object(
                run_kis, "ShardedNpyFeatureStore", return_value=store
            ), patch.object(run_kis, "load_faiss_flat_ip_index", return_value=_FGIndex(store.ordered_uids)), patch.object(
                run_kis, "load_fgclip2_text_encoder", return_value=encoder
            ), patch.object(sys, "argv", arguments):
                exit_code = run_kis.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["selected_encoder"], "fgclip2_large")
            self.assertEqual(payload["candidates"][0]["frame_id"], 0)

    def test_cli_selects_pecore_as_an_independent_faiss_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = (
                keyframe("L21_V001:000000", "L21_V001", 0, 0),
                keyframe("L21_V002:000000", "L21_V002", 0, 10),
            )
            store = _FGStore(records)
            output_path = root / "pe_debug.json"
            arguments = [
                "run_kis.py",
                "--encoder",
                "pecore_g14_448",
                "--coarse-only",
                "--query",
                "fixture PE query",
                "--top-k",
                "2",
                "--pe-embedding-manifest",
                str(root / "manifest.json"),
                "--pe-index-dir",
                str(root / "index"),
                "--debug-output",
                str(output_path),
            ]
            encoder = PECoreEncoder(_PEQueryBackend(), batch_size=2)
            with patch.object(run_kis, "load_keyframe_records_from_parquet", return_value=records), patch.object(
                run_kis, "ShardedNpyFeatureStore", return_value=store
            ), patch.object(run_kis, "load_faiss_flat_ip_index", return_value=_FGIndex(store.ordered_uids)), patch.object(
                run_kis, "load_pecore_text_encoder", return_value=encoder
            ), patch.object(sys, "argv", arguments):
                exit_code = run_kis.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["selected_encoder"], "pecore_g14_448")
            self.assertEqual(payload["candidates"][0]["source"], "pecore")
            self.assertEqual(payload["candidates"][0]["source_scores"][0]["rank"], 1)

    def test_cli_fuses_fgclip2_and_pecore_with_source_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = (
                keyframe("L21_V001:000000", "L21_V001", 0, 0),
                keyframe("L21_V002:000000", "L21_V002", 0, 10),
            )
            fg_store = _FGStore(records)
            pe_store = _FGStore(tuple(reversed(records)))
            output_path = root / "fusion_debug.json"
            arguments = [
                "run_kis.py",
                "--encoder",
                "fg_pe_fusion",
                "--coarse-only",
                "--query",
                "fixture fusion query",
                "--top-k",
                "2",
                "--fg-embedding-manifest",
                str(root / "fg_manifest.json"),
                "--fg-index-dir",
                str(root / "fg_index"),
                "--pe-embedding-manifest",
                str(root / "pe_manifest.json"),
                "--pe-index-dir",
                str(root / "pe_index"),
                "--debug-output",
                str(output_path),
            ]
            with patch.object(run_kis, "load_keyframe_records_from_parquet", return_value=records), patch.object(
                run_kis, "ShardedNpyFeatureStore", side_effect=(fg_store, pe_store)
            ), patch.object(
                run_kis,
                "load_faiss_flat_ip_index",
                side_effect=(_FGIndex(fg_store.ordered_uids), _FGIndex(pe_store.ordered_uids)),
            ), patch.object(
                run_kis, "load_fgclip2_text_encoder", return_value=FGCLIP2Encoder(_FGQueryBackend(), 2)
            ), patch.object(
                run_kis, "load_pecore_text_encoder", return_value=PECoreEncoder(_PEQueryBackend(), 2)
            ), patch.object(sys, "argv", arguments):
                exit_code = run_kis.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["selected_encoder"], "fg_pe_fusion")
            self.assertEqual(payload["metadata"]["fusion"]["method"], "rrf")
            self.assertEqual(set(payload["source_rankings"]), {"fgclip2", "pecore"})
            self.assertEqual(payload["candidates"][0]["source"], "rrf_fusion")
            self.assertEqual(len(payload["candidates"][0]["source_scores"]), 2)

    def test_cli_refines_fg_keyframe_to_original_video_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = (keyframe("L21_V001:000001", "L21_V001", 1, 10),)
            store = _FGStore(records)
            output_path = root / "refinement_debug.json"
            arguments = [
                "run_kis.py",
                "--encoder",
                "fgclip2_large",
                "--query",
                "fixture refinement query",
                "--top-k",
                "1",
                "--fg-embedding-manifest",
                str(root / "manifest.json"),
                "--fg-index-dir",
                str(root / "index"),
                "--video-manifest",
                str(root / "videos_manifest.parquet"),
                "--debug-output",
                str(output_path),
            ]
            encoder = FGCLIP2Encoder(_RefinementFGBackend(), batch_size=8)
            with patch.object(run_kis, "load_keyframe_records_from_parquet", return_value=records), patch.object(
                run_kis, "ShardedNpyFeatureStore", return_value=store
            ), patch.object(run_kis, "load_faiss_flat_ip_index", return_value=_FGIndex(store.ordered_uids)), patch.object(
                run_kis, "load_fgclip2_text_encoder", return_value=encoder
            ), patch.object(
                run_kis, "load_video_records_from_parquet", return_value=(video_record("L21_V001"),)
            ), patch.object(run_kis, "OpenCVVideoDecoder", return_value=_RefinementDecoder()), patch.object(
                sys, "argv", arguments
            ):
                exit_code = run_kis.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["metadata"]["refinement"]["enabled"])
            self.assertEqual(payload["coarse"]["candidates"][0]["frame_id"], 10)
            self.assertEqual(payload["candidates"][0]["frame_id"], 14)
            self.assertEqual(payload["candidates"][0]["source_keyframe_uid"], "L21_V001:000001")
