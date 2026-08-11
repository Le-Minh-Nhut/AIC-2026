#!/usr/bin/env python3
"""Run multi-frame Q&A with reusable KIS retrieval, M5 refinement, and local Qwen3-VL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_kis
from config import configured_data_root, load_yaml_config, repository_root
from data.video_repository import VideoManifestValidationError, load_video_records_from_parquet
from encoders.base import EncoderUnavailableError
from indexing.faiss_index import FaissIndexValidationError, FaissUnavailableError
from indexing.feature_store import FeatureMappingVerificationError, FeatureStoreValidationError
from hardening.reproducibility import configure_determinism
from qna.answerer import Qwen3VLAnswerer, VLMAnswererError, VLMUnavailableError
from qna.frame_selector import CandidateClipSelector, ClipSelectionError, ClipSelectorConfig
from refinement.frame_sampler import FrameSampler
from refinement.video_decoder import OpenCVVideoDecoder, VideoDecodingError
from tasks.kis_service import KisDenseRefinementService
from tasks.qna_service import (
    QnAQuery,
    QnaService,
    QnaServiceConfig,
    default_qna_debug_path,
    write_qna_debug,
)
from qna.answer_normalizer import AnswerNormalizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-description", required=True, help="Visual event used for retrieval")
    parser.add_argument("--question", required=True, help="Question answered from chronological clip frames")
    parser.add_argument("--query-id", help="Optional stable Q&A identifier for debug output")
    parser.add_argument(
        "--encoder",
        choices=("btc_clip", "fgclip2_large", "pecore_g14_448", "fg_pe_fusion"),
        help="Configured retrieval mode",
    )
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--seed", type=int, help="Deterministic seed recorded in debug metadata")
    parser.add_argument("--candidate-count", type=int, help="Retrieved candidates before VLM answering")
    parser.add_argument("--answer-candidate-count", type=int, help="Top retrieval candidates sent to VLM")
    parser.add_argument("--multi-frame-count", type=int, help="Chronological original-video frames for VLM")
    parser.add_argument("--clip-window-sec", type=float, help="Half-window around event frame for VLM clip")
    parser.add_argument(
        "--feature-file",
        action="append",
        type=Path,
        help="BTC CLIP .npy feature shard; repeat in verified order",
    )
    parser.add_argument("--feature-order-manifest", type=Path, help="Verified BTC row-to-UID manifest")
    parser.add_argument("--keyframe-manifest", type=Path, help="Parquet keyframe metadata manifest")
    parser.add_argument("--checkpoint", type=Path, help="Local BTC-compatible checkpoint")
    parser.add_argument("--model-name", help="OpenCLIP model identifier")
    parser.add_argument("--device", help="Torch device for retrieval encoders")
    parser.add_argument("--batch-size", type=int, help="BTC exact-search vector batch size")
    parser.add_argument("--fg-embedding-manifest", type=Path, help="FG-CLIP2 embedding manifest")
    parser.add_argument("--fg-index-dir", type=Path, help="FG-CLIP2 FAISS index directory")
    parser.add_argument("--fg-model-id", help="FG-CLIP2 cached model ID or local path")
    parser.add_argument("--fg-revision", help="FG-CLIP2 cached model revision")
    parser.add_argument("--fg-batch-size", type=int, help="FG-CLIP2 query/refinement batch size")
    parser.add_argument("--pe-embedding-manifest", type=Path, help="PE-Core embedding manifest")
    parser.add_argument("--pe-index-dir", type=Path, help="PE-Core FAISS index directory")
    parser.add_argument("--pe-checkpoint", type=Path, help="Local PE-Core checkpoint")
    parser.add_argument("--pe-model-config", help="PE-Core config name from perception_models")
    parser.add_argument("--pe-batch-size", type=int, help="PE-Core query/refinement batch size")
    parser.add_argument("--video-manifest", type=Path, help="Parquet original-video metadata manifest")
    parser.add_argument("--vlm-checkpoint", type=Path, help="Local Qwen3-VL checkpoint directory")
    parser.add_argument("--vlm-device-map", help="Qwen3-VL transformers device_map")
    parser.add_argument("--vlm-dtype", help="Qwen3-VL transformers dtype")
    parser.add_argument("--max-new-tokens", type=int, help="Maximum generated answer tokens")
    parser.add_argument("--coarse-only", action="store_true", help="Skip M5 dense retrieval refinement")
    parser.add_argument("--no-temporal-nms", action="store_true", help="Disable coarse temporal NMS")
    parser.add_argument("--debug-output", type=Path, help="Write debug JSON to this path")
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    positive_names = (
        "candidate_count",
        "answer_candidate_count",
        "multi_frame_count",
        "batch_size",
        "fg_batch_size",
        "pe_batch_size",
        "max_new_tokens",
    )
    invalid = [
        f"--{name.replace('_', '-')}"
        for name in positive_names
        if getattr(args, name) is not None and getattr(args, name) < 1
    ]
    if args.clip_window_sec is not None and args.clip_window_sec < 0:
        invalid.append("--clip-window-sec")
    if args.seed is not None and args.seed < 0:
        invalid.append("--seed")
    if invalid:
        raise SystemExit("Invalid positive/non-negative values: " + ", ".join(invalid))


def _optional_path(value: object, root: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _print_candidates(result: object) -> None:
    for candidate in result.candidates:
        print(
            f"{candidate.rank:3}  {candidate.video_id:12}  frame={candidate.frame_id:8}  "
            f"answer={candidate.normalized_answer!r}  source={candidate.source}"
        )


def main() -> int:
    args = parse_arguments()
    _validate_arguments(args)
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    models_config = load_yaml_config(root / "configs" / "models.yaml")
    retrieval_config = load_yaml_config(root / "configs" / "retrieval.yaml")
    qna_config = load_yaml_config(root / "configs" / "qna.yaml")
    hardening_config = load_yaml_config(root / "configs" / "hardening.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    keyframe_manifest = args.keyframe_manifest or data_root / "manifests" / "keyframes_manifest.parquet"
    video_manifest = args.video_manifest or data_root / "manifests" / "videos_manifest.parquet"
    encoder_name = args.encoder or str(qna_config["encoder"])
    candidate_count = (
        args.candidate_count if args.candidate_count is not None else int(qna_config["candidate_count"])
    )
    answer_candidate_count = (
        args.answer_candidate_count
        if args.answer_candidate_count is not None
        else int(qna_config["answer_candidate_count"])
    )
    multi_frame_count = (
        args.multi_frame_count
        if args.multi_frame_count is not None
        else int(qna_config["multi_frame_count"])
    )
    clip_window_sec = (
        args.clip_window_sec if args.clip_window_sec is not None else float(qna_config["clip_window_sec"])
    )
    query = QnAQuery(
        event_description=args.event_description,
        question=args.question,
        query_id=args.query_id,
    )
    seed = args.seed if args.seed is not None else int(hardening_config["reproducibility"]["random_seed"])
    determinism = configure_determinism(seed)
    try:
        keyframes = run_kis.load_keyframe_records_from_parquet(keyframe_manifest)
        coarse_runtime = run_kis.build_kis_coarse_runtime(
            args=args,
            root=root,
            data_root=data_root,
            retrieval_config=retrieval_config,
            models_config=models_config,
            keyframes=keyframes,
            encoder_name=encoder_name,
        )
        refinement_enabled = bool(qna_config["refinement_enabled"]) and not args.coarse_only
        searcher = coarse_runtime.service
        if refinement_enabled:
            refiner = run_kis.build_dense_frame_refiner(
                data_root=data_root,
                video_manifest=video_manifest,
                retrieval_config=retrieval_config,
                refinement_branches=coarse_runtime.refinement_branches,
                frame_fusion_config=coarse_runtime.frame_fusion_config,
            )
            searcher = KisDenseRefinementService(coarse_runtime.service, refiner)
        video_records = load_video_records_from_parquet(video_manifest)
        vlm_config = models_config["vlm"]
        if vlm_config.get("local_files_only") is not True:
            raise VLMUnavailableError("configs/models.yaml must keep Qwen3-VL implicit downloads disabled")
        checkpoint = args.vlm_checkpoint or _optional_path(vlm_config.get("checkpoint"), root)
        if checkpoint is None:
            raise VLMUnavailableError(
                "No local Qwen3-VL checkpoint is configured. Set vlm.checkpoint or pass --vlm-checkpoint."
            )
        answerer = Qwen3VLAnswerer.from_local_checkpoint(
            checkpoint=checkpoint,
            device_map=args.vlm_device_map or str(vlm_config["device_map"]),
            dtype=args.vlm_dtype or str(vlm_config["dtype"]),
            max_new_tokens=args.max_new_tokens or int(vlm_config["max_new_tokens"]),
        )
        service = QnaService(
            searcher=searcher,
            clip_selector=CandidateClipSelector(
                decoder=OpenCVVideoDecoder(),
                sampler=FrameSampler(),
                video_records=video_records,
                data_root=data_root,
                config=ClipSelectorConfig(
                    window_sec=clip_window_sec,
                    multi_frame_count=multi_frame_count,
                ),
            ),
            answerer=answerer,
            answer_normalizer=AnswerNormalizer(),
            config=QnaServiceConfig(
                retrieval_candidate_count=candidate_count,
                answer_candidate_count=answer_candidate_count,
            ),
        )
        result = service.answer(query)
    except (
        ClipSelectionError,
        EncoderUnavailableError,
        FaissUnavailableError,
        FaissIndexValidationError,
        FeatureStoreValidationError,
        FeatureMappingVerificationError,
        FileNotFoundError,
        VLMAnswererError,
        VLMUnavailableError,
        VideoDecodingError,
        VideoManifestValidationError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    debug_path = args.debug_output or default_qna_debug_path(root / "outputs", query)
    write_qna_debug(
        result,
        debug_path,
        metadata={
            "selected_encoder": encoder_name,
            "runtime": coarse_runtime.runtime_metadata,
            "qna": {
                "candidate_count": candidate_count,
                "answer_candidate_count": answer_candidate_count,
                "multi_frame_count": multi_frame_count,
                "clip_window_sec": clip_window_sec,
                "refinement_enabled": refinement_enabled,
                "video_manifest": str(video_manifest),
            },
            "vlm": {
                "name": vlm_config["name"],
                "model_id": vlm_config["model_id"],
                "checkpoint": str(checkpoint),
                "device_map": args.vlm_device_map or vlm_config["device_map"],
                "dtype": args.vlm_dtype or vlm_config["dtype"],
            },
            "reproducibility": {
                "seed": determinism.seed,
                "torch_configured": determinism.torch_configured,
                "torch_error": determinism.torch_error,
            },
        },
    )
    _print_candidates(result)
    print(f"Debug JSON: {debug_path}")
    if not result.candidates:
        print("ERROR: No Q&A candidate was answered successfully", file=sys.stderr)
        return 2
    if result.failures:
        print(f"WARNING: {len(result.failures)} Q&A candidates failed; details are in debug JSON", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
