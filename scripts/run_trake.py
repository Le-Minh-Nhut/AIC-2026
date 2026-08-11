#!/usr/bin/env python3
"""Run ordered-event TRAKE with reusable FG, PE, or FG+PE coarse retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_kis
from config import configured_data_root, load_yaml_config, repository_root
from data.video_repository import VideoManifestValidationError
from encoders.base import EncoderUnavailableError
from indexing.faiss_index import FaissIndexValidationError, FaissUnavailableError
from indexing.feature_store import FeatureMappingVerificationError, FeatureStoreValidationError
from hardening.reproducibility import configure_determinism
from refinement.video_decoder import VideoDecodingError
from query.event_decomposer import EventDecompositionError, RuleBasedEventDecomposer
from tasks.trake_service import (
    TrakeService,
    TrakeServiceConfig,
    default_trake_debug_path,
    write_trake_debug,
)
from trake.event_refiner import TrakeDenseEventRefiner
from trake.temporal_aligner import TemporalAligner, TemporalAlignmentConfig
from trake.video_selector import CandidateVideoSelector


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Ordered event description or explicit event list")
    parser.add_argument(
        "--encoder",
        choices=("btc_clip", "fgclip2_large", "pecore_g14_448", "fg_pe_fusion"),
        help="Configured retrieval mode",
    )
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--seed", type=int, help="Deterministic seed recorded in debug metadata")
    parser.add_argument("--event-top-k", type=int, help="Coarse keyframes retained for each event")
    parser.add_argument("--candidate-videos", type=int, help="Union videos retained before DP")
    parser.add_argument("--k-best-sequences", type=int, help="Final diverse candidate sequences")
    parser.add_argument("--sequences-to-refine", type=int, help="Top coarse sequences sent to M5")
    parser.add_argument("--local-frame-candidates", type=int, help="Dense options per event before local DP")
    parser.add_argument("--min-temporal-gap-sec", type=float, help="Optional minimum inter-event gap")
    parser.add_argument("--max-temporal-gap-sec", type=float, help="Optional maximum inter-event gap")
    parser.add_argument("--gap-penalty", type=float, help="Linear penalty for gap beyond the configured minimum")
    parser.add_argument("--sequence-dedup-window-sec", type=float, help="Near-duplicate sequence window")
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
    parser.add_argument("--device", help="Torch device for text/image encoding")
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
    parser.add_argument("--coarse-only", action="store_true", help="Skip dense frame refinement")
    parser.add_argument("--no-temporal-nms", action="store_true", help="Disable coarse temporal NMS")
    parser.add_argument("--debug-output", type=Path, help="Write debug JSON to this path")
    return parser.parse_args()


def _validate_positive(arguments: argparse.Namespace) -> None:
    names = ("event_top_k", "candidate_videos", "k_best_sequences", "sequences_to_refine", "local_frame_candidates")
    invalid = [f"--{name.replace('_', '-')}" for name in names if getattr(arguments, name) is not None and getattr(arguments, name) < 1]
    if arguments.batch_size is not None and arguments.batch_size < 1:
        invalid.append("--batch-size")
    if arguments.fg_batch_size is not None and arguments.fg_batch_size < 1:
        invalid.append("--fg-batch-size")
    if arguments.pe_batch_size is not None and arguments.pe_batch_size < 1:
        invalid.append("--pe-batch-size")
    if arguments.seed is not None and arguments.seed < 0:
        invalid.append("--seed")
    if invalid:
        raise SystemExit("Values must be at least 1: " + ", ".join(invalid))


def _value(arguments: argparse.Namespace, config: dict[str, object], name: str, cast_type: type[int] | type[float]) -> int | float:
    value = getattr(arguments, name)
    return cast_type(value if value is not None else config[name])


def _print_candidates(result: object) -> None:
    for candidate in result.candidates:
        frames = ",".join(str(frame_id) for frame_id in candidate.final_alignment.frame_ids)
        print(
            f"{candidate.rank:3}  {candidate.video_id:12}  frames=[{frames}]  "
            f"score={candidate.total_alignment_score:.6f}  {candidate.refinement_status}"
        )


def main() -> int:
    args = parse_arguments()
    _validate_positive(args)
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    models_config = load_yaml_config(root / "configs" / "models.yaml")
    retrieval_config = load_yaml_config(root / "configs" / "retrieval.yaml")
    trake_config = load_yaml_config(root / "configs" / "trake.yaml")
    hardening_config = load_yaml_config(root / "configs" / "hardening.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    keyframe_manifest = args.keyframe_manifest or data_root / "manifests" / "keyframes_manifest.parquet"
    video_manifest = args.video_manifest or data_root / "manifests" / "videos_manifest.parquet"
    encoder_name = args.encoder or str(trake_config["encoder"])
    temporal_config = TemporalAlignmentConfig(
        min_temporal_gap_sec=float(_value(args, trake_config, "min_temporal_gap_sec", float)),
        max_temporal_gap_sec=(
            float(args.max_temporal_gap_sec)
            if args.max_temporal_gap_sec is not None
            else (
                float(trake_config["max_temporal_gap_sec"])
                if trake_config.get("max_temporal_gap_sec") is not None
                else None
            )
        ),
        gap_penalty=float(_value(args, trake_config, "gap_penalty", float)),
        k_best_sequences=int(_value(args, trake_config, "k_best_sequences", int)),
        sequence_dedup_window_sec=float(
            _value(args, trake_config, "sequence_dedup_window_sec", float)
        ),
    )
    service_config = TrakeServiceConfig(
        event_top_k=int(_value(args, trake_config, "event_top_k", int)),
        candidate_videos=int(_value(args, trake_config, "candidate_videos", int)),
        k_best_sequences=temporal_config.k_best_sequences,
        sequences_to_refine=int(_value(args, trake_config, "sequences_to_refine", int)),
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
        refinement_enabled = bool(trake_config["refinement_enabled"]) and not args.coarse_only
        event_refiner = None
        if refinement_enabled:
            frame_refiner = run_kis.build_dense_frame_refiner(
                data_root=data_root,
                video_manifest=video_manifest,
                retrieval_config=retrieval_config,
                refinement_branches=coarse_runtime.refinement_branches,
                frame_fusion_config=coarse_runtime.frame_fusion_config,
            )
            event_refiner = TrakeDenseEventRefiner(
                frame_refiner=frame_refiner,
                temporal_aligner=TemporalAligner(temporal_config),
                local_frame_candidates=int(_value(args, trake_config, "local_frame_candidates", int)),
            )
        service = TrakeService(
            event_decomposer=RuleBasedEventDecomposer(),
            coarse_searcher=coarse_runtime.service,
            video_selector=CandidateVideoSelector(),
            temporal_aligner=TemporalAligner(temporal_config),
            config=service_config,
            event_refiner=event_refiner,
        )
        result = service.search(args.query)
    except (
        EncoderUnavailableError,
        EventDecompositionError,
        FaissUnavailableError,
        FaissIndexValidationError,
        FeatureStoreValidationError,
        FeatureMappingVerificationError,
        FileNotFoundError,
        VideoDecodingError,
        VideoManifestValidationError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    debug_path = args.debug_output or default_trake_debug_path(root / "outputs", args.query)
    write_trake_debug(
        result,
        debug_path,
        metadata={
            "selected_encoder": encoder_name,
            "runtime": coarse_runtime.runtime_metadata,
            "trake": {
                "event_top_k": service_config.event_top_k,
                "candidate_videos": service_config.candidate_videos,
                "k_best_sequences": service_config.k_best_sequences,
                "sequences_to_refine": service_config.sequences_to_refine,
                "local_frame_candidates": _value(args, trake_config, "local_frame_candidates", int),
                "refinement_enabled": refinement_enabled,
                "video_manifest": str(video_manifest) if refinement_enabled else None,
                "temporal_alignment": {
                    "min_temporal_gap_sec": temporal_config.min_temporal_gap_sec,
                    "max_temporal_gap_sec": temporal_config.max_temporal_gap_sec,
                    "gap_penalty": temporal_config.gap_penalty,
                    "sequence_dedup_window_sec": temporal_config.sequence_dedup_window_sec,
                },
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
        print("ERROR: No complete monotonic TRAKE sequence was found", file=sys.stderr)
        return 2
    if result.refinement_failures:
        print(
            f"WARNING: {len(result.refinement_failures)} sequence refinement failures; details are in debug JSON",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
