#!/usr/bin/env python3
"""Run KIS coarse retrieval with BTC, FG-CLIP2, PE-Core, or FG+PE RRF."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config, repository_root
from data.video_repository import VideoManifestValidationError, load_video_records_from_parquet
from encoders.base import EncoderUnavailableError
from encoders.btc_clip import BtcClipTextEncoder, OpenClipTextBackend
from encoders.fgclip2 import FGCLIP2Encoder, HuggingFaceFGCLIP2Backend
from encoders.pecore import PECoreEncoder, PerceptionCoreBackend
from indexing.exact_index import ExactCosineIndex
from indexing.faiss_index import FaissIndexValidationError, FaissUnavailableError, load_faiss_flat_ip_index
from indexing.feature_store import (
    BtcClipFeatureStore,
    FeatureMappingVerificationError,
    FeatureStoreValidationError,
    load_keyframe_records_from_parquet,
)
from indexing.sharded_feature_store import ShardedNpyFeatureStore
from domain.models import KeyframeRecord
from domain.protocols import ImageTextEncoder, QueryCandidateRetriever, Retriever, TextEncoder
from hardening.reproducibility import configure_determinism
from refinement.dense_frame_refiner import (
    DenseFrameRefiner,
    FrameScoringBranch,
    RefinementConfig,
    VisualFrameScorer,
)
from refinement.frame_sampler import FrameSampler
from refinement.video_decoder import OpenCVVideoDecoder, VideoDecodingError
from retrieval.fusion import WeightedRRFConfig
from retrieval.auxiliary_branches import AuxiliaryBranchConfigError, build_auxiliary_text_branches
from retrieval.visual_retriever import VectorRetriever
from tasks.kis_service import (
    FusedKisCoarseRetrievalService,
    KisCoarseSearcher,
    KisCoarseRetrievalService,
    KisDenseRefinementService,
    KisMultiSourceRetrievalService,
    KisRefinementResult,
    VisualQueryCandidateRetriever,
    default_debug_path,
    write_kis_debug,
)


@dataclass(frozen=True, slots=True)
class PreparedRetrieverBranch:
    source: str
    text_encoder: TextEncoder
    retriever: Retriever
    runtime_metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class CoarseKisRuntime:
    """Reusable encoder/index wiring for KIS and ordered-event TRAKE."""

    service: KisCoarseSearcher
    runtime_metadata: dict[str, object]
    refinement_branches: Mapping[str, ImageTextEncoder]
    frame_fusion_config: WeightedRRFConfig | None
    query_branches: Mapping[str, QueryCandidateRetriever]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Event description to retrieve")
    parser.add_argument(
        "--encoder",
        choices=("btc_clip", "fgclip2_large", "pecore_g14_448", "fg_pe_fusion"),
        help="Configured retrieval mode",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of temporally diverse keyframes")
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--seed", type=int, help="Deterministic seed recorded in debug metadata")
    parser.add_argument(
        "--feature-file",
        action="append",
        type=Path,
        help="BTC CLIP .npy feature shard; repeat in the order recorded by --feature-order-manifest",
    )
    parser.add_argument(
        "--feature-order-manifest",
        type=Path,
        help="Verified JSON row-to-keyframe UID manifest",
    )
    parser.add_argument("--keyframe-manifest", type=Path, help="Parquet keyframe metadata manifest")
    parser.add_argument("--checkpoint", type=Path, help="Local BTC-compatible ViT-B/32 checkpoint")
    parser.add_argument("--model-name", help="OpenCLIP model identifier")
    parser.add_argument("--device", help="Torch device for text encoding")
    parser.add_argument("--batch-size", type=int, help="BTC exact-search vector batch size")
    parser.add_argument("--fg-embedding-manifest", type=Path, help="FG-CLIP2 embedding manifest")
    parser.add_argument("--fg-index-dir", type=Path, help="FG-CLIP2 FAISS index directory")
    parser.add_argument("--fg-model-id", help="FG-CLIP2 cached model ID or local model path")
    parser.add_argument("--fg-revision", help="FG-CLIP2 cached model revision")
    parser.add_argument("--fg-batch-size", type=int, help="FG-CLIP2 query encoding batch size")
    parser.add_argument("--pe-embedding-manifest", type=Path, help="PE-Core embedding manifest")
    parser.add_argument("--pe-index-dir", type=Path, help="PE-Core FAISS index directory")
    parser.add_argument("--pe-checkpoint", type=Path, help="Local PE-Core checkpoint")
    parser.add_argument("--pe-model-config", help="PE-Core config name from perception_models")
    parser.add_argument("--pe-batch-size", type=int, help="PE-Core query encoding batch size")
    parser.add_argument("--video-manifest", type=Path, help="Parquet original-video metadata manifest")
    parser.add_argument("--coarse-only", action="store_true", help="Skip configured dense frame refinement")
    parser.add_argument("--no-temporal-nms", action="store_true", help="Disable configured temporal NMS")
    parser.add_argument("--debug-output", type=Path, help="Write debug JSON to this path")
    return parser.parse_args()


def load_btc_text_encoder(model_name: str, checkpoint: Path | None, device: str) -> BtcClipTextEncoder:
    if checkpoint is None:
        raise EncoderUnavailableError(
            "No local BTC-compatible CLIP checkpoint is configured. Set encoders.btc_clip.checkpoint "
            "in configs/models.yaml or pass --checkpoint. The baseline never downloads models implicitly."
        )
    backend = OpenClipTextBackend.from_local_checkpoint(checkpoint, model_name=model_name, device=device)
    return BtcClipTextEncoder(backend)


def load_fgclip2_text_encoder(model_config: dict[str, object], args: argparse.Namespace) -> FGCLIP2Encoder:
    if model_config.get("local_files_only") is not True:
        raise EncoderUnavailableError("FG-CLIP2 implicit model downloads must remain disabled")
    backend = HuggingFaceFGCLIP2Backend.from_pretrained(
        model_id=args.fg_model_id or str(model_config["model_id"]),
        revision=args.fg_revision if args.fg_revision is not None else model_config.get("revision"),
        device=args.device or str(model_config["device"]),
        local_files_only=True,
        max_num_patches=int(model_config["max_num_patches"]),
        text_max_length=int(model_config["text_max_length"]),
        text_walk_type=str(model_config["text_walk_type"]),
        use_autocast=bool(model_config["use_autocast"]),
    )
    return FGCLIP2Encoder(backend, batch_size=args.fg_batch_size or int(model_config["batch_size"]))


def load_pecore_text_encoder(
    model_config: dict[str, object],
    args: argparse.Namespace,
    root: Path,
) -> PECoreEncoder:
    if model_config.get("allow_model_download"):
        raise EncoderUnavailableError("configs/models.yaml must keep PE-Core implicit downloads disabled")
    checkpoint = args.pe_checkpoint or _optional_path(model_config.get("checkpoint"), root)
    if checkpoint is None:
        raise EncoderUnavailableError(
            "No local PE-Core checkpoint is configured. Set encoders.secondary.checkpoint "
            "or pass --pe-checkpoint; this repository never downloads a PE-Core model implicitly."
        )
    backend = PerceptionCoreBackend.from_local_checkpoint(
        checkpoint=checkpoint,
        model_config=args.pe_model_config or str(model_config["model_config"]),
        device=args.device or str(model_config["device"]),
        use_autocast=bool(model_config["use_autocast"]),
    )
    return PECoreEncoder(backend, batch_size=args.pe_batch_size or int(model_config["batch_size"]))


def _optional_path(value: object, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _feature_paths(arguments: argparse.Namespace, data_root: Path) -> list[Path]:
    if arguments.feature_file:
        return arguments.feature_file
    feature_root = data_root / "raw" / "btc_clip_features"
    return sorted(feature_root.rglob("*.npy")) if feature_root.exists() else []


def _print_candidates(result: Any) -> None:
    for candidate in result.candidates:
        source_keyframe_uid = (
            candidate.keyframe_uid if hasattr(candidate, "keyframe_uid") else candidate.source_keyframe_uid
        )
        global_score = getattr(candidate, "coarse_score", candidate.score)
        score_detail = (
            f"global={global_score:.6f} local={candidate.score:.6f}"
            if hasattr(candidate, "coarse_score")
            else f"score={candidate.score:.6f}"
        )
        print(
            f"{candidate.rank:3}  {candidate.video_id:12}  frame={candidate.original_frame_id:8}  "
            f"time={candidate.timestamp_sec:9.3f}  {score_detail}  "
            f"{source_keyframe_uid}"
        )


def _load_faiss_branch(
    source: str,
    embedding_manifest: Path,
    index_dir: Path,
    keyframes: tuple[KeyframeRecord, ...],
    text_encoder: TextEncoder,
    encoder_metadata: dict[str, object],
) -> PreparedRetrieverBranch:
    feature_store = ShardedNpyFeatureStore(embedding_manifest, keyframes, mmap=True)
    index = load_faiss_flat_ip_index(index_dir)
    index.validate_feature_store(feature_store)
    return PreparedRetrieverBranch(
        source=source,
        text_encoder=text_encoder,
        retriever=VectorRetriever(index, feature_store, source=source),
        runtime_metadata={
            "encoder": encoder_metadata,
            "index": {"type": "faiss_index_flat_ip", "index_dir": str(index_dir)},
            "feature_store": {
                "count": feature_store.count,
                "dimension": feature_store.dimension,
                "embedding_manifest": str(feature_store.manifest_path),
                "validation": asdict(feature_store.validation),
            },
        },
    )


def _fg_branch(
    args: argparse.Namespace,
    root: Path,
    retrieval_config: dict[str, object],
    models_config: dict[str, object],
    keyframes: tuple[KeyframeRecord, ...],
) -> PreparedRetrieverBranch:
    fg_config = retrieval_config["fgclip2"]
    model_config = models_config["encoders"]["primary"]
    embedding_manifest = args.fg_embedding_manifest or root / str(fg_config["embeddings_dir"]) / "manifest.json"
    index_dir = args.fg_index_dir or root / str(fg_config["index_dir"])
    text_encoder = load_fgclip2_text_encoder(model_config, args)
    return _load_faiss_branch(
        source="fgclip2",
        embedding_manifest=embedding_manifest,
        index_dir=index_dir,
        keyframes=keyframes,
        text_encoder=text_encoder,
        encoder_metadata={
            "adapter": "huggingface_fgclip2",
            "model_id": args.fg_model_id or model_config["model_id"],
            "revision": args.fg_revision if args.fg_revision is not None else model_config.get("revision"),
            "device": args.device or model_config["device"],
        },
    )


def _pecore_branch(
    args: argparse.Namespace,
    root: Path,
    retrieval_config: dict[str, object],
    models_config: dict[str, object],
    keyframes: tuple[KeyframeRecord, ...],
) -> PreparedRetrieverBranch:
    pe_config = retrieval_config["pecore"]
    model_config = models_config["encoders"]["secondary"]
    embedding_manifest = args.pe_embedding_manifest or root / str(pe_config["embeddings_dir"]) / "manifest.json"
    index_dir = args.pe_index_dir or root / str(pe_config["index_dir"])
    text_encoder = load_pecore_text_encoder(model_config, args, root)
    checkpoint = args.pe_checkpoint or _optional_path(model_config.get("checkpoint"), root)
    return _load_faiss_branch(
        source="pecore",
        embedding_manifest=embedding_manifest,
        index_dir=index_dir,
        keyframes=keyframes,
        text_encoder=text_encoder,
        encoder_metadata={
            "adapter": "perception_models_pecore",
            "model_id": model_config["model_id"],
            "model_config": args.pe_model_config or model_config["model_config"],
            "checkpoint": str(checkpoint) if checkpoint else None,
            "device": args.device or model_config["device"],
        },
    )


def build_kis_coarse_runtime(
    args: argparse.Namespace,
    root: Path,
    data_root: Path,
    retrieval_config: dict[str, object],
    models_config: dict[str, object],
    keyframes: tuple[KeyframeRecord, ...],
    encoder_name: str,
) -> CoarseKisRuntime:
    """Create the existing M2–M4 coarse service without enabling M5 refinement."""

    if encoder_name == "btc_clip":
        btc_config = models_config["encoders"]["btc_clip"]
        if btc_config.get("allow_model_download"):
            raise EncoderUnavailableError("configs/models.yaml must keep BTC CLIP implicit downloads disabled")
        features = _feature_paths(args, data_root)
        feature_order_manifest = args.feature_order_manifest or data_root / "manifests" / "btc_clip_feature_order.json"
        checkpoint = args.checkpoint or _optional_path(btc_config.get("checkpoint"), root)
        model_name = args.model_name or str(btc_config["model_name"])
        device = args.device or str(btc_config["device"])
        batch_size = args.batch_size or int(retrieval_config["search"]["exact_batch_size"])
        feature_store = BtcClipFeatureStore(features, keyframes, feature_order_manifest, mmap=True)
        text_encoder = load_btc_text_encoder(model_name, checkpoint, device)
        index = ExactCosineIndex(feature_store, batch_size=batch_size)
        runtime_metadata = {
            "encoder": {
                "adapter": "open_clip",
                "model_name": model_name,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "device": device,
            },
            "index": {"type": "exact_cosine", "batch_size": batch_size},
            "feature_store": {
                "count": feature_store.count,
                "dimension": feature_store.dimension,
                "verification_method": feature_store.verification_method,
                "validation": asdict(feature_store.validation),
            },
        }
        retriever = VectorRetriever(index, feature_store, source="btc_clip")
        refinement_branches: dict[str, ImageTextEncoder] = {}
        frame_fusion_config: WeightedRRFConfig | None = None
        query_branches: dict[str, QueryCandidateRetriever] = {
            "btc_clip": VisualQueryCandidateRetriever("btc_clip", text_encoder, retriever)
        }
    elif encoder_name == "fgclip2_large":
        branch = _fg_branch(args, root, retrieval_config, models_config, keyframes)
        text_encoder = branch.text_encoder
        retriever = branch.retriever
        runtime_metadata = branch.runtime_metadata
        refinement_branches = {branch.source: cast(ImageTextEncoder, branch.text_encoder)}
        frame_fusion_config = None
        query_branches = {
            branch.source: VisualQueryCandidateRetriever(branch.source, branch.text_encoder, branch.retriever)
        }
    elif encoder_name == "pecore_g14_448":
        branch = _pecore_branch(args, root, retrieval_config, models_config, keyframes)
        text_encoder = branch.text_encoder
        retriever = branch.retriever
        runtime_metadata = branch.runtime_metadata
        refinement_branches = {branch.source: cast(ImageTextEncoder, branch.text_encoder)}
        frame_fusion_config = None
        query_branches = {
            branch.source: VisualQueryCandidateRetriever(branch.source, branch.text_encoder, branch.retriever)
        }
    elif encoder_name != "fg_pe_fusion":
        raise ValueError(f"Unsupported KIS encoder mode: {encoder_name}")
    nms_config = retrieval_config["temporal_nms"]
    aggregation_config = retrieval_config["video_aggregation"]
    service_kwargs = {
        "temporal_nms_enabled": bool(nms_config["enabled"]) and not args.no_temporal_nms,
        "temporal_nms_window_sec": float(nms_config["window_sec"]),
        "candidate_pool_multiplier": int(retrieval_config["search"]["candidate_pool_multiplier"]),
        "video_aggregation_method": str(aggregation_config["method"]),
        "video_aggregation_top_m": int(aggregation_config["top_m"]),
    }
    if encoder_name == "fg_pe_fusion":
        fg_branch = _fg_branch(args, root, retrieval_config, models_config, keyframes)
        pe_branch = _pecore_branch(args, root, retrieval_config, models_config, keyframes)
        fusion_config = retrieval_config["fusion"]
        if fusion_config.get("method") != "rrf":
            raise ValueError("Milestone 4 supports only fusion.method=rrf")
        weights = fusion_config["weights"]
        frame_fusion_config = WeightedRRFConfig(
            k=int(fusion_config["rrf_k"]),
            weights={"fgclip2": float(weights["fgclip2"]), "pecore": float(weights["pecore"])},
        )
        service: KisCoarseSearcher = FusedKisCoarseRetrievalService(
            branches={
                fg_branch.source: (fg_branch.text_encoder, fg_branch.retriever),
                pe_branch.source: (pe_branch.text_encoder, pe_branch.retriever),
            },
            fusion_config=frame_fusion_config,
            **service_kwargs,
        )
        runtime_metadata = {
            "fusion": {
                "method": "rrf",
                "rrf_k": frame_fusion_config.k,
                "weights": dict(frame_fusion_config.weights),
            },
            "branches": {
                fg_branch.source: fg_branch.runtime_metadata,
                pe_branch.source: pe_branch.runtime_metadata,
            },
        }
        refinement_branches = {
            fg_branch.source: cast(ImageTextEncoder, fg_branch.text_encoder),
            pe_branch.source: cast(ImageTextEncoder, pe_branch.text_encoder),
        }
        query_branches = {
            fg_branch.source: VisualQueryCandidateRetriever(
                fg_branch.source,
                fg_branch.text_encoder,
                fg_branch.retriever,
            ),
            pe_branch.source: VisualQueryCandidateRetriever(
                pe_branch.source,
                pe_branch.text_encoder,
                pe_branch.retriever,
            ),
        }
    else:
        service = KisCoarseRetrievalService(text_encoder=text_encoder, retriever=retriever, **service_kwargs)
    auxiliary_config = retrieval_config.get("auxiliary_retrieval", {})
    if not isinstance(auxiliary_config, Mapping):
        raise AuxiliaryBranchConfigError("auxiliary_retrieval must be a mapping")
    auxiliary_runtime = build_auxiliary_text_branches(auxiliary_config, data_root, keyframes)
    if auxiliary_runtime.branches:
        all_branches = {**query_branches, **auxiliary_runtime.branches}
        fusion_config = retrieval_config["fusion"]
        if fusion_config.get("method") != "rrf":
            raise ValueError("Multimodal retrieval supports only fusion.method=rrf")
        configured_weights = fusion_config["weights"]
        missing_weights = sorted(set(all_branches) - set(configured_weights))
        if missing_weights:
            raise ValueError("Missing RRF weights for enabled sources: " + ", ".join(missing_weights))
        multimodal_rrf = WeightedRRFConfig(
            k=int(fusion_config["rrf_k"]),
            weights={source: float(configured_weights[source]) for source in sorted(all_branches)},
        )
        service = KisMultiSourceRetrievalService(
            branches=all_branches,
            fusion_config=multimodal_rrf,
            **service_kwargs,
        )
        runtime_metadata = {
            **runtime_metadata,
            "multimodal_fusion": {
                "method": "rrf",
                "rrf_k": multimodal_rrf.k,
                "weights": dict(multimodal_rrf.weights),
            },
        }
    runtime_metadata = {**runtime_metadata, "auxiliary_retrieval": auxiliary_runtime.metadata}
    return CoarseKisRuntime(
        service=service,
        runtime_metadata=runtime_metadata,
        refinement_branches=refinement_branches,
        frame_fusion_config=frame_fusion_config,
        query_branches=query_branches,
    )


def build_dense_frame_refiner(
    data_root: Path,
    video_manifest: Path,
    retrieval_config: dict[str, object],
    refinement_branches: Mapping[str, ImageTextEncoder],
    frame_fusion_config: WeightedRRFConfig | None,
) -> DenseFrameRefiner:
    """Construct the M5 refiner for KIS or TRAKE without duplicating frame logic."""

    if not refinement_branches:
        raise ValueError("Dense frame refinement requires FG-CLIP2, PE-Core, or FG+PE fusion mode")
    video_records = load_video_records_from_parquet(video_manifest)
    refinement_config = retrieval_config["refinement"]
    frame_scorer = VisualFrameScorer(
        branches=tuple(
            FrameScoringBranch(source=source, encoder=encoder)
            for source, encoder in refinement_branches.items()
        ),
        fusion_config=frame_fusion_config,
    )
    return DenseFrameRefiner(
        decoder=OpenCVVideoDecoder(),
        sampler=FrameSampler(),
        scorer=frame_scorer,
        video_records=video_records,
        data_root=data_root,
        config=RefinementConfig(
            coarse_window_sec=float(refinement_config["coarse_window_sec"]),
            sparse_fps=float(refinement_config["sparse_fps"]),
            dense_window_sec=float(refinement_config["dense_window_sec"]),
            candidate_count=int(refinement_config["candidate_count"]),
        ),
    )


def main() -> int:
    args = parse_arguments()
    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.fg_batch_size is not None and args.fg_batch_size < 1:
        raise SystemExit("--fg-batch-size must be at least 1")
    if args.pe_batch_size is not None and args.pe_batch_size < 1:
        raise SystemExit("--pe-batch-size must be at least 1")
    if args.seed is not None and args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    model_config = load_yaml_config(root / "configs" / "models.yaml")
    retrieval_config = load_yaml_config(root / "configs" / "retrieval.yaml")
    hardening_config = load_yaml_config(root / "configs" / "hardening.yaml")
    kis_config = load_yaml_config(root / "configs" / "kis.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    keyframe_manifest = args.keyframe_manifest or data_root / "manifests" / "keyframes_manifest.parquet"
    video_manifest = args.video_manifest or data_root / "manifests" / "videos_manifest.parquet"
    top_k = args.top_k or int(kis_config["candidate_count"])
    encoder_name = args.encoder or str(kis_config["encoder"])
    seed = args.seed if args.seed is not None else int(hardening_config["reproducibility"]["random_seed"])
    determinism = configure_determinism(seed)
    try:
        keyframes = load_keyframe_records_from_parquet(keyframe_manifest)
        coarse_runtime = build_kis_coarse_runtime(
            args=args,
            root=root,
            data_root=data_root,
            retrieval_config=retrieval_config,
            models_config=model_config,
            keyframes=keyframes,
            encoder_name=encoder_name,
        )
        service: KisCoarseSearcher | KisDenseRefinementService = coarse_runtime.service
        runtime_metadata = coarse_runtime.runtime_metadata
        refinement_config = retrieval_config["refinement"]
        refinement_enabled = (
            bool(refinement_config["enabled"])
            and not bool(kis_config.get("coarse_only", False))
            and not args.coarse_only
        )
        if refinement_enabled:
            refiner = build_dense_frame_refiner(
                data_root=data_root,
                video_manifest=video_manifest,
                retrieval_config=retrieval_config,
                refinement_branches=coarse_runtime.refinement_branches,
                frame_fusion_config=coarse_runtime.frame_fusion_config,
            )
            service = KisDenseRefinementService(service, refiner)
            runtime_metadata = {
                **runtime_metadata,
                "refinement": {
                    "enabled": True,
                    "video_manifest": str(video_manifest),
                    "coarse_window_sec": refinement_config["coarse_window_sec"],
                    "sparse_fps": refinement_config["sparse_fps"],
                    "dense_window_sec": refinement_config["dense_window_sec"],
                    "candidate_count": refinement_config["candidate_count"],
                },
            }
        else:
            runtime_metadata = {**runtime_metadata, "refinement": {"enabled": False}}
        result = service.search(args.query, top_k)
    except (
        EncoderUnavailableError,
        AuxiliaryBranchConfigError,
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
    debug_path = args.debug_output or default_debug_path(root / "outputs", args.query)
    write_kis_debug(
        result,
        debug_path,
        metadata={
            "selected_encoder": encoder_name,
            **runtime_metadata,
            "reproducibility": {
                "seed": determinism.seed,
                "torch_configured": determinism.torch_configured,
                "torch_error": determinism.torch_error,
            },
        },
    )
    _print_candidates(result)
    print(f"Debug JSON: {debug_path}")
    if isinstance(result, KisRefinementResult) and result.failures:
        if not result.candidates:
            print("ERROR: All selected coarse candidates failed dense frame refinement", file=sys.stderr)
            return 2
        print(
            f"WARNING: {len(result.failures)} coarse candidates failed dense frame refinement; "
            "details are in debug JSON",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
