#!/usr/bin/env python3
"""Encode keyframes into resumable FG-CLIP2 or PE-Core embedding shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config, repository_root
from encoders.base import EncoderUnavailableError
from encoders.fgclip2 import FGCLIP2Encoder, HuggingFaceFGCLIP2Backend
from encoders.pecore import PECoreEncoder, PerceptionCoreBackend
from indexing.embedding_pipeline import EmbeddingPipelineError, EmbeddingRunConfig, OfflineKeyframeEmbedder
from indexing.feature_store import FeatureStoreValidationError, load_keyframe_records_from_parquet


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("fgclip2_large", "pecore_g14_448"), default="fgclip2_large")
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--output", type=Path, help="Embedding output directory")
    parser.add_argument("--keyframe-manifest", type=Path, help="Parquet keyframe metadata manifest")
    parser.add_argument("--model-id", help="Model ID recorded in the embedding manifest")
    parser.add_argument("--revision", help="Optional cached FG-CLIP2 model revision")
    parser.add_argument("--checkpoint", type=Path, help="Local PE-Core checkpoint")
    parser.add_argument("--model-config", help="PE-Core config name from perception_models")
    parser.add_argument("--device", help="Torch device")
    parser.add_argument("--batch-size", type=int, help="Images per encoder call")
    parser.add_argument("--shard-size", type=int, help="Embeddings per output shard")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"))
    parser.add_argument("--no-resume", action="store_true", help="Fail rather than resume existing output")
    return parser.parse_args()


def load_fgclip2_encoder(model_config: dict[str, object], overrides: argparse.Namespace) -> FGCLIP2Encoder:
    if model_config.get("local_files_only") is not True:
        raise EncoderUnavailableError(
            "FG-CLIP2 implicit downloads must remain disabled; prepare the model cache on the target PC first"
        )
    backend = HuggingFaceFGCLIP2Backend.from_pretrained(
        model_id=overrides.model_id or str(model_config["model_id"]),
        revision=overrides.revision if overrides.revision is not None else model_config.get("revision"),
        device=overrides.device or str(model_config["device"]),
        local_files_only=True,
        max_num_patches=int(model_config["max_num_patches"]),
        text_max_length=int(model_config["text_max_length"]),
        text_walk_type=str(model_config["text_walk_type"]),
        use_autocast=bool(model_config["use_autocast"]),
    )
    return FGCLIP2Encoder(backend, batch_size=overrides.batch_size or int(model_config["batch_size"]))


def _optional_path(value: object, root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def load_pecore_encoder(
    model_config: dict[str, object],
    overrides: argparse.Namespace,
    root: Path,
) -> PECoreEncoder:
    if model_config.get("allow_model_download"):
        raise EncoderUnavailableError("PE-Core implicit downloads must remain disabled")
    checkpoint = overrides.checkpoint or _optional_path(model_config.get("checkpoint"), root)
    if checkpoint is None:
        raise EncoderUnavailableError(
            "PE-Core needs a local checkpoint. Set encoders.secondary.checkpoint or pass --checkpoint."
        )
    backend = PerceptionCoreBackend.from_local_checkpoint(
        checkpoint=checkpoint,
        model_config=overrides.model_config or str(model_config["model_config"]),
        device=overrides.device or str(model_config["device"]),
        use_autocast=bool(model_config["use_autocast"]),
    )
    return PECoreEncoder(backend, batch_size=overrides.batch_size or int(model_config["batch_size"]))


def main() -> int:
    args = parse_arguments()
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.shard_size is not None and args.shard_size < 1:
        raise SystemExit("--shard-size must be at least 1")
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    models_config = load_yaml_config(root / "configs" / "models.yaml")
    retrieval_config = load_yaml_config(root / "configs" / "retrieval.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    model_key = "primary" if args.encoder == "fgclip2_large" else "secondary"
    retrieval_key = "fgclip2" if args.encoder == "fgclip2_large" else "pecore"
    model_config = models_config["encoders"][model_key]
    output = args.output or root / str(retrieval_config[retrieval_key]["embeddings_dir"])
    keyframe_manifest = args.keyframe_manifest or data_root / "manifests" / "keyframes_manifest.parquet"
    storage_dtype = args.storage_dtype or str(model_config["storage_dtype"])
    try:
        keyframes = load_keyframe_records_from_parquet(keyframe_manifest)
        encoder = (
            load_fgclip2_encoder(model_config, args)
            if args.encoder == "fgclip2_large"
            else load_pecore_encoder(model_config, args, root)
        )
        config = EmbeddingRunConfig(
            encoder_name=args.encoder,
            model_id=args.model_id or str(model_config["model_id"]),
            model_revision=(
                args.revision if args.revision is not None else model_config.get("revision")
            ),
            preprocessing_config=encoder.preprocessing_config,
            output_dir=output,
            data_root=data_root,
            shard_size=args.shard_size or int(model_config["shard_size"]),
            batch_size=encoder.batch_size,
            storage_dtype=storage_dtype,
        )
        result = OfflineKeyframeEmbedder(encoder, config).run(keyframes, resume=not args.no_resume)
    except (EncoderUnavailableError, EmbeddingPipelineError, FeatureStoreValidationError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Manifest: {result.manifest_path}")
    print(f"Embeddings: {result.count}; dimension: {result.dimension}; new shards: {result.encoded_shards}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
