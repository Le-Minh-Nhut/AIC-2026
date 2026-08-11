#!/usr/bin/env python3
"""Build a persistent exact FAISS IndexFlatIP from image-text embedding shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import configured_data_root, load_yaml_config, repository_root
from indexing.faiss_index import (
    FaissIndexValidationError,
    FaissUnavailableError,
    build_faiss_flat_ip_index,
)
from indexing.feature_store import FeatureMappingVerificationError, FeatureStoreValidationError, load_keyframe_records_from_parquet
from indexing.sharded_feature_store import ShardedNpyFeatureStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("fgclip2_large", "pecore_g14_448"), default="fgclip2_large")
    parser.add_argument("--data-root", type=Path, help="Override configured data root")
    parser.add_argument("--embedding-manifest", type=Path, help="Encoder embedding manifest path")
    parser.add_argument("--keyframe-manifest", type=Path, help="Parquet keyframe metadata manifest")
    parser.add_argument("--output", type=Path, help="FAISS output directory")
    parser.add_argument("--batch-size", type=int, help="Vectors added per FAISS batch")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing index files")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    root = repository_root()
    data_config = load_yaml_config(root / "configs" / "data.yaml")
    retrieval_config = load_yaml_config(root / "configs" / "retrieval.yaml")
    data_root = configured_data_root(data_config, args.data_root)
    retrieval_key = "fgclip2" if args.encoder == "fgclip2_large" else "pecore"
    encoder_config = retrieval_config[retrieval_key]
    embedding_manifest = args.embedding_manifest or root / str(encoder_config["embeddings_dir"]) / "manifest.json"
    output = args.output or root / str(encoder_config["index_dir"])
    keyframe_manifest = args.keyframe_manifest or data_root / "manifests" / "keyframes_manifest.parquet"
    try:
        keyframes = load_keyframe_records_from_parquet(keyframe_manifest)
        store = ShardedNpyFeatureStore(embedding_manifest, keyframes, mmap=True)
        result = build_faiss_flat_ip_index(
            store,
            output,
            batch_size=args.batch_size or int(encoder_config["faiss_batch_size"]),
            overwrite=args.overwrite,
        )
    except (FaissUnavailableError, FaissIndexValidationError, FeatureMappingVerificationError, FeatureStoreValidationError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Index: {result.index_path}")
    print(f"Manifest: {result.manifest_path}; vectors: {result.count}; dimension: {result.dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
