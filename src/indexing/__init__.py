"""Feature storage and exact vector-search implementations."""

from indexing.exact_index import ExactCosineIndex
from indexing.embedding_pipeline import EmbeddingRunConfig, OfflineKeyframeEmbedder
from indexing.faiss_index import FaissFlatIPIndex, build_faiss_flat_ip_index, load_faiss_flat_ip_index
from indexing.feature_store import (
    BtcClipFeatureStore,
    FeatureMappingVerificationError,
    FeatureStoreValidationError,
    load_keyframe_records_from_parquet,
)
from indexing.sharded_feature_store import ShardedNpyFeatureStore

__all__ = [
    "BtcClipFeatureStore",
    "EmbeddingRunConfig",
    "ExactCosineIndex",
    "FaissFlatIPIndex",
    "FeatureMappingVerificationError",
    "FeatureStoreValidationError",
    "OfflineKeyframeEmbedder",
    "ShardedNpyFeatureStore",
    "build_faiss_flat_ip_index",
    "load_faiss_flat_ip_index",
    "load_keyframe_records_from_parquet",
]
