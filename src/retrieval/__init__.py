"""Reusable retrieval, fusion, post-processing, and aggregation services."""

from retrieval.fusion import (
    FusedRankedScore,
    FusionValidationError,
    RankedScore,
    WeightedRRFConfig,
    weighted_reciprocal_rank_fusion,
    weighted_reciprocal_rank_scores,
)
from retrieval.temporal_nms import temporal_nms
from retrieval.text_retriever import ASRTextRetriever, MetadataTextRetriever, OCRTextRetriever
from retrieval.video_aggregation import aggregate_video_candidates
from retrieval.visual_retriever import VectorRetriever

__all__ = [
    "FusionValidationError",
    "FusedRankedScore",
    "ASRTextRetriever",
    "MetadataTextRetriever",
    "OCRTextRetriever",
    "RankedScore",
    "VectorRetriever",
    "WeightedRRFConfig",
    "aggregate_video_candidates",
    "temporal_nms",
    "weighted_reciprocal_rank_fusion",
    "weighted_reciprocal_rank_scores",
]
