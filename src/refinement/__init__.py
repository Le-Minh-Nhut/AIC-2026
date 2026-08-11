"""Coarse-to-dense original-video frame refinement primitives."""

from refinement.dense_frame_refiner import (
    DenseFrameRefiner,
    FrameScoringBranch,
    RefinedFrameCandidate,
    RefinementConfig,
    VisualFrameScorer,
)
from refinement.frame_sampler import FrameSampler
from refinement.video_decoder import OpenCVVideoDecoder

__all__ = [
    "DenseFrameRefiner",
    "FrameSampler",
    "FrameScoringBranch",
    "OpenCVVideoDecoder",
    "RefinedFrameCandidate",
    "RefinementConfig",
    "VisualFrameScorer",
]
