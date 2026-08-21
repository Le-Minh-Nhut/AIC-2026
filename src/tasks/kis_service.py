"""Milestone 2 KIS coarse retrieval orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from domain.models import Candidate, VideoCandidate
from domain.protocols import QueryCandidateRetriever, Retriever, TextEncoder
from refinement.dense_frame_refiner import (
    DenseFrameRefiner,
    RefinedFrameCandidate,
    RefinementFailure,
)
from retrieval.fusion import WeightedRRFConfig, weighted_reciprocal_rank_fusion
from retrieval.temporal_nms import temporal_nms
from retrieval.video_aggregation import AggregationMethod, aggregate_video_candidates


@dataclass(frozen=True, slots=True)
class KisCoarseResult:
    query: str
    candidates: tuple[Candidate, ...]
    video_candidates: tuple[VideoCandidate, ...]
    initial_candidate_count: int
    temporal_nms_enabled: bool
    source_rankings: dict[str, tuple[Candidate, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        candidates = [_candidate_dict(candidate) for candidate in self.candidates]
        return {
            "query": self.query,
            "initial_candidate_count": self.initial_candidate_count,
            "temporal_nms_enabled": self.temporal_nms_enabled,
            "candidates": candidates,
            "source_rankings": {
                source: [_candidate_dict(candidate) for candidate in ranking]
                for source, ranking in sorted(self.source_rankings.items())
            },
            "video_candidates": [asdict(candidate) for candidate in self.video_candidates],
        }


def _candidate_dict(candidate: Candidate) -> dict[str, object]:
    value = asdict(candidate)
    value["frame_id"] = candidate.original_frame_id
    return value


def _refined_frame_dict(candidate: RefinedFrameCandidate) -> dict[str, object]:
    value = asdict(candidate)
    value["frame_id"] = candidate.original_frame_id
    value["coarse_frame_id"] = candidate.coarse_original_frame_id
    value["sparse_frame_id"] = candidate.sparse_original_frame_id
    value["global_score"] = candidate.coarse_score
    value["local_score"] = candidate.score
    return value


@dataclass(frozen=True, slots=True)
class KisRefinementResult:
    """KIS coarse debug plus all coarse candidates with refined/fallback frames."""

    coarse_result: KisCoarseResult
    candidates: tuple[RefinedFrameCandidate, ...]
    failures: tuple[RefinementFailure, ...]

    def as_dict(self) -> dict[str, object]:
        refined_count = sum(
            candidate.refinement_status == "refined" for candidate in self.candidates
        )
        fallback_count = len(self.candidates) - refined_count
        return {
            "query": self.coarse_result.query,
            "candidates": [_refined_frame_dict(candidate) for candidate in self.candidates],
            "coarse": self.coarse_result.as_dict(),
            "refinement": {
                "refined_candidate_count": refined_count,
                "fallback_candidate_count": fallback_count,
                "final_candidate_count": len(self.candidates),
                "failure_count": len(self.failures),
                "failures": [asdict(failure) for failure in self.failures],
            },
        }


class KisCoarseSearcher(Protocol):
    def search(self, query: str, top_k: int) -> KisCoarseResult: ...


class VisualQueryCandidateRetriever:
    """Adapts an existing text encoder/vector retriever to query-text candidate ranking."""

    def __init__(self, source: str, text_encoder: TextEncoder, retriever: Retriever) -> None:
        if not source:
            raise ValueError("Visual retrieval source must be non-empty")
        self.source = source
        self._text_encoder = text_encoder
        self._retriever = retriever

    def retrieve(self, query: str, top_k: int) -> Sequence[Candidate]:
        return self._retriever.retrieve(self._text_encoder.encode_texts([query])[0], top_k)


class KisCandidatePostprocessor:
    """Shared temporal NMS and video aggregation after one or many rankers."""

    def __init__(
        self,
        temporal_nms_enabled: bool,
        temporal_nms_window_sec: float,
        candidate_pool_multiplier: int,
        video_aggregation_method: AggregationMethod = "max",
        video_aggregation_top_m: int = 3,
    ) -> None:
        if temporal_nms_window_sec < 0:
            raise ValueError("temporal_nms_window_sec must be non-negative")
        if candidate_pool_multiplier < 1:
            raise ValueError("candidate_pool_multiplier must be at least 1")
        self._temporal_nms_enabled = temporal_nms_enabled
        self._temporal_nms_window_sec = temporal_nms_window_sec
        self._candidate_pool_multiplier = candidate_pool_multiplier
        self._video_aggregation_method = video_aggregation_method
        self._video_aggregation_top_m = video_aggregation_top_m

    def process(
        self,
        query: str,
        initial: Sequence[Candidate],
        top_k: int,
        source_rankings: Mapping[str, Sequence[Candidate]] | None = None,
    ) -> KisCoarseResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self._temporal_nms_enabled:
            candidates = temporal_nms(initial, self._temporal_nms_window_sec, max_candidates=top_k)
        else:
            candidates = tuple(
                replace(candidate, rank=rank)
                for rank, candidate in enumerate(initial[:top_k], start=1)
            )
        videos = aggregate_video_candidates(
            candidates,
            method=self._video_aggregation_method,
            top_m=self._video_aggregation_top_m,
        )
        return KisCoarseResult(
            query=query,
            candidates=candidates,
            video_candidates=videos,
            initial_candidate_count=len(initial),
            temporal_nms_enabled=self._temporal_nms_enabled,
            source_rankings={
                source: tuple(ranking) for source, ranking in (source_rankings or {}).items()
            },
        )

    @property
    def candidate_pool_multiplier(self) -> int:
        return self._candidate_pool_multiplier


class KisCoarseRetrievalService:
    """KIS coarse retrieval for a single encoder branch."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        retriever: Retriever,
        temporal_nms_enabled: bool,
        temporal_nms_window_sec: float,
        candidate_pool_multiplier: int,
        video_aggregation_method: AggregationMethod = "max",
        video_aggregation_top_m: int = 3,
    ) -> None:
        self._text_encoder = text_encoder
        self._retriever = retriever
        self._postprocessor = KisCandidatePostprocessor(
            temporal_nms_enabled=temporal_nms_enabled,
            temporal_nms_window_sec=temporal_nms_window_sec,
            candidate_pool_multiplier=candidate_pool_multiplier,
            video_aggregation_method=video_aggregation_method,
            video_aggregation_top_m=video_aggregation_top_m,
        )

    def search(self, query: str, top_k: int) -> KisCoarseResult:
        query_vector = self._text_encoder.encode_texts([query])[0]
        initial = tuple(
            self._retriever.retrieve(query_vector, top_k * self._postprocessor.candidate_pool_multiplier)
        )
        source = initial[0].source if initial else "unknown"
        return self._postprocessor.process(query, initial, top_k, source_rankings={source: initial})


class FusedKisCoarseRetrievalService:
    """KIS retrieval that fuses independent text-to-keyframe rankers with RRF."""

    def __init__(
        self,
        branches: Mapping[str, tuple[TextEncoder, Retriever]],
        fusion_config: WeightedRRFConfig,
        temporal_nms_enabled: bool,
        temporal_nms_window_sec: float,
        candidate_pool_multiplier: int,
        video_aggregation_method: AggregationMethod = "max",
        video_aggregation_top_m: int = 3,
    ) -> None:
        if len(branches) < 2:
            raise ValueError("Fused KIS retrieval needs at least two encoder branches")
        self._branches = dict(sorted(branches.items()))
        self._fusion_config = fusion_config
        self._postprocessor = KisCandidatePostprocessor(
            temporal_nms_enabled=temporal_nms_enabled,
            temporal_nms_window_sec=temporal_nms_window_sec,
            candidate_pool_multiplier=candidate_pool_multiplier,
            video_aggregation_method=video_aggregation_method,
            video_aggregation_top_m=video_aggregation_top_m,
        )

    def search(self, query: str, top_k: int) -> KisCoarseResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        pool_size = top_k * self._postprocessor.candidate_pool_multiplier
        rankings = {
            source: tuple(retriever.retrieve(text_encoder.encode_texts([query])[0], pool_size))
            for source, (text_encoder, retriever) in self._branches.items()
        }
        fused = weighted_reciprocal_rank_fusion(rankings, self._fusion_config)
        return self._postprocessor.process(query, fused, top_k, source_rankings=rankings)


class KisMultiSourceRetrievalService:
    """Fuses visual and auxiliary text ranking branches through the existing RRF core."""

    def __init__(
        self,
        branches: Mapping[str, QueryCandidateRetriever],
        fusion_config: WeightedRRFConfig,
        temporal_nms_enabled: bool,
        temporal_nms_window_sec: float,
        candidate_pool_multiplier: int,
        video_aggregation_method: AggregationMethod = "max",
        video_aggregation_top_m: int = 3,
    ) -> None:
        if len(branches) < 2:
            raise ValueError("Multi-source KIS retrieval needs at least two branches")
        sources = set(branches)
        if sources != set(fusion_config.weights):
            raise ValueError("Multi-source KIS branches must exactly match configured RRF weights")
        if any(branch.source != source for source, branch in branches.items()):
            raise ValueError("Multi-source KIS branch source keys must match branch source names")
        self._branches = dict(sorted(branches.items()))
        self._fusion_config = fusion_config
        self._postprocessor = KisCandidatePostprocessor(
            temporal_nms_enabled=temporal_nms_enabled,
            temporal_nms_window_sec=temporal_nms_window_sec,
            candidate_pool_multiplier=candidate_pool_multiplier,
            video_aggregation_method=video_aggregation_method,
            video_aggregation_top_m=video_aggregation_top_m,
        )

    def search(self, query: str, top_k: int) -> KisCoarseResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        pool_size = top_k * self._postprocessor.candidate_pool_multiplier
        rankings = {
            source: tuple(branch.retrieve(query, pool_size)) for source, branch in self._branches.items()
        }
        fused = weighted_reciprocal_rank_fusion(rankings, self._fusion_config)
        return self._postprocessor.process(query, fused, top_k, source_rankings=rankings)


class KisDenseRefinementService:
    """Composes a coarse KIS service with original-video frame refinement."""

    def __init__(self, coarse_service: KisCoarseSearcher, refiner: DenseFrameRefiner) -> None:
        self._coarse_service = coarse_service
        self._refiner = refiner

    def search(self, query: str, top_k: int) -> KisRefinementResult:
        coarse_result = self._coarse_service.search(query, top_k)
        coarse_candidates = tuple(
            sorted(
                coarse_result.candidates,
                key=lambda candidate: (candidate.rank, candidate.keyframe_uid),
            )
        )
        run = self._refiner.refine(query, coarse_candidates)
        refined_by_uid = {
            candidate.source_keyframe_uid: candidate
            for candidate in run.candidates
        }
        final_candidates = tuple(
            replace(
                refined_by_uid.get(coarse.keyframe_uid, RefinedFrameCandidate.from_coarse(coarse)),
                rank=rank,
            )
            for rank, coarse in enumerate(coarse_candidates, start=1)
        )
        return KisRefinementResult(
            coarse_result=coarse_result,
            candidates=final_candidates,
            failures=run.failures,
        )


def write_kis_debug(
    result: KisCoarseResult | KisRefinementResult,
    path: Path,
    metadata: dict[str, object],
) -> Path:
    payload = {"schema_version": "1.0", **result.as_dict(), "metadata": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, path)
    return path


def default_debug_path(outputs_root: Path, query: str) -> Path:
    query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return outputs_root / "retrieval_debug" / f"kis_{query_id}.json"
