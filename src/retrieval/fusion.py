"""Rank-based fusion for independent retrieval branches."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from domain.models import Candidate, CandidateSourceScore


class FusionValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WeightedRRFConfig:
    """Explicit, score-scale-independent configuration for weighted RRF."""

    k: int
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.k < 0:
            raise FusionValidationError("RRF k must be non-negative")
        if not self.weights:
            raise FusionValidationError("RRF needs at least one positive source weight")
        invalid = [
            source
            for source, weight in self.weights.items()
            if not source or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0
        ]
        if invalid:
            raise FusionValidationError(
                "RRF weights must be finite, positive values for named sources: " + ", ".join(sorted(invalid))
            )


@dataclass(frozen=True, slots=True)
class RankedScore:
    """One model's scored rank for a stable item identifier."""

    item_id: str
    rank: int
    score: float
    evidence_id: str | None = None
    evidence_text: str | None = None


@dataclass(frozen=True, slots=True)
class FusedRankedScore:
    """Weighted RRF result with the source evidence needed for debugging."""

    item_id: str
    score: float
    rank: int
    source_scores: tuple[CandidateSourceScore, ...]


def weighted_reciprocal_rank_scores(
    rankings: Mapping[str, Sequence[RankedScore]],
    config: WeightedRRFConfig,
) -> tuple[FusedRankedScore, ...]:
    """Fuse generic ranked items so keyframes and decoded frames share RRF logic."""

    _validate_sources(rankings, config)
    values: dict[str, tuple[list[CandidateSourceScore], float]] = {}
    for source in sorted(rankings):
        seen_item_ids: set[str] = set()
        weight = float(config.weights[source])
        for item in rankings[source]:
            if not item.item_id or item.item_id in seen_item_ids:
                raise FusionValidationError(f"RRF source {source} contains a duplicate or empty item ID")
            seen_item_ids.add(item.item_id)
            if item.rank < 1 or not math.isfinite(item.score):
                raise FusionValidationError(
                    f"RRF source {source} has invalid rank or score for {item.item_id}"
                )
            contribution = weight / (config.k + item.rank)
            breakdown = CandidateSourceScore(
                source=source,
                rank=item.rank,
                score=item.score,
                weight=weight,
                rrf_contribution=contribution,
                evidence_id=item.evidence_id,
                evidence_text=item.evidence_text,
            )
            source_scores, score = values.get(item.item_id, ([], 0.0))
            source_scores.append(breakdown)
            values[item.item_id] = (source_scores, score + contribution)
    ordered = sorted(values.items(), key=lambda value: (-value[1][1], value[0]))
    return tuple(
        FusedRankedScore(
            item_id=item_id,
            score=score,
            rank=rank,
            source_scores=tuple(sorted(source_scores, key=lambda item: item.source)),
        )
        for rank, (item_id, (source_scores, score)) in enumerate(ordered, start=1)
    )


def weighted_reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Candidate]],
    config: WeightedRRFConfig,
) -> tuple[Candidate, ...]:
    """Fuse rankings deterministically while retaining every contributing rank/score."""

    candidates_by_uid: dict[str, Candidate] = {}
    generic_rankings: dict[str, tuple[RankedScore, ...]] = {}
    for source in sorted(rankings):
        generic_rankings[source] = tuple(
            RankedScore(
                item_id=candidate.keyframe_uid,
                rank=candidate.rank,
                score=candidate.score,
                evidence_id=_source_evidence(candidate, source).evidence_id,
                evidence_text=_source_evidence(candidate, source).evidence_text,
            )
            for candidate in rankings[source]
        )
        for candidate in rankings[source]:
            reference = candidates_by_uid.get(candidate.keyframe_uid)
            if reference is not None:
                if _candidate_metadata(reference) != _candidate_metadata(candidate):
                    raise FusionValidationError(
                        f"RRF keyframe metadata differs across sources: {candidate.keyframe_uid}"
                    )
                continue
            candidates_by_uid[candidate.keyframe_uid] = candidate

    fused_scores = weighted_reciprocal_rank_scores(generic_rankings, config)
    return tuple(
        replace(
            candidates_by_uid[item.item_id],
            score=item.score,
            rank=item.rank,
            source="rrf_fusion",
            source_scores=item.source_scores,
        )
        for item in fused_scores
    )


def _validate_sources(rankings: Mapping[str, Sequence[object]], config: WeightedRRFConfig) -> None:
    sources = set(rankings)
    configured_sources = set(config.weights)
    if not sources:
        raise FusionValidationError("RRF needs at least one source ranking")
    if sources != configured_sources:
        missing = sorted(sources - configured_sources)
        extra = sorted(configured_sources - sources)
        details = []
        if missing:
            details.append("missing weights: " + ", ".join(missing))
        if extra:
            details.append("unused weights: " + ", ".join(extra))
        raise FusionValidationError("RRF source configuration mismatch (" + "; ".join(details) + ")")


def _candidate_metadata(candidate: Candidate) -> tuple[str, int, float, str]:
    return (
        candidate.video_id,
        candidate.original_frame_id,
        candidate.timestamp_sec,
        candidate.keyframe_path,
    )


def _source_evidence(candidate: Candidate, source: str) -> CandidateSourceScore:
    return next(
        (
            score
            for score in candidate.source_scores
            if score.source == source
        ),
        CandidateSourceScore(source=source, rank=candidate.rank, score=candidate.score),
    )
