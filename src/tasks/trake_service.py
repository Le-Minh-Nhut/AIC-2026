"""End-to-end ordered-event TRAKE orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from query.event_decomposer import EventDecomposer, EventQuery
from tasks.kis_service import KisCoarseResult, KisCoarseSearcher
from trake.event_candidates import EventCandidate
from trake.event_refiner import TrakeDenseEventRefiner, TrakeRefinementFailure
from trake.temporal_aligner import TemporalAlignment, TemporalAligner, build_candidate_matrix
from trake.video_selector import CandidateVideoEvidence, CandidateVideoSelector


@dataclass(frozen=True, slots=True)
class TrakeServiceConfig:
    event_top_k: int
    candidate_videos: int
    k_best_sequences: int
    sequences_to_refine: int

    def __post_init__(self) -> None:
        values = {
            "event_top_k": self.event_top_k,
            "candidate_videos": self.candidate_videos,
            "k_best_sequences": self.k_best_sequences,
            "sequences_to_refine": self.sequences_to_refine,
        }
        invalid = [name for name, value in values.items() if value < 1]
        if invalid:
            raise ValueError("TRAKE configuration values must be at least 1: " + ", ".join(invalid))


@dataclass(frozen=True, slots=True)
class TrakeSequence:
    video_id: str
    coarse_alignment: TemporalAlignment
    refined_alignment: TemporalAlignment | None
    total_alignment_score: float
    rank: int = 0
    refinement_status: str = "not_selected"
    refinement_error: str | None = None

    @property
    def final_alignment(self) -> TemporalAlignment:
        return self.refined_alignment or self.coarse_alignment

    def as_dict(self) -> dict[str, object]:
        final = self.final_alignment
        coarse_by_event = {match.event.index: match for match in self.coarse_alignment.matches}
        refined_by_event = (
            {match.event.index: match for match in self.refined_alignment.matches}
            if self.refined_alignment is not None
            else {}
        )
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "ordered_frame_ids": list(final.frame_ids),
            "event_scores": list(final.event_scores),
            "total_alignment_score": self.total_alignment_score,
            "transition_penalty": final.transition_penalty,
            "refinement_status": self.refinement_status,
            "refinement_error": self.refinement_error,
            "events": [
                {
                    "event": coarse_by_event[event_index].event.as_dict(),
                    "coarse": coarse_by_event[event_index].as_dict(),
                    "refined": refined_by_event[event_index].as_dict()
                    if event_index in refined_by_event
                    else None,
                }
                for event_index in sorted(coarse_by_event)
            ],
            "coarse_alignment": self.coarse_alignment.as_dict(),
            "refined_alignment": self.refined_alignment.as_dict()
            if self.refined_alignment is not None
            else None,
        }


@dataclass(frozen=True, slots=True)
class TrakeResult:
    query: str
    events: tuple[EventQuery, ...]
    event_retrievals: Mapping[int, KisCoarseResult]
    candidate_videos: tuple[CandidateVideoEvidence, ...]
    coarse_alignments: tuple[TemporalAlignment, ...]
    candidates: tuple[TrakeSequence, ...]
    refinement_failures: tuple[TrakeRefinementFailure, ...]
    refinement_enabled: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "events": [event.as_dict() for event in self.events],
            "event_retrievals": {
                str(event_index): result.as_dict()
                for event_index, result in sorted(self.event_retrievals.items())
            },
            "candidate_videos": [candidate.as_dict() for candidate in self.candidate_videos],
            "coarse_alignments": [alignment.as_dict() for alignment in self.coarse_alignments],
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "refinement": {
                "enabled": self.refinement_enabled,
                "failure_count": len(self.refinement_failures),
                "failures": [failure.as_dict() for failure in self.refinement_failures],
            },
        }


class TrakeService:
    """Retrieve every event, align complete video sequences, then refine them jointly."""

    def __init__(
        self,
        event_decomposer: EventDecomposer,
        coarse_searcher: KisCoarseSearcher,
        video_selector: CandidateVideoSelector,
        temporal_aligner: TemporalAligner,
        config: TrakeServiceConfig,
        event_refiner: TrakeDenseEventRefiner | None = None,
    ) -> None:
        self._event_decomposer = event_decomposer
        self._coarse_searcher = coarse_searcher
        self._video_selector = video_selector
        self._temporal_aligner = temporal_aligner
        self._config = config
        self._event_refiner = event_refiner

    def search(self, query: str) -> TrakeResult:
        events = tuple(self._event_decomposer.decompose(query))
        if not events:
            raise ValueError("TRAKE event decomposition returned no events")
        event_retrievals = {
            event.index: self._coarse_searcher.search(event.retrieval_text, self._config.event_top_k)
            for event in events
        }
        event_candidates = tuple(
            EventCandidate(event=event, candidate=candidate)
            for event in events
            for candidate in event_retrievals[event.index].candidates
        )
        candidate_videos = self._video_selector.select(event_candidates, self._config.candidate_videos)
        coarse_alignments = self._coarse_align(events, event_candidates, candidate_videos)
        final_sequences, failures = self._refine(coarse_alignments)
        return TrakeResult(
            query=query,
            events=events,
            event_retrievals=event_retrievals,
            candidate_videos=candidate_videos,
            coarse_alignments=coarse_alignments,
            candidates=final_sequences,
            refinement_failures=failures,
            refinement_enabled=self._event_refiner is not None,
        )

    def _coarse_align(
        self,
        events: Sequence[EventQuery],
        event_candidates: Sequence[EventCandidate],
        candidate_videos: Sequence[CandidateVideoEvidence],
    ) -> tuple[TemporalAlignment, ...]:
        alignments = [
            alignment
            for video in candidate_videos
            for alignment in self._temporal_aligner.align(
                build_candidate_matrix(video.video_id, events, event_candidates)
            )
        ]
        deduplicated = self._temporal_aligner.deduplicate(alignments)
        limit = max(self._config.k_best_sequences, self._config.sequences_to_refine)
        return deduplicated[:limit]

    def _refine(
        self,
        coarse_alignments: Sequence[TemporalAlignment],
    ) -> tuple[tuple[TrakeSequence, ...], tuple[TrakeRefinementFailure, ...]]:
        sequences: list[TrakeSequence] = []
        failures: list[TrakeRefinementFailure] = []
        for sequence_index, coarse_alignment in enumerate(coarse_alignments):
            if self._event_refiner is None:
                sequences.append(
                    TrakeSequence(
                        video_id=coarse_alignment.video_id,
                        coarse_alignment=coarse_alignment,
                        refined_alignment=None,
                        total_alignment_score=coarse_alignment.total_score,
                        refinement_status="not_configured",
                    )
                )
                continue
            if sequence_index >= self._config.sequences_to_refine:
                sequences.append(
                    TrakeSequence(
                        video_id=coarse_alignment.video_id,
                        coarse_alignment=coarse_alignment,
                        refined_alignment=None,
                        total_alignment_score=coarse_alignment.total_score,
                        refinement_status="not_selected",
                    )
                )
                continue
            run = self._event_refiner.refine(coarse_alignment)
            failures.extend(run.failures)
            if run.alignments:
                sequences.extend(
                    TrakeSequence(
                        video_id=alignment.video_id,
                        coarse_alignment=coarse_alignment,
                        refined_alignment=alignment,
                        total_alignment_score=alignment.total_score,
                        refinement_status="refined",
                    )
                    for alignment in run.alignments
                )
                continue
            message = "; ".join(failure.error for failure in run.failures) or "Unknown refinement failure"
            sequences.append(
                TrakeSequence(
                    video_id=coarse_alignment.video_id,
                    coarse_alignment=coarse_alignment,
                    refined_alignment=None,
                    total_alignment_score=coarse_alignment.total_score,
                    refinement_status="failed",
                    refinement_error=message,
                )
            )
        return self._rank_sequences(sequences), tuple(failures)

    def _rank_sequences(self, sequences: Sequence[TrakeSequence]) -> tuple[TrakeSequence, ...]:
        kept: list[TrakeSequence] = []
        for sequence in sorted(
            sequences,
            key=lambda item: (
                -item.total_alignment_score,
                item.video_id,
                item.final_alignment.frame_ids,
            ),
        ):
            if any(
                self._temporal_aligner.are_near_duplicates(
                    sequence.final_alignment,
                    existing.final_alignment,
                )
                for existing in kept
            ):
                continue
            kept.append(sequence)
        return tuple(
            replace(sequence, rank=rank)
            for rank, sequence in enumerate(kept[: self._config.k_best_sequences], start=1)
        )


def write_trake_debug(result: TrakeResult, path: Path, metadata: dict[str, object]) -> Path:
    payload = {"schema_version": "1.0", **result.as_dict(), "metadata": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, path)
    return path


def default_trake_debug_path(outputs_root: Path, query: str) -> Path:
    query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return outputs_root / "retrieval_debug" / f"trake_{query_id}.json"
