"""Joint TRAKE dense refinement that preserves ordered event identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from refinement.dense_frame_refiner import (
    DenseFrameRefiner,
    FrameScoringError,
    RefinementError,
)
from refinement.frame_sampler import FrameSamplingError
from refinement.video_decoder import VideoDecodingError
from trake.event_candidates import EventCandidate, RefinedEventCandidate
from trake.temporal_aligner import TemporalAlignment, TemporalAligner, build_candidate_matrix


@dataclass(frozen=True, slots=True)
class TrakeRefinementFailure:
    video_id: str
    event_index: int | None
    event_text: str | None
    coarse_frame_id: int | None
    error: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrakeRefinementRun:
    alignments: tuple[TemporalAlignment, ...]
    failures: tuple[TrakeRefinementFailure, ...]


class TrakeDenseEventRefiner:
    """Refine every event independently, then rerun local joint temporal DP."""

    def __init__(
        self,
        frame_refiner: DenseFrameRefiner,
        temporal_aligner: TemporalAligner,
        local_frame_candidates: int,
    ) -> None:
        if local_frame_candidates < 1:
            raise ValueError("local_frame_candidates must be at least 1")
        self._frame_refiner = frame_refiner
        self._temporal_aligner = temporal_aligner
        self._local_frame_candidates = local_frame_candidates

    def refine(self, coarse_alignment: TemporalAlignment) -> TrakeRefinementRun:
        refined_candidates: list[RefinedEventCandidate] = []
        failures: list[TrakeRefinementFailure] = []
        for match in coarse_alignment.matches:
            if not isinstance(match, EventCandidate):
                raise ValueError("TRAKE dense refinement requires coarse EventCandidate matches")
            try:
                refined_frames = self._frame_refiner.refine_candidate_frames(
                    match.event.retrieval_text,
                    match.candidate,
                    top_k=self._local_frame_candidates,
                )
            except (
                FrameSamplingError,
                FrameScoringError,
                RefinementError,
                VideoDecodingError,
                OSError,
                ValueError,
            ) as error:
                failures.append(
                    TrakeRefinementFailure(
                        video_id=match.video_id,
                        event_index=match.event.index,
                        event_text=match.event.text,
                        coarse_frame_id=match.original_frame_id,
                        error=str(error),
                    )
                )
                continue
            refined_candidates.extend(
                RefinedEventCandidate(
                    event=match.event,
                    coarse_candidate=match.candidate,
                    refined_candidate=refined_frame,
                )
                for refined_frame in refined_frames
            )
        if failures:
            return TrakeRefinementRun(alignments=(), failures=tuple(failures))
        matrix = build_candidate_matrix(
            video_id=coarse_alignment.video_id,
            events=tuple(match.event for match in coarse_alignment.matches),
            candidates=refined_candidates,
        )
        alignments = self._temporal_aligner.align(matrix)
        if alignments:
            return TrakeRefinementRun(alignments=alignments, failures=())
        return TrakeRefinementRun(
            alignments=(),
            failures=(
                TrakeRefinementFailure(
                    video_id=coarse_alignment.video_id,
                    event_index=None,
                    event_text=None,
                    coarse_frame_id=None,
                    error="No monotonic dense-frame sequence satisfies the TRAKE temporal constraints",
                ),
            ),
        )
