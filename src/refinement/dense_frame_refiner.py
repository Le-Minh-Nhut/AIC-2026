"""Two-pass original-video frame refinement using existing image-text encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from domain.models import Candidate, CandidateSourceScore, VideoRecord
from domain.protocols import ImageTextEncoder
from refinement.frame_sampler import FrameSampler, FrameSamplingError
from refinement.video_decoder import DecodedFrame, VideoDecodingError, VideoFrameDecoder
from retrieval.fusion import RankedScore, WeightedRRFConfig, weighted_reciprocal_rank_scores


class FrameScoringError(ValueError):
    pass


class RefinementError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FrameScoringBranch:
    source: str
    encoder: ImageTextEncoder


@dataclass(frozen=True, slots=True)
class ScoredVideoFrame:
    original_frame_id: int
    timestamp_sec: float
    score: float
    rank: int
    source: str
    source_scores: tuple[CandidateSourceScore, ...]


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    coarse_window_sec: float
    sparse_fps: float
    dense_window_sec: float
    candidate_count: int

    def __post_init__(self) -> None:
        if self.coarse_window_sec < 0 or not math.isfinite(self.coarse_window_sec):
            raise ValueError("coarse_window_sec must be finite and non-negative")
        if self.sparse_fps <= 0 or not math.isfinite(self.sparse_fps):
            raise ValueError("sparse_fps must be finite and positive")
        if self.dense_window_sec < 0 or not math.isfinite(self.dense_window_sec):
            raise ValueError("dense_window_sec must be finite and non-negative")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be at least 1")


@dataclass(frozen=True, slots=True)
class RefinedFrameCandidate:
    source_keyframe_uid: str
    video_id: str
    coarse_original_frame_id: int
    coarse_timestamp_sec: float
    coarse_score: float
    sparse_original_frame_id: int
    sparse_timestamp_sec: float
    sparse_score: float
    original_frame_id: int
    timestamp_sec: float
    score: float
    rank: int
    source: str
    source_scores: tuple[CandidateSourceScore, ...]
    refinement_status: str = "refined"

    @classmethod
    def from_coarse(cls, coarse: Candidate) -> "RefinedFrameCandidate":
        """Represent a coarse candidate safely when dense refinement is unavailable.

        The coarse score and frame remain the candidate's global retrieval
        evidence.  A caller can overlay a refined frame later without losing
        the original candidate's identity or rank coverage.
        """

        return cls(
            source_keyframe_uid=coarse.keyframe_uid,
            video_id=coarse.video_id,
            coarse_original_frame_id=coarse.original_frame_id,
            coarse_timestamp_sec=coarse.timestamp_sec,
            coarse_score=coarse.score,
            sparse_original_frame_id=coarse.original_frame_id,
            sparse_timestamp_sec=coarse.timestamp_sec,
            sparse_score=coarse.score,
            original_frame_id=coarse.original_frame_id,
            timestamp_sec=coarse.timestamp_sec,
            score=coarse.score,
            rank=coarse.rank,
            source=coarse.source,
            source_scores=coarse.source_scores,
            refinement_status="coarse_fallback",
        )


@dataclass(frozen=True, slots=True)
class RefinementFailure:
    source_keyframe_uid: str
    video_id: str
    coarse_original_frame_id: int
    coarse_score: float
    error: str


@dataclass(frozen=True, slots=True)
class DenseRefinementRun:
    candidates: tuple[RefinedFrameCandidate, ...]
    failures: tuple[RefinementFailure, ...]


class VisualFrameScorer:
    """Scores decoded images with one encoder or fuses independent model ranks with RRF."""

    def __init__(
        self,
        branches: Sequence[FrameScoringBranch],
        fusion_config: WeightedRRFConfig | None = None,
    ) -> None:
        if not branches:
            raise ValueError("At least one visual frame scoring branch is required")
        sources = [branch.source for branch in branches]
        if len(set(sources)) != len(sources) or any(not source for source in sources):
            raise ValueError("Frame scoring branch sources must be unique non-empty strings")
        if len(branches) > 1 and fusion_config is None:
            raise ValueError("Multiple frame scoring branches require an RRF fusion configuration")
        if len(branches) == 1 and fusion_config is not None:
            raise ValueError("Single frame scoring branch must not receive an RRF fusion configuration")
        self._branches = tuple(sorted(branches, key=lambda branch: branch.source))
        self._fusion_config = fusion_config

    def prepare(self, query: str) -> "PreparedVisualFrameScorer":
        if not query.strip():
            raise FrameScoringError("Refinement query must be non-empty")
        query_vectors: dict[str, np.ndarray] = {}
        for branch in self._branches:
            vector = np.asarray(branch.encoder.encode_texts([query]), dtype=np.float32)
            if vector.ndim != 2 or vector.shape[0] != 1 or not np.isfinite(vector).all():
                raise FrameScoringError(f"{branch.source} text encoder returned an invalid query vector")
            query_vectors[branch.source] = vector[0]
        return PreparedVisualFrameScorer(self._branches, query_vectors, self._fusion_config)


class PreparedVisualFrameScorer:
    def __init__(
        self,
        branches: Sequence[FrameScoringBranch],
        query_vectors: Mapping[str, np.ndarray],
        fusion_config: WeightedRRFConfig | None,
    ) -> None:
        self._branches = tuple(branches)
        self._query_vectors = dict(query_vectors)
        self._fusion_config = fusion_config

    def score(self, frames: Sequence[DecodedFrame]) -> tuple[ScoredVideoFrame, ...]:
        if not frames:
            raise FrameScoringError("At least one decoded frame is required for scoring")
        ordered_frames = tuple(sorted(frames, key=lambda frame: frame.frame_id))
        if len({frame.frame_id for frame in ordered_frames}) != len(ordered_frames):
            raise FrameScoringError("Decoded frames contain duplicate frame IDs")
        per_source = {
            branch.source: self._rank_branch(branch, ordered_frames, self._query_vectors[branch.source])
            for branch in self._branches
        }
        by_frame_id = {frame.frame_id: frame for frame in ordered_frames}
        if self._fusion_config is None:
            source = self._branches[0].source
            return tuple(
                ScoredVideoFrame(
                    original_frame_id=by_frame_id[item.frame_id].frame_id,
                    timestamp_sec=by_frame_id[item.frame_id].timestamp_sec,
                    score=item.score,
                    rank=item.rank,
                    source=source,
                    source_scores=(
                        CandidateSourceScore(source=source, rank=item.rank, score=item.score),
                    ),
                )
                for item in per_source[source]
            )
        fused = weighted_reciprocal_rank_scores(
            {
                source: tuple(
                    RankedScore(item_id=str(item.frame_id), rank=item.rank, score=item.score)
                    for item in items
                )
                for source, items in per_source.items()
            },
            self._fusion_config,
        )
        return tuple(
            ScoredVideoFrame(
                original_frame_id=int(item.item_id),
                timestamp_sec=by_frame_id[int(item.item_id)].timestamp_sec,
                score=item.score,
                rank=item.rank,
                source="rrf_fusion",
                source_scores=item.source_scores,
            )
            for item in fused
        )

    @dataclass(frozen=True, slots=True)
    class _BranchFrameScore:
        frame_id: int
        score: float
        rank: int

    def _rank_branch(
        self,
        branch: FrameScoringBranch,
        frames: Sequence[DecodedFrame],
        query_vector: np.ndarray,
    ) -> tuple[_BranchFrameScore, ...]:
        vectors = np.asarray(branch.encoder.encode_images([frame.image for frame in frames]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(frames):
            raise FrameScoringError(f"{branch.source} image encoder returned an invalid frame matrix")
        if vectors.shape[1] != query_vector.shape[0]:
            raise FrameScoringError(
                f"{branch.source} image/query embedding dimensions differ: "
                f"{vectors.shape[1]} != {query_vector.shape[0]}"
            )
        scores = vectors @ query_vector
        if not np.isfinite(scores).all():
            raise FrameScoringError(f"{branch.source} produced non-finite frame scores")
        ordered = sorted(
            zip(frames, scores, strict=True),
            key=lambda value: (-float(value[1]), value[0].frame_id),
        )
        return tuple(
            self._BranchFrameScore(frame_id=frame.frame_id, score=float(score), rank=rank)
            for rank, (frame, score) in enumerate(ordered, start=1)
        )


class DenseFrameRefiner:
    """Runs sparse then full-FPS scoring around each selected coarse keyframe."""

    def __init__(
        self,
        decoder: VideoFrameDecoder,
        sampler: FrameSampler,
        scorer: VisualFrameScorer,
        video_records: Sequence[VideoRecord],
        data_root: Path,
        config: RefinementConfig,
    ) -> None:
        records = {record.video_id: record for record in video_records}
        if len(records) != len(video_records):
            raise RefinementError("Video manifest contains duplicate video_id values")
        self._decoder = decoder
        self._sampler = sampler
        self._scorer = scorer
        self._records = records
        self._data_root = data_root
        self._config = config

    def refine(self, query: str, coarse_candidates: Sequence[Candidate]) -> DenseRefinementRun:
        selected = tuple(
            sorted(coarse_candidates, key=lambda candidate: (candidate.rank, candidate.keyframe_uid))
        )[
            : self._config.candidate_count
        ]
        try:
            prepared_scorer = self._scorer.prepare(query)
        except (FrameScoringError, OSError, ValueError) as error:
            return DenseRefinementRun(
                candidates=(),
                failures=tuple(
                    RefinementFailure(
                        source_keyframe_uid=coarse.keyframe_uid,
                        video_id=coarse.video_id,
                        coarse_original_frame_id=coarse.original_frame_id,
                        coarse_score=coarse.score,
                        error=str(error),
                    )
                    for coarse in selected
                ),
            )
        successes: list[RefinedFrameCandidate] = []
        failures: list[RefinementFailure] = []
        for coarse in selected:
            try:
                successes.append(self._refine_candidate(coarse, prepared_scorer))
            except (
                FrameSamplingError,
                FrameScoringError,
                RefinementError,
                VideoDecodingError,
                OSError,
                ValueError,
            ) as error:
                failures.append(
                    RefinementFailure(
                        source_keyframe_uid=coarse.keyframe_uid,
                        video_id=coarse.video_id,
                        coarse_original_frame_id=coarse.original_frame_id,
                        coarse_score=coarse.score,
                        error=str(error),
                    )
                )
        ordered = sorted(
            successes,
            key=lambda candidate: (
                -candidate.score,
                candidate.video_id,
                candidate.original_frame_id,
                candidate.source_keyframe_uid,
            ),
        )
        return DenseRefinementRun(
            candidates=tuple(replace(candidate, rank=rank) for rank, candidate in enumerate(ordered, start=1)),
            failures=tuple(failures),
        )

    def refine_candidate_frames(
        self,
        query: str,
        coarse_candidate: Candidate,
        top_k: int | None = None,
    ) -> tuple[RefinedFrameCandidate, ...]:
        """Return ranked dense-frame alternatives for one fixed coarse candidate.

        TRAKE uses these alternatives for a joint temporal alignment pass after
        independent visual scoring.  KIS continues to use :meth:`refine`, which
        keeps only the strongest alternative for each selected keyframe.
        """

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1 when provided")
        prepared_scorer = self._scorer.prepare(query)
        candidates = self._refine_candidate_frames(coarse_candidate, prepared_scorer)
        return candidates if top_k is None else candidates[:top_k]

    def _refine_candidate(
        self,
        coarse: Candidate,
        scorer: PreparedVisualFrameScorer,
    ) -> RefinedFrameCandidate:
        return self._refine_candidate_frames(coarse, scorer)[0]

    def _refine_candidate_frames(
        self,
        coarse: Candidate,
        scorer: PreparedVisualFrameScorer,
    ) -> tuple[RefinedFrameCandidate, ...]:
        video = self._records.get(coarse.video_id)
        if video is None:
            raise RefinementError(f"No video-manifest record exists for {coarse.video_id}")
        if not video.is_readable:
            raise RefinementError(f"Video manifest marks {coarse.video_id} unreadable: {video.probe_error}")
        video_path = self._data_root / video.video_path
        info = self._decoder.inspect(video_path)
        frame_count = info.frame_count if info.frame_count is not None else video.frame_count
        sparse_ids = self._sampler.sparse_frame_ids(
            center_frame_id=coarse.original_frame_id,
            fps=info.fps,
            window_sec=self._config.coarse_window_sec,
            sample_fps=self._config.sparse_fps,
            frame_count=frame_count,
        )
        sparse_frames = self._decoder.decode_frames(video_path, sparse_ids)
        sparse_best = scorer.score(sparse_frames)[0]
        dense_ids = self._sampler.dense_frame_ids(
            center_frame_id=sparse_best.original_frame_id,
            fps=info.fps,
            window_sec=self._config.dense_window_sec,
            frame_count=frame_count,
        )
        dense_frames = self._decoder.decode_frames(video_path, dense_ids)
        return tuple(
            RefinedFrameCandidate(
                source_keyframe_uid=coarse.keyframe_uid,
                video_id=coarse.video_id,
                coarse_original_frame_id=coarse.original_frame_id,
                coarse_timestamp_sec=coarse.timestamp_sec,
                coarse_score=coarse.score,
                sparse_original_frame_id=sparse_best.original_frame_id,
                sparse_timestamp_sec=sparse_best.timestamp_sec,
                sparse_score=sparse_best.score,
                original_frame_id=refined.original_frame_id,
                timestamp_sec=refined.timestamp_sec,
                score=refined.score,
                rank=refined.rank,
                source=refined.source,
                source_scores=refined.source_scores,
                refinement_status="refined",
            )
            for refined in scorer.score(dense_frames)
        )
