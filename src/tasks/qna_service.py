"""End-to-end multi-frame Q&A orchestration over reusable KIS retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from domain.models import Candidate, CandidateSourceScore
from domain.protocols import VisualAnswerer
from qna.answer_normalizer import AnswerNormalizer
from qna.frame_selector import CandidateClipSelector, ClipSelectionError, SampledClip
from refinement.dense_frame_refiner import RefinedFrameCandidate
from refinement.video_decoder import VideoDecodingError
from tasks.kis_service import KisCoarseResult, KisRefinementResult


class QnaServiceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QnAQuery:
    event_description: str
    question: str
    query_id: str | None = None

    def __post_init__(self) -> None:
        if not self.event_description.strip():
            raise QnaServiceError("Q&A event_description must be non-empty")
        if not self.question.strip():
            raise QnaServiceError("Q&A question must be non-empty")


@dataclass(frozen=True, slots=True)
class QnaServiceConfig:
    retrieval_candidate_count: int
    answer_candidate_count: int

    def __post_init__(self) -> None:
        if self.retrieval_candidate_count < 1:
            raise QnaServiceError("retrieval_candidate_count must be at least 1")
        if self.answer_candidate_count < 1:
            raise QnaServiceError("answer_candidate_count must be at least 1")
        if self.answer_candidate_count > self.retrieval_candidate_count:
            raise QnaServiceError("answer_candidate_count cannot exceed retrieval_candidate_count")


class QnaSearcher(Protocol):
    def search(self, query: str, top_k: int) -> KisCoarseResult | KisRefinementResult: ...


@dataclass(frozen=True, slots=True)
class _AnswerTarget:
    source_keyframe_uid: str
    video_id: str
    frame_id: int
    timestamp_sec: float
    retrieval_score: float
    refinement_score: float | None
    rank: int
    source: str
    source_scores: tuple[CandidateSourceScore, ...]
    coarse_frame_id: int | None


@dataclass(frozen=True, slots=True)
class QnaCandidate:
    video_id: str
    frame_id: int
    timestamp_sec: float
    raw_answer: str
    normalized_answer: str
    retrieval_score: float
    refinement_score: float | None
    rank: int
    source: str
    source_scores: tuple[CandidateSourceScore, ...]
    source_keyframe_uid: str
    coarse_frame_id: int | None
    clip: SampledClip

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "raw_answer": self.raw_answer,
            "normalized_answer": self.normalized_answer,
            "retrieval_score": self.retrieval_score,
            "refinement_score": self.refinement_score,
            "source": self.source,
            "source_scores": [asdict(score) for score in self.source_scores],
            "source_keyframe_uid": self.source_keyframe_uid,
            "coarse_frame_id": self.coarse_frame_id,
            "debug_candidate_frames": {
                "anchor_frame_id": self.clip.anchor_frame_id,
                "anchor_timestamp_sec": self.clip.anchor_timestamp_sec,
                "frame_ids": list(self.clip.frame_ids),
                "timestamps_sec": list(self.clip.timestamps_sec),
            },
        }


@dataclass(frozen=True, slots=True)
class QnaFailure:
    rank: int
    video_id: str
    frame_id: int
    stage: str
    error: str


@dataclass(frozen=True, slots=True)
class QnaResult:
    query: QnAQuery
    candidates: tuple[QnaCandidate, ...]
    retrieval: KisCoarseResult | KisRefinementResult
    failures: tuple[QnaFailure, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "query": asdict(self.query),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "retrieval": self.retrieval.as_dict(),
            "failure_count": len(self.failures),
            "failures": [asdict(failure) for failure in self.failures],
        }


class QnaService:
    """Retrieve event frames, sample temporal context, then ask an injected VLM."""

    def __init__(
        self,
        searcher: QnaSearcher,
        clip_selector: CandidateClipSelector,
        answerer: VisualAnswerer,
        answer_normalizer: AnswerNormalizer,
        config: QnaServiceConfig,
    ) -> None:
        self._searcher = searcher
        self._clip_selector = clip_selector
        self._answerer = answerer
        self._answer_normalizer = answer_normalizer
        self._config = config

    def answer(self, query: QnAQuery) -> QnaResult:
        """Use event_description only for retrieval; the question is VLM-only context."""

        retrieval = self._searcher.search(
            query.event_description,
            top_k=self._config.retrieval_candidate_count,
        )
        targets = self._targets(retrieval)[: self._config.answer_candidate_count]
        candidates: list[QnaCandidate] = []
        failures: list[QnaFailure] = []
        for target in targets:
            try:
                clip = self._clip_selector.select(target.video_id, target.frame_id)
            except (ClipSelectionError, VideoDecodingError, OSError, ValueError) as error:
                failures.append(self._failure(target, "clip_selection", error))
                continue
            try:
                raw_answer = self._answerer.answer(
                    [frame.image for frame in clip.frames],
                    query.event_description,
                    query.question,
                )
            except Exception as error:
                failures.append(self._failure(target, "vlm_answer", error))
                continue
            try:
                normalized_answer = self._answer_normalizer.normalize(raw_answer)
            except ValueError as error:
                failures.append(self._failure(target, "answer_normalization", error))
                continue
            candidates.append(
                QnaCandidate(
                    video_id=target.video_id,
                    frame_id=target.frame_id,
                    timestamp_sec=target.timestamp_sec,
                    raw_answer=raw_answer,
                    normalized_answer=normalized_answer,
                    retrieval_score=target.retrieval_score,
                    refinement_score=target.refinement_score,
                    rank=target.rank,
                    source=target.source,
                    source_scores=target.source_scores,
                    source_keyframe_uid=target.source_keyframe_uid,
                    coarse_frame_id=target.coarse_frame_id,
                    clip=clip,
                )
            )
        return QnaResult(
            query=query,
            candidates=tuple(replace(candidate, rank=rank) for rank, candidate in enumerate(candidates, start=1)),
            retrieval=retrieval,
            failures=tuple(failures),
        )

    @staticmethod
    def _targets(retrieval: KisCoarseResult | KisRefinementResult) -> tuple[_AnswerTarget, ...]:
        if isinstance(retrieval, KisRefinementResult):
            return tuple(_target_from_refined(candidate) for candidate in retrieval.candidates)
        return tuple(_target_from_coarse(candidate) for candidate in retrieval.candidates)

    @staticmethod
    def _failure(target: _AnswerTarget, stage: str, error: Exception) -> QnaFailure:
        return QnaFailure(
            rank=target.rank,
            video_id=target.video_id,
            frame_id=target.frame_id,
            stage=stage,
            error=str(error),
        )


def _target_from_coarse(candidate: Candidate) -> _AnswerTarget:
    return _AnswerTarget(
        source_keyframe_uid=candidate.keyframe_uid,
        video_id=candidate.video_id,
        frame_id=candidate.original_frame_id,
        timestamp_sec=candidate.timestamp_sec,
        retrieval_score=candidate.score,
        refinement_score=None,
        rank=candidate.rank,
        source=candidate.source,
        source_scores=candidate.source_scores,
        coarse_frame_id=None,
    )


def _target_from_refined(candidate: RefinedFrameCandidate) -> _AnswerTarget:
    return _AnswerTarget(
        source_keyframe_uid=candidate.source_keyframe_uid,
        video_id=candidate.video_id,
        frame_id=candidate.original_frame_id,
        timestamp_sec=candidate.timestamp_sec,
        retrieval_score=candidate.coarse_score,
        refinement_score=(candidate.score if candidate.refinement_status == "refined" else None),
        rank=candidate.rank,
        source=candidate.source,
        source_scores=candidate.source_scores,
        coarse_frame_id=candidate.coarse_original_frame_id,
    )


def write_qna_debug(result: QnaResult, path: Path, metadata: dict[str, object]) -> Path:
    payload = {"schema_version": "1.0", **result.as_dict(), "metadata": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, path)
    return path


def default_qna_debug_path(outputs_root: Path, query: QnAQuery) -> Path:
    query_id = query.query_id or hashlib.sha256(
        f"{query.event_description}\n{query.question}".encode("utf-8")
    ).hexdigest()[:16]
    return outputs_root / "retrieval_debug" / f"qna_{query_id}.json"
