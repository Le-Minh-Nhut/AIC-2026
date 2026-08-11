"""Candidate mapping from vector-search hits to stable keyframe metadata."""

from __future__ import annotations

from domain.models import Candidate, CandidateSourceScore, KeyframeRecord, SearchHit


def candidate_from_hit(hit: SearchHit, record: KeyframeRecord, rank: int, source: str) -> Candidate:
    if record.original_frame_id is None or record.timestamp_sec is None:
        raise ValueError(
            f"Keyframe {record.keyframe_uid} lacks verified original-frame/timestamp mapping"
        )
    return Candidate(
        keyframe_uid=record.keyframe_uid,
        video_id=record.video_id,
        original_frame_id=record.original_frame_id,
        timestamp_sec=record.timestamp_sec,
        keyframe_path=record.keyframe_path,
        score=hit.score,
        rank=rank,
        source=source,
        source_scores=(CandidateSourceScore(source=source, rank=rank, score=hit.score),),
    )
