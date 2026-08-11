"""Atomic JSON submission I/O and conversion from KIS/Q&A/TRAKE debug artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from domain.competition import CompetitionContractError, SubmissionCandidate, SubmissionQuery, TaskType
from submission.ranker import (
    FrameDiversityConfig,
    RankedFrame,
    RankedSequence,
    SequenceDiversityConfig,
    diversify_ranked_frames,
    diversify_ranked_sequences,
)


SUBMISSION_SCHEMA_VERSION = "1.0"


class SubmissionFormatError(ValueError):
    pass


def write_submission(path: Path, queries: Sequence[SubmissionQuery]) -> Path:
    if len({query.query_id for query in queries}) != len(queries):
        raise SubmissionFormatError("Submission query_id values must be unique")
    if any(len(query.candidates) > 100 for query in queries):
        raise SubmissionFormatError("Submission has more than 100 results for a query")
    payload = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "queries": [submission_query_to_dict(query) for query in queries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_submission(path: Path) -> tuple[SubmissionQuery, ...]:
    value = _read_json_object(path, "submission")
    if value.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        raise SubmissionFormatError("Unsupported submission schema_version")
    raw_queries = value.get("queries")
    if not isinstance(raw_queries, list):
        raise SubmissionFormatError("Submission queries must be a list")
    try:
        queries = tuple(_query_from_dict(item) for item in raw_queries)
    except CompetitionContractError as error:
        raise SubmissionFormatError(str(error)) from error
    if len({query.query_id for query in queries}) != len(queries):
        raise SubmissionFormatError("Submission query_id values must be unique")
    return queries


def submission_from_debug(
    task: TaskType,
    query_id: str,
    debug_payload: Mapping[str, object],
    frame_config: FrameDiversityConfig | None = None,
    sequence_config: SequenceDiversityConfig | None = None,
) -> SubmissionQuery:
    raw_candidates = debug_payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise SubmissionFormatError("Debug artifact candidates must be a list")
    if not raw_candidates:
        raise SubmissionFormatError("Debug artifact has no candidates to submit")
    if task is TaskType.TRAKE:
        ranked = tuple(_sequence_from_debug(value) for value in raw_candidates)
        selected = diversify_ranked_sequences(ranked, sequence_config or SequenceDiversityConfig())
        return SubmissionQuery(
            query_id=query_id,
            task=task,
            candidates=tuple(SubmissionCandidate(item.video_id, item.frame_ids) for item in selected),
        )
    ranked_frames = tuple(_frame_from_debug(value, task) for value in raw_candidates)
    selected_frames = diversify_ranked_frames(ranked_frames, frame_config or FrameDiversityConfig())
    return SubmissionQuery(
        query_id=query_id,
        task=task,
        candidates=tuple(
            SubmissionCandidate(
                video_id=item.video_id,
                frame_ids=(item.frame_id,),
                answer=item.answer if task is TaskType.QNA else None,
            )
            for item in selected_frames
        ),
    )


def submission_query_to_dict(query: SubmissionQuery) -> dict[str, object]:
    """Serialize one already-validated query for UI preview or JSON submission output."""

    return {
        "query_id": query.query_id,
        "task": query.task.value,
        "results": [_candidate_to_dict(query.task, candidate) for candidate in query.candidates],
    }


def _candidate_to_dict(task: TaskType, candidate: SubmissionCandidate) -> dict[str, object]:
    if task is TaskType.KIS:
        return {"video_id": candidate.video_id, "frame_id": candidate.frame_ids[0]}
    if task is TaskType.QNA:
        return {
            "video_id": candidate.video_id,
            "frame_id": candidate.frame_ids[0],
            "answer": candidate.answer,
        }
    return {"video_id": candidate.video_id, "frame_ids": list(candidate.frame_ids)}


def _query_from_dict(value: object) -> SubmissionQuery:
    if not isinstance(value, dict):
        raise SubmissionFormatError("Submission query must be an object")
    try:
        task = TaskType(str(value.get("task")))
    except ValueError as error:
        raise SubmissionFormatError("Submission task must be kis, qna, or trake") from error
    query_id = str(value.get("query_id") or "").strip()
    raw_results = value.get("results")
    if not isinstance(raw_results, list):
        raise SubmissionFormatError("Submission results must be a list")
    if set(value) != {"query_id", "task", "results"}:
        raise SubmissionFormatError("Submission query must contain only query_id, task, and results")
    return SubmissionQuery(
        query_id=query_id,
        task=task,
        candidates=tuple(_candidate_from_dict(task, result) for result in raw_results),
    )


def _candidate_from_dict(task: TaskType, value: object) -> SubmissionCandidate:
    if not isinstance(value, dict):
        raise SubmissionFormatError("Submission result must be an object")
    video_id = str(value.get("video_id") or "").strip()
    if task in {TaskType.KIS, TaskType.QNA}:
        expected_fields = {"video_id", "frame_id"} | ({"answer"} if task is TaskType.QNA else set())
        if set(value) != expected_fields:
            raise SubmissionFormatError(
                f"{task.value.upper()} result fields must be: " + ", ".join(sorted(expected_fields))
            )
        frame_id = value.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise SubmissionFormatError("KIS/Q&A frame_id must be an integer")
        answer = str(value.get("answer") or "").strip() if task is TaskType.QNA else None
        return SubmissionCandidate(video_id=video_id, frame_ids=(frame_id,), answer=answer)
    if set(value) != {"video_id", "frame_ids"}:
        raise SubmissionFormatError("TRAKE result fields must be: frame_ids, video_id")
    frame_ids = value.get("frame_ids")
    if not isinstance(frame_ids, list) or any(isinstance(frame_id, bool) or not isinstance(frame_id, int) for frame_id in frame_ids):
        raise SubmissionFormatError("TRAKE frame_ids must be an integer list")
    return SubmissionCandidate(video_id=video_id, frame_ids=tuple(frame_ids))


def _frame_from_debug(value: object, task: TaskType) -> RankedFrame:
    if not isinstance(value, dict):
        raise SubmissionFormatError("Debug candidate must be an object")
    try:
        video_id = str(value["video_id"])
        frame_id = int(value["frame_id"])
        timestamp_sec = float(value.get("timestamp_sec", frame_id))
        score = float(value.get("score", value.get("retrieval_score")))
    except (KeyError, TypeError, ValueError) as error:
        raise SubmissionFormatError("Debug frame candidate needs video_id, frame_id, timestamp_sec, and score") from error
    answer = None
    if task is TaskType.QNA:
        answer = str(value.get("normalized_answer") or "").strip()
        if not answer:
            raise SubmissionFormatError("Q&A debug candidate needs normalized_answer")
    return RankedFrame(video_id, frame_id, timestamp_sec, score, answer)


def _sequence_from_debug(value: object) -> RankedSequence:
    if not isinstance(value, dict):
        raise SubmissionFormatError("TRAKE debug candidate must be an object")
    try:
        video_id = str(value["video_id"])
        frame_ids = tuple(int(frame_id) for frame_id in value["ordered_frame_ids"])
        score = float(value["total_alignment_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise SubmissionFormatError(
            "TRAKE debug candidate needs video_id, ordered_frame_ids, and total_alignment_score"
        ) from error
    return RankedSequence(video_id, frame_ids, score)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label.capitalize()} file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubmissionFormatError(f"Invalid {label} JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SubmissionFormatError(f"{label.capitalize()} root must be an object")
    return value
