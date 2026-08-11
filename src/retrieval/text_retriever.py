"""BM25-first OCR, ASR, and metadata retrieval mapped to stable keyframes."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from data.text_artifacts import ASRTranscriptRecord, MetadataTextRecord, OCRTextRecord
from domain.models import Candidate, CandidateSourceScore, KeyframeRecord


class TextRetrievalError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TextDocument:
    document_id: str
    text: str


@dataclass(frozen=True, slots=True)
class TextSearchHit:
    document_id: str
    text: str
    score: float


class BM25TextIndex:
    """Small deterministic BM25 index for normalized text artifacts."""

    def __init__(self, documents: Sequence[TextDocument], k1: float = 1.5, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise TextRetrievalError("BM25 k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise TextRetrievalError("BM25 b must be finite and in [0, 1]")
        if len({document.document_id for document in documents}) != len(documents):
            raise TextRetrievalError("BM25 document IDs must be unique")
        if any(not document.document_id or not document.text.strip() for document in documents):
            raise TextRetrievalError("BM25 documents need non-empty IDs and text")
        self._documents = {document.document_id: document for document in documents}
        self._tokens = {document.document_id: _tokenize(document.text) for document in documents}
        self._term_frequencies = {
            document_id: Counter(tokens) for document_id, tokens in self._tokens.items()
        }
        self._document_frequencies: Counter[str] = Counter(
            token for tokens in self._tokens.values() for token in set(tokens)
        )
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens.values()) / len(self._tokens)
            if self._tokens
            else 0.0
        )
        self._k1 = k1
        self._b = b

    def search(self, query: str, top_k: int) -> tuple[TextSearchHit, ...]:
        if top_k < 1:
            raise TextRetrievalError("top_k must be at least 1")
        query_tokens = _tokenize(query)
        if not query_tokens or not self._documents:
            return ()
        document_count = len(self._documents)
        scores: dict[str, float] = defaultdict(float)
        for token in query_tokens:
            document_frequency = self._document_frequencies.get(token, 0)
            if not document_frequency:
                continue
            inverse_frequency = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for document_id, frequencies in self._term_frequencies.items():
                term_frequency = frequencies.get(token, 0)
                if not term_frequency:
                    continue
                length = len(self._tokens[document_id])
                denominator = term_frequency + self._k1 * (
                    1 - self._b + self._b * length / self._average_length
                )
                scores[document_id] += inverse_frequency * term_frequency * (self._k1 + 1) / denominator
        ordered = sorted(scores.items(), key=lambda value: (-value[1], value[0]))
        return tuple(
            TextSearchHit(document_id=document_id, text=self._documents[document_id].text, score=score)
            for document_id, score in ordered[:top_k]
        )


class KeyframeCandidateMapper:
    """Maps text hits to verified keyframe/video/frame metadata without guessing."""

    def __init__(self, keyframes: Sequence[KeyframeRecord]) -> None:
        records = {record.keyframe_uid: record for record in keyframes}
        if len(records) != len(keyframes):
            raise TextRetrievalError("Keyframe manifest contains duplicate keyframe_uid values")
        mapped = {
            uid: record
            for uid, record in records.items()
            if record.original_frame_id is not None and record.timestamp_sec is not None
        }
        self._records = mapped
        by_video: dict[str, list[KeyframeRecord]] = defaultdict(list)
        for record in mapped.values():
            by_video[record.video_id].append(record)
        self._by_video = {
            video_id: tuple(sorted(items, key=lambda record: (record.timestamp_sec or 0.0, record.keyframe_uid)))
            for video_id, items in by_video.items()
        }

    def candidate_for_keyframe(
        self,
        keyframe_uid: str,
        score: float,
        rank: int,
        source: str,
        evidence_id: str,
        evidence_text: str,
    ) -> Candidate:
        record = self.keyframe_for_uid(keyframe_uid)
        return self._candidate(record, score, rank, source, evidence_id, evidence_text)

    def keyframe_for_uid(self, keyframe_uid: str) -> KeyframeRecord:
        record = self._records.get(keyframe_uid)
        if record is None:
            raise TextRetrievalError(
                f"Text artifact keyframe lacks a verified keyframe/frame mapping: {keyframe_uid}"
            )
        return record

    def require_video(self, video_id: str) -> None:
        if video_id not in self._by_video:
            raise TextRetrievalError(f"Text artifact video lacks any verified keyframe mapping: {video_id}")

    def keyframe_for_video_timestamp(self, video_id: str, timestamp_sec: float) -> KeyframeRecord:
        if not math.isfinite(timestamp_sec) or timestamp_sec < 0:
            raise TextRetrievalError("ASR timestamp must be finite and non-negative")
        return self._nearest_keyframe(video_id, timestamp_sec)

    def representative_keyframe_for_video(self, video_id: str) -> KeyframeRecord:
        self.require_video(video_id)
        return self._by_video[video_id][0]

    def candidate_for_video_timestamp(
        self,
        video_id: str,
        timestamp_sec: float,
        score: float,
        rank: int,
        source: str,
        evidence_id: str,
        evidence_text: str,
    ) -> Candidate:
        record = self.keyframe_for_video_timestamp(video_id, timestamp_sec)
        return self._candidate(record, score, rank, source, evidence_id, evidence_text)

    def candidate_for_video(
        self,
        video_id: str,
        score: float,
        rank: int,
        source: str,
        evidence_id: str,
        evidence_text: str,
    ) -> Candidate:
        record = self.representative_keyframe_for_video(video_id)
        return self._candidate(record, score, rank, source, evidence_id, evidence_text)

    def _nearest_keyframe(self, video_id: str, timestamp_sec: float) -> KeyframeRecord:
        self.require_video(video_id)
        records = self._by_video[video_id]
        return min(
            records,
            key=lambda record: (abs((record.timestamp_sec or 0.0) - timestamp_sec), record.keyframe_uid),
        )

    @staticmethod
    def _candidate(
        record: KeyframeRecord,
        score: float,
        rank: int,
        source: str,
        evidence_id: str,
        evidence_text: str,
    ) -> Candidate:
        if record.original_frame_id is None or record.timestamp_sec is None:
            raise TextRetrievalError(f"Keyframe {record.keyframe_uid} has no verified mapping")
        return Candidate(
            keyframe_uid=record.keyframe_uid,
            video_id=record.video_id,
            original_frame_id=record.original_frame_id,
            timestamp_sec=record.timestamp_sec,
            keyframe_path=record.keyframe_path,
            score=score,
            rank=rank,
            source=source,
            source_scores=(
                CandidateSourceScore(
                    source=source,
                    rank=rank,
                    score=score,
                    evidence_id=evidence_id,
                    evidence_text=evidence_text,
                ),
            ),
        )


class _BM25CandidateRetriever:
    source: str

    def __init__(
        self,
        source: str,
        documents: Mapping[str, TextDocument],
        resolve: Callable[[str], tuple[str, str, str]],
        mapper: KeyframeCandidateMapper,
    ) -> None:
        if not source:
            raise TextRetrievalError("Text retrieval source must be non-empty")
        self.source = source
        self._documents = dict(documents)
        self._index = BM25TextIndex(tuple(self._documents.values()))
        self._resolve = resolve
        self._mapper = mapper

    def retrieve(self, query: str, top_k: int) -> tuple[Candidate, ...]:
        if top_k < 1:
            raise TextRetrievalError("top_k must be at least 1")
        hits = self._index.search(query, max(len(self._documents), top_k))
        selected: dict[str, tuple[TextSearchHit, str, str]] = {}
        for hit in hits:
            candidate_key, evidence_id, evidence_text = self._resolve(hit.document_id)
            previous = selected.get(candidate_key)
            if previous is None or (-hit.score, hit.document_id) < (-previous[0].score, previous[0].document_id):
                selected[candidate_key] = (hit, evidence_id, evidence_text)
        ordered = sorted(selected.items(), key=lambda item: (-item[1][0].score, item[0]))[:top_k]
        return tuple(
            self._map_candidate(candidate_key, hit, rank, evidence_id, evidence_text)
            for rank, (candidate_key, (hit, evidence_id, evidence_text)) in enumerate(ordered, start=1)
        )

    def _map_candidate(
        self,
        candidate_key: str,
        hit: TextSearchHit,
        rank: int,
        evidence_id: str,
        evidence_text: str,
    ) -> Candidate:
        return self._mapper.candidate_for_keyframe(
            candidate_key,
            hit.score,
            rank,
            self.source,
            evidence_id,
            evidence_text,
        )


class OCRTextRetriever(_BM25CandidateRetriever):
    source = "ocr"

    def __init__(self, records: Sequence[OCRTextRecord], mapper: KeyframeCandidateMapper) -> None:
        for record in records:
            mapper.keyframe_for_uid(record.keyframe_uid)
        by_id = {
            record.record_id: TextDocument(document_id=record.record_id, text=record.text) for record in records
        }
        lookup = {record.record_id: record for record in records}
        super().__init__(
            self.source,
            by_id,
            lambda document_id: (
                lookup[document_id].keyframe_uid,
                lookup[document_id].record_id,
                lookup[document_id].text,
            ),
            mapper,
        )


class ASRTextRetriever(_BM25CandidateRetriever):
    source = "asr"

    def __init__(self, records: Sequence[ASRTranscriptRecord], mapper: KeyframeCandidateMapper) -> None:
        for record in records:
            mapper.require_video(record.video_id)
        by_id = {
            record.segment_id: TextDocument(document_id=record.segment_id, text=record.text) for record in records
        }
        self._lookup = {record.segment_id: record for record in records}
        super().__init__(
            self.source,
            by_id,
            self._resolve_segment,
            mapper,
        )

    def _resolve_segment(self, document_id: str) -> tuple[str, str, str]:
        record = self._lookup[document_id]
        keyframe = self._mapper.keyframe_for_video_timestamp(record.video_id, record.midpoint_sec)
        return keyframe.keyframe_uid, record.segment_id, record.text


class MetadataTextRetriever(_BM25CandidateRetriever):
    source = "metadata"

    def __init__(self, records: Sequence[MetadataTextRecord], mapper: KeyframeCandidateMapper) -> None:
        for record in records:
            mapper.require_video(record.video_id)
        by_id = {
            record.video_id: TextDocument(
                document_id=record.video_id,
                text=" ".join(f"{field}: {text}" for field, text in sorted(record.fields.items())),
            )
            for record in records
        }
        self._lookup = {record.video_id: record for record in records}
        super().__init__(self.source, by_id, self._resolve_video, mapper)

    def _resolve_video(self, document_id: str) -> tuple[str, str, str]:
        record = self._lookup[document_id]
        evidence_text = " ".join(f"{field}: {text}" for field, text in sorted(record.fields.items()))
        keyframe = self._mapper.representative_keyframe_for_video(record.video_id)
        return keyframe.keyframe_uid, record.video_id, evidence_text


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
