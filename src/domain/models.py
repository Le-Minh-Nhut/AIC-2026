"""Typed contracts shared by downloader, manifests, and analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ArchiveCategory(StrEnum):
    KEYFRAMES = "keyframes"
    VIDEOS = "videos"
    BTC_CLIP_FEATURES = "btc_clip_features"
    MAP_KEYFRAMES = "map_keyframes"
    MEDIA_INFO = "media_info"
    OBJECTS = "objects"
    UNKNOWN = "unknown"

    @property
    def is_support(self) -> bool:
        return self in {
            ArchiveCategory.BTC_CLIP_FEATURES,
            ArchiveCategory.MAP_KEYFRAMES,
            ArchiveCategory.MEDIA_INFO,
            ArchiveCategory.OBJECTS,
        }


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    is_zip: bool
    zip_valid: bool
    entry_count: int | None
    compressed_size: int | None
    uncompressed_size: int | None
    error: str | None = None


@dataclass(slots=True)
class ArchiveRecord:
    filename: str
    download_url: str
    category: ArchiveCategory
    downloaded: bool = False
    archive_path: str | None = None
    file_size_bytes: int | None = None
    sha256_local: str | None = None
    zip_valid: bool | None = None
    entry_count: int | None = None
    compressed_size: int | None = None
    estimated_uncompressed_size: int | None = None
    extracted: bool = False
    status: str = "listed"
    error: str | None = None
    updated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["category"] = self.category.value
        return record

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArchiveRecord":
        copied = dict(value)
        copied["category"] = ArchiveCategory(copied["category"])
        return cls(**copied)


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    video_path: str
    group_id: str | None
    fps: float | None
    frame_count: int | None
    duration_sec: float | None
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    has_audio: bool
    container: str | None
    file_size_bytes: int
    is_readable: bool
    bitrate: int | None = None
    probe_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KeyframeRecord:
    keyframe_uid: str
    video_id: str
    keyframe_index: int | None
    keyframe_path: str
    original_frame_id: int | None
    timestamp_sec: float | None
    width: int | None
    height: int | None
    file_size_bytes: int
    is_readable: bool
    has_mapping: bool
    image_mode: str | None = None
    read_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MappingRecord:
    video_id: str
    keyframe_index: int
    original_frame_id: int
    source_path: str


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    affected_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "affected_ids": list(self.affected_ids),
        }


@dataclass(slots=True)
class MappingValidationReport:
    matched_count: int = 0
    missing_mapping_count: int = 0
    missing_image_count: int = 0
    unknown_video_count: int = 0
    invalid_frame_count: int = 0
    duplicate_keyframe_index_count: int = 0
    duplicate_frame_mapping_count: int = 0
    non_monotonic_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched_count": self.matched_count,
            "missing_mapping_count": self.missing_mapping_count,
            "missing_image_count": self.missing_image_count,
            "unknown_video_count": self.unknown_video_count,
            "invalid_frame_count": self.invalid_frame_count,
            "duplicate_keyframe_index_count": self.duplicate_keyframe_index_count,
            "duplicate_frame_mapping_count": self.duplicate_frame_mapping_count,
            "non_monotonic_count": self.non_monotonic_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    query_id: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSourceScore:
    """One retriever's original rank/score contribution for a keyframe."""

    source: str
    rank: int
    score: float
    weight: float | None = None
    rrf_contribution: float | None = None
    evidence_id: str | None = None
    evidence_text: str | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    keyframe_uid: str
    video_id: str
    original_frame_id: int
    timestamp_sec: float
    keyframe_path: str
    score: float
    rank: int
    source: str
    source_scores: tuple[CandidateSourceScore, ...] = ()


@dataclass(frozen=True, slots=True)
class VideoCandidate:
    video_id: str
    score: float
    best_keyframe_uid: str
    contributing_keyframes: int


def as_posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
