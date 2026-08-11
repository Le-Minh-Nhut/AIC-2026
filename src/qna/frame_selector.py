"""Original-video multi-frame clip sampling for visual question answering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from domain.models import VideoRecord
from refinement.frame_sampler import FrameSampler, FrameSamplingError
from refinement.video_decoder import DecodedFrame, VideoDecodingError, VideoFrameDecoder


class ClipSelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClipSelectorConfig:
    window_sec: float
    multi_frame_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_sec) or self.window_sec < 0:
            raise ClipSelectionError("Q&A clip window_sec must be finite and non-negative")
        if self.multi_frame_count < 1:
            raise ClipSelectionError("Q&A multi_frame_count must be at least 1")


@dataclass(frozen=True, slots=True)
class SampledClip:
    video_id: str
    anchor_frame_id: int
    anchor_timestamp_sec: float
    frames: tuple[DecodedFrame, ...]

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(frame.frame_id for frame in self.frames)

    @property
    def timestamps_sec(self) -> tuple[float, ...]:
        return tuple(frame.timestamp_sec for frame in self.frames)


class CandidateClipSelector:
    """Decodes a chronologically ordered original-video clip around one event frame."""

    def __init__(
        self,
        decoder: VideoFrameDecoder,
        sampler: FrameSampler,
        video_records: Sequence[VideoRecord],
        data_root: Path,
        config: ClipSelectorConfig,
    ) -> None:
        records = {record.video_id: record for record in video_records}
        if len(records) != len(video_records):
            raise ClipSelectionError("Video manifest contains duplicate video_id values")
        self._decoder = decoder
        self._sampler = sampler
        self._records = records
        self._data_root = data_root
        self._config = config

    def select(self, video_id: str, anchor_frame_id: int) -> SampledClip:
        video = self._records.get(video_id)
        if video is None:
            raise ClipSelectionError(f"No video-manifest record exists for {video_id}")
        if not video.is_readable:
            raise ClipSelectionError(f"Video manifest marks {video_id} unreadable: {video.probe_error}")
        video_path = self._data_root / video.video_path
        info = self._decoder.inspect(video_path)
        frame_count = info.frame_count if info.frame_count is not None else video.frame_count
        frame_ids = self._sampler.evenly_spaced_frame_ids(
            center_frame_id=anchor_frame_id,
            fps=info.fps,
            window_sec=self._config.window_sec,
            frame_count=frame_count,
            sample_count=self._config.multi_frame_count,
        )
        frames = self._decoder.decode_frames(video_path, frame_ids)
        self._validate_frames(video_id, frame_ids, frames)
        return SampledClip(
            video_id=video_id,
            anchor_frame_id=anchor_frame_id,
            anchor_timestamp_sec=anchor_frame_id / info.fps,
            frames=frames,
        )

    @staticmethod
    def _validate_frames(
        video_id: str,
        requested_ids: tuple[int, ...],
        frames: Sequence[DecodedFrame],
    ) -> None:
        actual_ids = tuple(frame.frame_id for frame in frames)
        if actual_ids != requested_ids:
            raise ClipSelectionError(
                f"Q&A decoder returned unexpected frames for {video_id}: "
                f"requested {requested_ids}, received {actual_ids}"
            )
        if any(left >= right for left, right in zip(actual_ids, actual_ids[1:])):
            raise ClipSelectionError("Q&A clip frames must be strictly chronological")


__all__ = [
    "CandidateClipSelector",
    "ClipSelectionError",
    "ClipSelectorConfig",
    "SampledClip",
    "FrameSamplingError",
    "VideoDecodingError",
]
