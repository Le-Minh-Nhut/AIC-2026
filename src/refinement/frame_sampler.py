"""Deterministic sparse and dense frame-ID sampling around an original frame."""

from __future__ import annotations

import math
from dataclasses import dataclass


class FrameSamplingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FrameWindow:
    start_frame_id: int
    end_frame_id: int
    center_frame_id: int


class FrameSampler:
    def sparse_frame_ids(
        self,
        center_frame_id: int,
        fps: float,
        window_sec: float,
        sample_fps: float,
        frame_count: int | None,
    ) -> tuple[int, ...]:
        if sample_fps <= 0 or not math.isfinite(sample_fps):
            raise FrameSamplingError("Sparse sampling FPS must be a finite positive value")
        window = self.window(center_frame_id, fps, window_sec, frame_count)
        stride = max(1, int(round(fps / sample_fps)))
        frame_ids = set(range(window.start_frame_id, window.end_frame_id + 1, stride))
        frame_ids.update((window.start_frame_id, window.center_frame_id, window.end_frame_id))
        return tuple(sorted(frame_ids))

    def dense_frame_ids(
        self,
        center_frame_id: int,
        fps: float,
        window_sec: float,
        frame_count: int | None,
    ) -> tuple[int, ...]:
        window = self.window(center_frame_id, fps, window_sec, frame_count)
        return tuple(range(window.start_frame_id, window.end_frame_id + 1))

    def evenly_spaced_frame_ids(
        self,
        center_frame_id: int,
        fps: float,
        window_sec: float,
        frame_count: int | None,
        sample_count: int,
    ) -> tuple[int, ...]:
        """Select ordered unique frames around an anchor for temporal VLM context."""

        if sample_count < 1:
            raise FrameSamplingError("Sample count must be at least 1")
        window = self.window(center_frame_id, fps, window_sec, frame_count)
        available_count = window.end_frame_id - window.start_frame_id + 1
        if sample_count >= available_count:
            return tuple(range(window.start_frame_id, window.end_frame_id + 1))
        if sample_count == 1:
            return (window.center_frame_id,)
        span = window.end_frame_id - window.start_frame_id
        frame_ids = [
            int(round(window.start_frame_id + index * span / (sample_count - 1)))
            for index in range(sample_count)
        ]
        closest_index = min(
            range(len(frame_ids)),
            key=lambda index: (abs(frame_ids[index] - window.center_frame_id), index),
        )
        frame_ids[closest_index] = window.center_frame_id
        selected = set(frame_ids)
        if len(selected) < sample_count:
            for frame_id in range(window.start_frame_id, window.end_frame_id + 1):
                selected.add(frame_id)
                if len(selected) == sample_count:
                    break
        return tuple(sorted(selected))

    @staticmethod
    def window(
        center_frame_id: int,
        fps: float,
        window_sec: float,
        frame_count: int | None,
    ) -> FrameWindow:
        if not isinstance(center_frame_id, int) or center_frame_id < 0:
            raise FrameSamplingError("Center frame ID must be a non-negative integer")
        if not math.isfinite(fps) or fps <= 0:
            raise FrameSamplingError("Video FPS must be a finite positive value")
        if not math.isfinite(window_sec) or window_sec < 0:
            raise FrameSamplingError("Frame window seconds must be finite and non-negative")
        if frame_count is not None and (frame_count < 1 or center_frame_id >= frame_count):
            raise FrameSamplingError("Center frame ID is outside known video frame bounds")
        radius = int(round(window_sec * fps))
        start = max(0, center_frame_id - radius)
        end = center_frame_id + radius
        if frame_count is not None:
            end = min(frame_count - 1, end)
        return FrameWindow(start_frame_id=start, end_frame_id=end, center_frame_id=center_frame_id)
