"""Original-video decoding behind an optional OpenCV implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from PIL import Image


class VideoDecodingError(RuntimeError):
    pass


class VideoDecoderUnavailableError(VideoDecodingError):
    pass


@dataclass(frozen=True, slots=True)
class DecodedVideoInfo:
    fps: float
    frame_count: int | None


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    frame_id: int
    timestamp_sec: float
    image: Image.Image


class VideoFrameDecoder(Protocol):
    def inspect(self, video_path: Path) -> DecodedVideoInfo: ...

    def decode_frames(self, video_path: Path, frame_ids: Sequence[int]) -> tuple[DecodedFrame, ...]: ...


class OpenCVVideoDecoder:
    """Sequentially decodes requested frame IDs after one seek to the window start."""

    def inspect(self, video_path: Path) -> DecodedVideoInfo:
        capture, cv2 = self._open(video_path)
        try:
            return self._info(capture, cv2, video_path)
        finally:
            capture.release()

    def decode_frames(self, video_path: Path, frame_ids: Sequence[int]) -> tuple[DecodedFrame, ...]:
        requested = tuple(frame_ids)
        if not requested:
            raise VideoDecodingError("At least one frame ID is required for decoding")
        if any(not isinstance(frame_id, int) or frame_id < 0 for frame_id in requested):
            raise VideoDecodingError("Frame IDs must be non-negative integers")
        if len(set(requested)) != len(requested):
            raise VideoDecodingError("Frame decode request contains duplicate frame IDs")
        ordered_ids = tuple(sorted(requested))
        capture, cv2 = self._open(video_path)
        try:
            info = self._info(capture, cv2, video_path)
            if info.frame_count is not None and ordered_ids[-1] >= info.frame_count:
                raise VideoDecodingError(
                    f"Requested frame {ordered_ids[-1]} is outside video frame count {info.frame_count}"
                )
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, ordered_ids[0]):
                raise VideoDecodingError(f"Unable to seek to frame {ordered_ids[0]} in {video_path}")
            decoded: list[DecodedFrame] = []
            current_frame_id = ordered_ids[0]
            for requested_frame_id in ordered_ids:
                while current_frame_id <= requested_frame_id:
                    success, frame = capture.read()
                    if not success:
                        raise VideoDecodingError(
                            f"Unable to decode requested frame {current_frame_id} in {video_path}"
                        )
                    if current_frame_id == requested_frame_id:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        decoded.append(
                            DecodedFrame(
                                frame_id=current_frame_id,
                                timestamp_sec=current_frame_id / info.fps,
                                image=Image.fromarray(rgb),
                            )
                        )
                    current_frame_id += 1
            if len(decoded) != len(ordered_ids):
                raise VideoDecodingError("Decoder returned fewer frames than requested")
            return tuple(decoded)
        finally:
            capture.release()

    @staticmethod
    def _load_cv2() -> object:
        try:
            import cv2
        except ImportError as error:
            raise VideoDecoderUnavailableError(
                "Dense frame refinement requires optional dependency opencv-python-headless"
            ) from error
        return cv2

    def _open(self, video_path: Path) -> tuple[object, object]:
        if not video_path.is_file():
            raise FileNotFoundError(f"Original video does not exist: {video_path}")
        cv2 = self._load_cv2()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoDecodingError(f"OpenCV cannot open original video: {video_path}")
        return capture, cv2

    @staticmethod
    def _info(capture: object, cv2: object, video_path: Path) -> DecodedVideoInfo:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise VideoDecodingError(f"Video decoder returned invalid FPS for {video_path}: {fps}")
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(round(raw_count)) if math.isfinite(raw_count) and raw_count > 0 else None
        return DecodedVideoInfo(fps=fps, frame_count=frame_count)
