"""Keyframe discovery and image-readability manifest records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from data.keyframe_mapping import infer_keyframe_index, mapping_lookup
from domain.models import KeyframeRecord, MappingRecord, VideoRecord


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _image_details(path: Path) -> tuple[int | None, int | None, str | None, bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width, image.height, image.mode, True, None
    except (OSError, UnidentifiedImageError) as error:
        return None, None, None, False, str(error)


def _video_fps(videos: Iterable[VideoRecord]) -> dict[str, float]:
    return {video.video_id: video.fps for video in videos if video.fps and video.fps > 0}


def scan_keyframe_records(
    data_root: Path,
    videos: Iterable[VideoRecord],
    mappings: Iterable[MappingRecord],
) -> list[KeyframeRecord]:
    keyframes_root = data_root / "raw" / "keyframes"
    if not keyframes_root.exists():
        return []
    mapping_by_keyframe = mapping_lookup(mappings)
    fps_by_video = _video_fps(videos)
    paths = sorted(
        path for path in keyframes_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    records: list[KeyframeRecord] = []
    for path in paths:
        relative = path.relative_to(keyframes_root)
        video_id = relative.parts[0] if len(relative.parts) > 1 else "UNRESOLVED"
        keyframe_index = infer_keyframe_index(path.stem)
        original_frame_id = (
            mapping_by_keyframe.get((video_id, keyframe_index)) if keyframe_index is not None else None
        )
        timestamp_sec = None
        if original_frame_id is not None and video_id in fps_by_video:
            timestamp_sec = original_frame_id / fps_by_video[video_id]
        width, height, image_mode, is_readable, read_error = _image_details(path)
        uid_suffix = f"{keyframe_index:06d}" if keyframe_index is not None else path.stem
        records.append(
            KeyframeRecord(
                keyframe_uid=f"{video_id}:{uid_suffix}",
                video_id=video_id,
                keyframe_index=keyframe_index,
                keyframe_path=str(path.relative_to(data_root)),
                original_frame_id=original_frame_id,
                timestamp_sec=timestamp_sec,
                width=width,
                height=height,
                file_size_bytes=path.stat().st_size,
                is_readable=is_readable,
                has_mapping=original_frame_id is not None,
                image_mode=image_mode,
                read_error=read_error,
            )
        )
    return records
