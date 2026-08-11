"""Video discovery and ffprobe-backed manifest records."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from domain.models import VideoRecord


def _parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        numerator, denominator = value.split("/", maxsplit=1)
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    except (AttributeError, ValueError, ZeroDivisionError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value not in {None, "N/A", ""} else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in {None, "N/A", ""} else None
    except (TypeError, ValueError):
        return None


def infer_group_id(value: str) -> str | None:
    match = re.search(r"L\d+", value, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _unreadable_record(path: Path, data_root: Path, error: str) -> VideoRecord:
    return VideoRecord(
        video_id=path.stem,
        video_path=str(path.relative_to(data_root)),
        group_id=infer_group_id(str(path)),
        fps=None,
        frame_count=None,
        duration_sec=None,
        width=None,
        height=None,
        video_codec=None,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
        has_audio=False,
        container=None,
        file_size_bytes=path.stat().st_size,
        is_readable=False,
        probe_error=error,
    )


def probe_video_file(path: Path, data_root: Path) -> VideoRecord:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration,bit_rate:stream=index,codec_type,codec_name,"
        "width,height,avg_frame_rate,nb_frames,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return _unreadable_record(path, data_root, str(error))
    if completed.returncode != 0:
        return _unreadable_record(path, data_root, completed.stderr.strip() or "ffprobe failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return _unreadable_record(path, data_root, f"Invalid ffprobe JSON: {error}")
    streams = payload.get("streams", [])
    format_data = payload.get("format", {})
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video_stream is None:
        return _unreadable_record(path, data_root, "ffprobe found no video stream")
    fps = _parse_fraction(video_stream.get("avg_frame_rate"))
    frame_count = _integer(video_stream.get("nb_frames"))
    duration = _float(format_data.get("duration"))
    return VideoRecord(
        video_id=path.stem,
        video_path=str(path.relative_to(data_root)),
        group_id=infer_group_id(str(path)),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        width=_integer(video_stream.get("width")),
        height=_integer(video_stream.get("height")),
        video_codec=video_stream.get("codec_name"),
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        audio_sample_rate=_integer(audio_stream.get("sample_rate")) if audio_stream else None,
        audio_channels=_integer(audio_stream.get("channels")) if audio_stream else None,
        has_audio=audio_stream is not None,
        container=format_data.get("format_name"),
        file_size_bytes=path.stat().st_size,
        is_readable=True,
        bitrate=_integer(format_data.get("bit_rate")),
    )


def scan_video_records(data_root: Path) -> list[VideoRecord]:
    videos_root = data_root / "raw" / "videos"
    if not videos_root.exists():
        return []
    paths = sorted(
        path for path in videos_root.rglob("*") if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"}
    )
    return [probe_video_file(path, data_root) for path in paths]
