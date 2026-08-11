"""Validated loading of video-manifest records for downstream frame work."""

from __future__ import annotations

from pathlib import Path

from domain.models import VideoRecord


class VideoManifestValidationError(ValueError):
    pass


def _video_record_from_dict(value: dict[str, object]) -> VideoRecord:
    required = {
        "video_id",
        "video_path",
        "group_id",
        "fps",
        "frame_count",
        "duration_sec",
        "width",
        "height",
        "video_codec",
        "audio_codec",
        "audio_sample_rate",
        "audio_channels",
        "has_audio",
        "container",
        "file_size_bytes",
        "is_readable",
        "bitrate",
        "probe_error",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise VideoManifestValidationError(
            "Video manifest row is missing refinement metadata fields: " + ", ".join(missing)
        )
    return VideoRecord(
        video_id=str(value["video_id"]),
        video_path=str(value["video_path"]),
        group_id=str(value["group_id"]) if value["group_id"] is not None else None,
        fps=float(value["fps"]) if value["fps"] is not None else None,
        frame_count=int(value["frame_count"]) if value["frame_count"] is not None else None,
        duration_sec=float(value["duration_sec"]) if value["duration_sec"] is not None else None,
        width=int(value["width"]) if value["width"] is not None else None,
        height=int(value["height"]) if value["height"] is not None else None,
        video_codec=str(value["video_codec"]) if value["video_codec"] is not None else None,
        audio_codec=str(value["audio_codec"]) if value["audio_codec"] is not None else None,
        audio_sample_rate=(
            int(value["audio_sample_rate"]) if value["audio_sample_rate"] is not None else None
        ),
        audio_channels=int(value["audio_channels"]) if value["audio_channels"] is not None else None,
        has_audio=bool(value["has_audio"]),
        container=str(value["container"]) if value["container"] is not None else None,
        file_size_bytes=int(value["file_size_bytes"]),
        is_readable=bool(value["is_readable"]),
        bitrate=int(value["bitrate"]) if value["bitrate"] is not None else None,
        probe_error=str(value["probe_error"]) if value["probe_error"] is not None else None,
    )


def load_video_records_from_parquet(path: Path) -> tuple[VideoRecord, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise VideoManifestValidationError(
            "Reading videos_manifest.parquet requires pyarrow. Install project dependencies first."
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Video manifest does not exist: {path}")
    rows = pq.read_table(path).to_pylist()
    records = tuple(_video_record_from_dict(row) for row in rows)
    if len({record.video_id for record in records}) != len(records):
        raise VideoManifestValidationError("Video manifest contains duplicate video_id values")
    return records
