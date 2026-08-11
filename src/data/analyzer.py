"""Deterministic, non-mutating audit of downloaded AIC 2026 data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from data.archive_manifest import ArchiveManifestStore
from data.keyframe_mapping import infer_video_id
from data.manifest_builder import collect_manifest_records
from data.source_sheet import parse_source_sheet_csv
from domain.models import ArchiveCategory, KeyframeRecord, ValidationIssue, VideoRecord
from download.integrity import inspect_zip_archive


SCHEMA_VERSION = "1.0"
SEVERITY_ORDER = {"BLOCKER": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    data_root: Path
    report_path: Path
    sample_decode_videos: int = 20
    sample_images: int = 100
    random_seed: int = 2026
    timestamp_tolerance_seconds: float = 1.0
    duration_tolerance_seconds: float = 1.0


def _numeric_stats(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    numbers = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not numbers:
        return {"count": 0, "min": None, "mean": None, "median": None, "p10": None, "p90": None, "p95": None, "max": None}

    def percentile(percent: float) -> float:
        index = (len(numbers) - 1) * percent
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return numbers[lower]
        return numbers[lower] + (numbers[upper] - numbers[lower]) * (index - lower)

    return {
        "count": len(numbers),
        "min": numbers[0],
        "mean": sum(numbers) / len(numbers),
        "median": percentile(0.5),
        "p10": percentile(0.1),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "max": numbers[-1],
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if value not in {None, "", "N/A"}).items()))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class DataAnalyzer:
    def __init__(self, options: AnalysisOptions) -> None:
        self.options = options
        self.issues: list[ValidationIssue] = []

    def run(self) -> dict[str, Any]:
        data_root = self.options.data_root
        manifest_result = collect_manifest_records(
            data_root, timestamp_tolerance_seconds=self.options.timestamp_tolerance_seconds
        )
        archives = self._archive_inventory()
        files = self._file_inventory()
        videos = self._video_analysis(manifest_result.videos)
        keyframes = self._keyframe_analysis(manifest_result.keyframes, manifest_result.videos)
        mapping = self._mapping_analysis(manifest_result)
        btc_clip = self._btc_clip_analysis(manifest_result.keyframes)
        objects = self._json_component_analysis("objects")
        media_info = self._json_component_analysis("media_info")
        audio = self._audio_analysis(manifest_result.videos)
        storage = self._storage_analysis()
        cross_modal = self._cross_modal_analysis(
            manifest_result.videos,
            manifest_result.keyframes,
            manifest_result.mapping_load.records,
            objects,
            media_info,
        )
        corruption = self._corruption_sampling(manifest_result.videos, manifest_result.keyframes)
        readiness = self._readiness(
            manifest_result.videos,
            manifest_result.keyframes,
            mapping,
            btc_clip,
        )
        temporal = self._temporal_feasibility(manifest_result.keyframes, manifest_result.videos)
        self.issues.extend(manifest_result.mapping_validation.issues)
        report = {
            "schema_version": SCHEMA_VERSION,
            "dataset": {
                "generated_at": datetime.now(UTC).isoformat(),
                "data_root": str(data_root),
                "snapshot_hash": self._snapshot_hash(),
                "archives": archives,
            },
            "files": files,
            "videos": videos,
            "keyframes": keyframes,
            "mapping": mapping,
            "btc_clip": btc_clip,
            "objects": objects,
            "media_info": media_info,
            "audio": audio,
            "storage": storage,
            "cross_modal": cross_modal,
            "corruption_checks": corruption,
            "temporal_refinement": temporal,
            "readiness": readiness,
            "issues": self._serialized_issues(),
        }
        self._write_reports(report, mapping, videos)
        return report

    def _snapshot_hash(self) -> str | None:
        path = self.options.data_root / "manifests" / "source_sheet_snapshot.csv"
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _archive_inventory(self) -> dict[str, Any]:
        data_root = self.options.data_root
        snapshot_path = data_root / "manifests" / "source_sheet_snapshot.csv"
        expected: list[Any] = []
        parse_error: str | None = None
        if snapshot_path.exists():
            try:
                expected = list(parse_source_sheet_csv(snapshot_path.read_text(encoding="utf-8")).archives)
            except (OSError, ValueError) as error:
                parse_error = str(error)
                self.issues.append(ValidationIssue("HIGH", "invalid_source_snapshot", parse_error))
        else:
            self.issues.append(ValidationIssue("BLOCKER", "missing_source_snapshot", "Source sheet snapshot is absent"))
        try:
            manifest_records = ArchiveManifestStore(data_root / "manifests" / "archives_manifest.jsonl").load()
        except ValueError as error:
            manifest_records = {}
            self.issues.append(ValidationIssue("HIGH", "invalid_archives_manifest", str(error)))
        archive_root = data_root / "raw" / "archives"
        entries: list[dict[str, Any]] = []
        seen_names: Counter[str] = Counter()
        seen_urls: Counter[str] = Counter()
        for source in expected:
            seen_names[source.filename] += 1
            seen_urls[source.download_url] += 1
            path = archive_root / source.filename
            prior = manifest_records.get((source.filename, source.download_url))
            inspection = inspect_zip_archive(path) if path.exists() else None
            if path.exists() and inspection and not inspection.zip_valid:
                self.issues.append(
                    ValidationIssue("BLOCKER", "corrupt_archive", inspection.error or "Invalid ZIP", (source.filename,))
                )
            if path.exists() and path.stat().st_size == 0:
                self.issues.append(ValidationIssue("BLOCKER", "empty_archive", "Archive is empty", (source.filename,)))
            entries.append(
                {
                    "filename": source.filename,
                    "category": source.category.value,
                    "download_url": source.download_url,
                    "downloaded": bool(path.exists() and inspection and inspection.zip_valid),
                    "archive_path": _relative(path, data_root) if path.exists() else None,
                    "file_size_bytes": path.stat().st_size if path.exists() else None,
                    "sha256_local": prior.sha256_local if prior else None,
                    "zip_valid": inspection.zip_valid if inspection else None,
                    "entry_count": inspection.entry_count if inspection else None,
                    "compressed_size": inspection.compressed_size if inspection else None,
                    "estimated_uncompressed_size": inspection.uncompressed_size if inspection else None,
                    "extracted": prior.extracted if prior else False,
                    "status": prior.status if prior else ("missing" if not path.exists() else "unrecorded"),
                    "error": prior.error if prior else (inspection.error if inspection else None),
                }
            )
        duplicate_names = sorted(name for name, count in seen_names.items() if count > 1)
        duplicate_urls = sorted(url for url, count in seen_urls.items() if count > 1)
        if duplicate_names:
            self.issues.append(ValidationIssue("HIGH", "duplicate_archive_filename", "Duplicate source archive filenames", tuple(duplicate_names)))
        if duplicate_urls:
            self.issues.append(ValidationIssue("HIGH", "duplicate_archive_url", "Duplicate source archive URLs", tuple(duplicate_urls)))
        missing = [entry["filename"] for entry in entries if not entry["downloaded"]]
        if missing:
            self.issues.append(
                ValidationIssue(
                    "BLOCKER",
                    "missing_required_archives",
                    "Expected archives have not been downloaded and verified",
                    tuple(missing[:50]),
                )
            )
        return {
            "expected_count": len(expected),
            "downloaded_count": len(entries) - len(missing),
            "missing_count": len(missing),
            "missing_archives": missing,
            "corrupt_count": sum(entry["zip_valid"] is False for entry in entries),
            "compressed_bytes": sum(entry["file_size_bytes"] or 0 for entry in entries),
            "uncompressed_bytes_from_zip_metadata": sum(entry["estimated_uncompressed_size"] or 0 for entry in entries),
            "unusual_layout_archives": self._unusual_archive_layouts(entries),
            "snapshot_parse_error": parse_error,
            "entries": entries,
        }

    def _unusual_archive_layouts(self, entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        unusual: list[dict[str, Any]] = []
        for entry in entries:
            archive_path = entry["archive_path"]
            if not archive_path or entry["zip_valid"] is not True:
                continue
            category = ArchiveCategory(entry["category"])
            if category not in {ArchiveCategory.KEYFRAMES, ArchiveCategory.VIDEOS}:
                continue
            path = self.options.data_root / archive_path
            try:
                with zipfile.ZipFile(path) as archive:
                    members = [info.filename for info in archive.infolist() if not info.is_dir()][:100]
            except (OSError, zipfile.BadZipFile):
                continue
            if category is ArchiveCategory.KEYFRAMES:
                expected_marker = any(infer_video_id(member) for member in members)
            else:
                expected_marker = any("L" in Path(member).name.upper() or "L" in member.upper() for member in members)
            if not expected_marker:
                unusual.append({"filename": entry["filename"], "reason": "Expected logical ID marker not found in first 100 files"})
        return unusual

    def _file_inventory(self) -> dict[str, Any]:
        raw_root = self.options.data_root / "raw"
        if not raw_root.exists():
            return {"total_files": 0, "total_bytes": 0, "extension_counts": {}, "top_largest_files": [], "unexpected_extensions": [], "empty_files": [], "duplicate_relative_paths": []}
        files = [path for path in raw_root.rglob("*") if path.is_file() and "archives" not in path.relative_to(raw_root).parts]
        extension_counts = Counter(path.suffix.lower() or "[no extension]" for path in files)
        expected_extensions = {".mp4", ".mkv", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".npy", ".json", ".jsonl", ".csv", ".txt"}
        unexpected = sorted(extension for extension in extension_counts if extension not in expected_extensions)
        empty_files = [_relative(path, self.options.data_root) for path in files if path.stat().st_size == 0]
        if empty_files:
            self.issues.append(ValidationIssue("MEDIUM", "empty_extracted_file", "Extracted files with zero bytes", tuple(empty_files[:50])))
        return {
            "total_files": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "extension_counts": dict(sorted(extension_counts.items())),
            "top_largest_files": [
                {"path": _relative(path, self.options.data_root), "bytes": path.stat().st_size}
                for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:20]
            ],
            "unexpected_extensions": unexpected,
            "empty_files": empty_files[:100],
            "duplicate_relative_paths": [],
        }

    def _video_analysis(self, videos: Sequence[VideoRecord]) -> dict[str, Any]:
        unreadable = [video.video_id for video in videos if not video.is_readable]
        if unreadable:
            self.issues.append(ValidationIssue("HIGH", "unreadable_video", "Video probe failed", tuple(unreadable[:50])))
        duplicate_ids = [video_id for video_id, count in Counter(video.video_id for video in videos).items() if count > 1]
        if duplicate_ids:
            self.issues.append(ValidationIssue("HIGH", "duplicate_video_id", "Duplicate video IDs", tuple(duplicate_ids)))
        invalid_geometry = [
            video.video_id
            for video in videos
            if video.is_readable
            and (
                not video.fps
                or video.fps <= 0
                or not video.duration_sec
                or video.duration_sec <= 0
                or not video.width
                or not video.height
            )
        ]
        if invalid_geometry:
            self.issues.append(ValidationIssue("HIGH", "invalid_video_geometry", "Video has invalid timing or geometry", tuple(invalid_geometry[:50])))
        duration_mismatch: list[str] = []
        for video in videos:
            if video.fps and video.frame_count and video.duration_sec:
                if abs(video.frame_count / video.fps - video.duration_sec) > self.options.duration_tolerance_seconds:
                    duration_mismatch.append(video.video_id)
        return {
            "count": len(videos),
            "readable_count": len(videos) - len(unreadable),
            "unreadable_count": len(unreadable),
            "unreadable_videos": unreadable,
            "by_group": _counts(video.group_id for video in videos),
            "duration_sec_stats": _numeric_stats(video.duration_sec for video in videos),
            "total_duration_hours": sum(video.duration_sec or 0.0 for video in videos) / 3600,
            "fps_stats": _numeric_stats(video.fps for video in videos),
            "fps_distribution": _counts(video.fps for video in videos),
            "resolution_distribution": _counts(
                f"{video.width}x{video.height}" for video in videos if video.width and video.height
            ),
            "video_codec_distribution": _counts(video.video_codec for video in videos),
            "audio_codec_distribution": _counts(video.audio_codec for video in videos),
            "audio_sample_rate_distribution": _counts(video.audio_sample_rate for video in videos),
            "duration_frame_count_mismatch": duration_mismatch,
            "records": [video.as_dict() for video in videos],
        }

    def _keyframe_analysis(
        self, keyframes: Sequence[KeyframeRecord], videos: Sequence[VideoRecord]
    ) -> dict[str, Any]:
        by_video: dict[str, list[KeyframeRecord]] = defaultdict(list)
        for keyframe in keyframes:
            by_video[keyframe.video_id].append(keyframe)
        unreadable = [keyframe.keyframe_uid for keyframe in keyframes if not keyframe.is_readable]
        duplicate_uids = [uid for uid, count in Counter(keyframe.keyframe_uid for keyframe in keyframes).items() if count > 1]
        if unreadable:
            self.issues.append(ValidationIssue("HIGH", "unreadable_keyframe", "Keyframe image failed to load", tuple(unreadable[:50])))
        if duplicate_uids:
            self.issues.append(ValidationIssue("HIGH", "duplicate_keyframe_uid", "Duplicate stable keyframe IDs", tuple(duplicate_uids[:50])))
        intervals: list[float] = []
        intervals_by_video: dict[str, float] = {}
        fps_by_video = {video.video_id: video.fps for video in videos if video.fps and video.fps > 0}
        for video_id, values in by_video.items():
            mapped = sorted(
                (value for value in values if value.original_frame_id is not None),
                key=lambda value: value.keyframe_index if value.keyframe_index is not None else -1,
            )
            video_intervals = [
                (current.original_frame_id - previous.original_frame_id) / fps_by_video[video_id]
                for previous, current in zip(mapped, mapped[1:])
                if video_id in fps_by_video
                and previous.original_frame_id is not None
                and current.original_frame_id is not None
            ]
            intervals.extend(video_intervals)
            if video_intervals:
                intervals_by_video[video_id] = _numeric_stats(video_intervals)["median"]
        return {
            "count": len(keyframes),
            "videos_with_keyframes": len(by_video),
            "per_video_stats": _numeric_stats(len(values) for values in by_video.values()),
            "per_video_counts": dict(sorted((key, len(value)) for key, value in by_video.items())),
            "interval_sec_stats": _numeric_stats(intervals),
            "median_interval_sec_by_video": intervals_by_video,
            "resolution_distribution": _counts(
                f"{keyframe.width}x{keyframe.height}"
                for keyframe in keyframes
                if keyframe.width and keyframe.height
            ),
            "mode_distribution": _counts(keyframe.image_mode for keyframe in keyframes),
            "readable_count": len(keyframes) - len(unreadable),
            "unreadable_keyframes": unreadable,
            "duplicate_keyframe_uids": duplicate_uids,
        }

    def _mapping_analysis(self, manifest_result: Any) -> dict[str, Any]:
        validation = manifest_result.mapping_validation
        keyframes = manifest_result.keyframes
        mappings = manifest_result.mapping_load.records
        by_video_keyframes = Counter(keyframe.video_id for keyframe in keyframes)
        by_video_mappings = Counter(mapping.video_id for mapping in mappings)
        summary_table = []
        for video_id in sorted(set(by_video_keyframes) | set(by_video_mappings)):
            video_keyframes = [keyframe for keyframe in keyframes if keyframe.video_id == video_id]
            matched = sum(keyframe.has_mapping for keyframe in video_keyframes)
            summary_table.append(
                {
                    "video_id": video_id,
                    "keyframes": len(video_keyframes),
                    "mappings": by_video_mappings[video_id],
                    "matched": matched,
                    "missing": len(video_keyframes) - matched,
                    "invalid": 0,
                    "monotonic": video_id not in {
                        issue_id
                        for issue in validation.issues
                        if issue.code == "non_monotonic_mapping"
                        for issue_id in issue.affected_ids
                    },
                }
            )
        coverage = validation.matched_count / len(keyframes) if keyframes else 0.0
        return {
            "mapping_records": len(mappings),
            "coverage": coverage,
            "invalid_count": validation.invalid_frame_count,
            "non_monotonic_count": validation.non_monotonic_count,
            "validation": validation.as_dict(),
            "unsupported_files": list(manifest_result.mapping_load.unsupported_files),
            "malformed_rows": list(manifest_result.mapping_load.malformed_rows),
            "summary_table": summary_table,
        }

    def _btc_clip_analysis(self, keyframes: Sequence[KeyframeRecord]) -> dict[str, Any]:
        feature_root = self.options.data_root / "raw" / "btc_clip_features"
        entries: list[dict[str, Any]] = []
        if feature_root.exists():
            for path in sorted(feature_root.rglob("*.npy")):
                entry: dict[str, Any] = {"path": _relative(path, self.options.data_root), "byte_size": path.stat().st_size}
                try:
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    entry["shape"] = list(array.shape)
                    entry["dtype"] = str(array.dtype)
                    entry["row_count"] = int(array.shape[0]) if array.ndim >= 1 else None
                    if array.ndim != 2:
                        entry["status"] = "UNRESOLVED — expected a two-dimensional feature matrix"
                    else:
                        nan_count = 0
                        inf_count = 0
                        zero_count = 0
                        norms: list[float] = []
                        for start in range(0, array.shape[0], 4096):
                            chunk = np.asarray(array[start : start + 4096], dtype=np.float64)
                            nan_count += int(np.isnan(chunk).sum())
                            inf_count += int(np.isinf(chunk).sum())
                            chunk_norms = np.linalg.norm(chunk, axis=1)
                            zero_count += int(np.count_nonzero(chunk_norms == 0))
                            norms.extend(chunk_norms[np.isfinite(chunk_norms)].tolist())
                        entry.update(
                            {
                                "nan_count": nan_count,
                                "inf_count": inf_count,
                                "zero_vector_count": zero_count,
                                "norm_stats": _numeric_stats(norms),
                                "near_l2_normalized": bool(norms) and max(abs(norm - 1.0) for norm in norms) < 1e-3,
                                "status": "UNRESOLVED — feature row ordering is not verified",
                            }
                        )
                        if nan_count or inf_count:
                            self.issues.append(
                                ValidationIssue("HIGH", "invalid_btc_clip_values", "BTC CLIP features contain NaN or Inf", (entry["path"],))
                            )
                except (OSError, ValueError, MemoryError) as error:
                    entry["status"] = f"ERROR — {error}"
                    self.issues.append(ValidationIssue("HIGH", "unreadable_btc_clip", str(error), (entry["path"],)))
                entries.append(entry)
        row_count = sum(entry.get("row_count") or 0 for entry in entries)
        return {
            "file_count": len(entries),
            "files": entries,
            "total_rows": row_count,
            "keyframe_count": len(keyframes),
            "row_count_matches_keyframe_total": row_count == len(keyframes) if entries else False,
            "order_status": "UNRESOLVED — no verified official feature-to-keyframe ordering adapter is available",
            "ready": False,
        }

    def _json_component_analysis(self, component: str) -> dict[str, Any]:
        root = self.options.data_root / "raw" / component
        files = sorted(root.rglob("*.json")) if root.exists() else []
        schema_examples: list[dict[str, Any]] = []
        corrupt: list[str] = []
        top_types: Counter[str] = Counter()
        top_keys: Counter[str] = Counter()
        field_presence: Counter[str] = Counter()
        field_empty: Counter[str] = Counter()
        field_types: dict[str, Counter[str]] = defaultdict(Counter)
        dict_documents = 0
        inferred_ids: set[str] = set()
        for path in files:
            inferred = infer_video_id(str(path))
            if inferred:
                inferred_ids.add(inferred)
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                corrupt.append(_relative(path, self.options.data_root))
                continue
            value_type = type(value).__name__
            top_types[value_type] += 1
            if isinstance(value, dict):
                dict_documents += 1
                top_keys.update(str(key) for key in value.keys())
                for key, field_value in value.items():
                    field_name = str(key)
                    field_presence[field_name] += 1
                    field_types[field_name][type(field_value).__name__] += 1
                    if field_value is None or field_value == "" or field_value == [] or field_value == {}:
                        field_empty[field_name] += 1
                if len(schema_examples) < 5:
                    schema_examples.append(
                        {
                            "path": _relative(path, self.options.data_root),
                            "top_level_type": value_type,
                            "top_level_keys": sorted(str(key) for key in value.keys())[:30],
                        }
                    )
            elif len(schema_examples) < 5:
                schema_examples.append(
                    {"path": _relative(path, self.options.data_root), "top_level_type": value_type}
                )
        if corrupt:
            self.issues.append(ValidationIssue("MEDIUM", f"corrupt_{component}_json", "JSON files could not be parsed", tuple(corrupt[:50])))
        return {
            "file_count": len(files),
            "corrupt_count": len(corrupt),
            "corrupt_files": corrupt,
            "top_level_type_distribution": dict(top_types),
            "top_level_key_distribution": dict(top_keys.most_common(50)),
            "field_presence": {
                field: {
                    "present_count": count,
                    "presence_rate": count / dict_documents if dict_documents else 0.0,
                    "empty_count": field_empty[field],
                    "type_distribution": dict(field_types[field]),
                }
                for field, count in sorted(field_presence.items())
            },
            "schema_examples": schema_examples,
            "inferred_video_ids": sorted(inferred_ids),
            "schema_status": "NOT PRESENT" if not files else "DISCOVERED — semantic adapter intentionally not assumed",
        }

    def _audio_analysis(self, videos: Sequence[VideoRecord]) -> dict[str, Any]:
        with_audio = [video for video in videos if video.has_audio]
        return {
            "videos_with_audio": len(with_audio),
            "videos_without_audio": len(videos) - len(with_audio),
            "audio_duration_hours": sum(video.duration_sec or 0.0 for video in with_audio) / 3600,
            "audio_duration_hours_by_group": {
                group: sum(video.duration_sec or 0.0 for video in videos if video.group_id == group and video.has_audio) / 3600
                for group in sorted({video.group_id for video in videos if video.group_id})
            },
            "codec_distribution": _counts(video.audio_codec for video in with_audio),
            "sample_rate_distribution": _counts(video.audio_sample_rate for video in with_audio),
            "channels_distribution": _counts(video.audio_channels for video in with_audio),
        }

    def _storage_analysis(self) -> dict[str, Any]:
        raw_root = self.options.data_root / "raw"
        component_sizes = {}
        for name in ("archives", "videos", "keyframes", "btc_clip_features", "objects", "media_info", "map_keyframes"):
            root = raw_root / name
            component_sizes[name] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0
        processed_root = self.options.data_root / "processed"
        component_sizes["processed"] = sum(
            path.stat().st_size for path in processed_root.rglob("*") if path.is_file()
        ) if processed_root.exists() else 0
        disk = shutil.disk_usage(self.options.data_root if self.options.data_root.exists() else Path.cwd())
        return {"bytes": component_sizes, "free_disk_bytes": disk.free, "total_disk_bytes": disk.total, "embedding_estimates": "NOT PRESENT — runtime embedding dimension is unknown"}

    def _cross_modal_analysis(
        self,
        videos: Sequence[VideoRecord],
        keyframes: Sequence[KeyframeRecord],
        mappings: Sequence[Any],
        objects: dict[str, Any],
        media_info: dict[str, Any],
    ) -> dict[str, list[str]]:
        video_ids = {video.video_id for video in videos}
        keyframe_ids = {keyframe.video_id for keyframe in keyframes}
        mapping_ids = {mapping.video_id for mapping in mappings}
        object_ids = set(objects["inferred_video_ids"])
        metadata_ids = set(media_info["inferred_video_ids"])
        return {
            "K_minus_V": sorted(keyframe_ids - video_ids),
            "V_minus_K": sorted(video_ids - keyframe_ids),
            "M_minus_V": sorted(mapping_ids - video_ids),
            "V_minus_M": sorted(video_ids - mapping_ids),
            "O_minus_K": sorted(object_ids - keyframe_ids),
            "K_minus_O": sorted(keyframe_ids - object_ids),
            "I_minus_V": sorted(metadata_ids - video_ids),
            "V_minus_I": sorted(video_ids - metadata_ids),
        }

    def _corruption_sampling(
        self, videos: Sequence[VideoRecord], keyframes: Sequence[KeyframeRecord]
    ) -> dict[str, Any]:
        rng = random.Random(self.options.random_seed)
        image_failures: list[str] = []
        video_failures: list[str] = []
        video_skipped: list[str] = []
        keyframe_groups: dict[str, list[KeyframeRecord]] = defaultdict(list)
        for keyframe in keyframes:
            keyframe_groups[keyframe.video_id.split("_", maxsplit=1)[0]].append(keyframe)
        for group, records in keyframe_groups.items():
            sample = rng.sample(records, min(self.options.sample_images, len(records)))
            for record in sample:
                path = self.options.data_root / record.keyframe_path
                try:
                    with Image.open(path) as image:
                        image.convert("RGB")
                except (OSError, UnidentifiedImageError) as error:
                    image_failures.append(f"{record.keyframe_uid}: {error}")
        video_groups: dict[str, list[VideoRecord]] = defaultdict(list)
        for video in videos:
            video_groups[video.group_id or "UNRESOLVED"].append(video)
        for group, records in video_groups.items():
            sample = rng.sample(records, min(self.options.sample_decode_videos, len(records)))
            for record in sample:
                if not record.duration_sec or not record.is_readable:
                    video_skipped.append(f"{record.video_id}: unavailable duration or probe failure")
                    continue
                path = self.options.data_root / record.video_path
                for point, offset in (("first", 0.0), ("middle", record.duration_sec / 2), ("last", max(record.duration_sec - 0.05, 0.0))):
                    command = ["ffmpeg", "-v", "error", "-ss", str(offset), "-i", str(path), "-frames:v", "1", "-f", "null", "-"]
                    try:
                        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        video_skipped.append(f"{record.video_id}: ffmpeg unavailable ({error})")
                        break
                    if result.returncode != 0:
                        video_failures.append(f"{record.video_id} {point}: {result.stderr.strip() or 'decode failed'}")
        if image_failures or video_failures:
            self.issues.append(ValidationIssue("HIGH", "corruption_sampling_failure", "Sample media decoding failed", tuple((image_failures + video_failures)[:50])))
        return {
            "seed": self.options.random_seed,
            "image_failures": image_failures,
            "video_failures": video_failures,
            "video_decode_skipped": video_skipped,
        }

    def _temporal_feasibility(
        self, keyframes: Sequence[KeyframeRecord], videos: Sequence[VideoRecord]
    ) -> dict[str, Any]:
        fps_values = [video.fps for video in videos if video.fps and video.fps > 0]
        intervals: list[int] = []
        by_video: dict[str, list[KeyframeRecord]] = defaultdict(list)
        for keyframe in keyframes:
            if keyframe.original_frame_id is not None:
                by_video[keyframe.video_id].append(keyframe)
        for values in by_video.values():
            ordered = sorted(values, key=lambda value: value.keyframe_index if value.keyframe_index is not None else -1)
            intervals.extend(
                current.original_frame_id - previous.original_frame_id
                for previous, current in zip(ordered, ordered[1:])
                if previous.original_frame_id is not None and current.original_frame_id is not None
            )
        return {
            "keyframe_gap_frame_stats": _numeric_stats(intervals),
            "fps_stats": _numeric_stats(fps_values),
            "frames_in_windows": {
                window: _numeric_stats((fps or 0) * seconds for fps in fps_values)
                for window, seconds in {"plus_minus_1_sec": 2, "plus_minus_2_sec": 4, "plus_minus_3_sec": 6}.items()
            },
            "implication": "Geometry only: sparse mapped keyframes require original-video dense refinement; no accuracy claim is made.",
        }

    def _readiness(
        self,
        videos: Sequence[VideoRecord],
        keyframes: Sequence[KeyframeRecord],
        mapping: dict[str, Any],
        btc_clip: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        unique_keyframes = len({keyframe.keyframe_uid for keyframe in keyframes}) == len(keyframes)
        all_images_readable = bool(keyframes) and all(keyframe.is_readable for keyframe in keyframes)
        all_videos_readable = bool(videos) and all(video.is_readable for video in videos)
        mapping_complete = bool(keyframes) and mapping["coverage"] == 1.0 and mapping["invalid_count"] == 0
        return {
            "BTC_CLIP_READY": {
                "ready": False,
                "reason": "Feature row order is UNRESOLVED" if btc_clip["file_count"] else "BTC CLIP files are NOT PRESENT",
            },
            "FGCLIP2_ENCODING_READY": {
                "ready": unique_keyframes and all_images_readable,
                "reason": "Keyframe IDs must be unique and all images readable",
            },
            "PECORE_ENCODING_READY": {
                "ready": unique_keyframes and all_images_readable,
                "reason": "Keyframe IDs must be unique and all images readable",
            },
            "VIDEO_REFINEMENT_READY": {
                "ready": all_videos_readable and mapping_complete,
                "reason": "Videos must be readable and keyframe mapping complete/in-bounds",
            },
            "OCR_READY": {"ready": False, "reason": "SKIPPED — optional OCR dependency is not configured"},
            "ASR_READY": {"ready": False, "reason": "SKIPPED — ASR is out of Milestone 1 scope"},
        }

    def _serialized_issues(self) -> list[dict[str, Any]]:
        unique = {(issue.severity, issue.code, issue.message, issue.affected_ids): issue for issue in self.issues}
        return [
            issue.as_dict()
            for issue in sorted(unique.values(), key=lambda issue: (SEVERITY_ORDER[issue.severity], issue.code, issue.message))
        ]

    def _write_reports(self, report: dict[str, Any], mapping: dict[str, Any], videos: dict[str, Any]) -> None:
        reports_root = self.options.data_root / "reports"
        reports_root.mkdir(parents=True, exist_ok=True)
        json_path = reports_root / "data_analysis.json"
        json_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tables_root = reports_root / "data_analysis_tables"
        tables_root.mkdir(parents=True, exist_ok=True)
        self._write_csv(tables_root / "mapping_summary.csv", mapping["summary_table"])
        self._write_csv(
            tables_root / "videos.csv",
            videos["records"],
        )
        self.options.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.options.report_path.write_text(self._render_markdown(report), encoding="utf-8")

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _render_markdown(self, report: dict[str, Any]) -> str:
        archives = report["dataset"]["archives"]
        videos = report["videos"]
        keyframes = report["keyframes"]
        mapping = report["mapping"]
        btc_clip = report["btc_clip"]
        readiness = report["readiness"]
        issue_rows = report["issues"]
        blocker_rows = [issue for issue in issue_rows if issue["severity"] == "BLOCKER"]
        lines = [
            "# AIC 2026 Data Analysis",
            "",
            "## Executive Summary",
            f"- Generated at: {report['dataset']['generated_at']}",
            f"- Dataset snapshot SHA-256: {report['dataset']['snapshot_hash'] or 'NOT PRESENT'}",
            f"- Archives: {archives['downloaded_count']}/{archives['expected_count']} downloaded",
            f"- Videos: {videos['count']} ({videos['readable_count']} readable)",
            f"- Keyframes: {keyframes['count']} ({keyframes['readable_count']} readable)",
            f"- Mapping coverage: {mapping['coverage']:.2%}",
            f"- BTC CLIP status: {btc_clip['order_status']}",
            f"- Blocking issues: {len(blocker_rows)}",
            "",
            "## 1. Archives",
            f"- Expected: {archives['expected_count']}",
            f"- Missing: {archives['missing_count']}",
            f"- Corrupt: {archives['corrupt_count']}",
            f"- Compressed bytes: {archives['compressed_bytes']}",
            f"- ZIP metadata uncompressed bytes: {archives['uncompressed_bytes_from_zip_metadata']}",
            f"- Archives with unusual layout: {len(archives['unusual_layout_archives'])}",
            "",
            "## 2. Videos",
            f"- Videos per group: {json.dumps(videos['by_group'], ensure_ascii=False, sort_keys=True)}",
            f"- Duration statistics (seconds): {json.dumps(videos['duration_sec_stats'], sort_keys=True)}",
            f"- FPS distribution: {json.dumps(videos['fps_distribution'], sort_keys=True)}",
            f"- Resolution distribution: {json.dumps(videos['resolution_distribution'], sort_keys=True)}",
            "",
            "## 3. Keyframes",
            f"- Keyframes/video statistics: {json.dumps(keyframes['per_video_stats'], sort_keys=True)}",
            f"- Sampling interval statistics (seconds): {json.dumps(keyframes['interval_sec_stats'], sort_keys=True)}",
            f"- Image mode distribution: {json.dumps(keyframes['mode_distribution'], sort_keys=True)}",
            "",
            "## 4. Keyframe Mapping",
            f"- Mapping rows: {mapping['mapping_records']}",
            f"- Invalid mappings: {mapping['invalid_count']}",
            f"- Non-monotonic videos: {mapping['non_monotonic_count']}",
            f"- Unsupported mapping files: {json.dumps(mapping['unsupported_files'], ensure_ascii=False)}",
            "",
            "## 5. BTC CLIP Features",
            f"- Feature files: {btc_clip['file_count']}",
            f"- Total feature rows: {btc_clip['total_rows']}",
            f"- Ordering: {btc_clip['order_status']}",
            "",
            "## 6. Objects",
            f"- JSON files: {report['objects']['file_count']}",
            f"- Schema status: {report['objects']['schema_status']}",
            "",
            "## 7. Media Info",
            f"- JSON files: {report['media_info']['file_count']}",
            f"- Schema status: {report['media_info']['schema_status']}",
            f"- Top-level field presence: {json.dumps(report['media_info']['field_presence'], ensure_ascii=False, sort_keys=True)}",
            "",
            "## 8. Audio",
            f"- Videos with audio: {report['audio']['videos_with_audio']}",
            f"- Estimated ASR workload (hours): {report['audio']['audio_duration_hours']:.4f}",
            "",
            "## 9. Storage",
            f"- Bytes by component: {json.dumps(report['storage']['bytes'], sort_keys=True)}",
            f"- Free disk bytes: {report['storage']['free_disk_bytes']}",
            "",
            "## 10. Cross-modal Consistency",
        ]
        lines.extend(f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in report["cross_modal"].items())
        lines.extend(
            [
                "",
                "## 11. Corruption Checks",
                f"- Image failures: {len(report['corruption_checks']['image_failures'])}",
                f"- Video failures: {len(report['corruption_checks']['video_failures'])}",
                f"- Video checks skipped: {len(report['corruption_checks']['video_decode_skipped'])}",
                "",
                "## 12. Retrieval Readiness",
            ]
        )
        lines.extend(
            f"- {name}: {str(value['ready']).lower()} — {value['reason']}" for name, value in readiness.items()
        )
        lines.extend(
            [
                "",
                "## 13. Task Implications",
                "### KIS",
                "- Use mapped keyframes only for coarse candidates; readiness and interval statistics govern dense-frame refinement.",
                "### Q&A",
                "- Audio and metadata coverage are reported above; no model-performance claim is made.",
                "### TRAKE",
                "- Mapping completeness and keyframe gaps determine whether original-video refinement is required.",
                "",
                "## 14. Issues",
            ]
        )
        if issue_rows:
            lines.extend(
                f"- {issue['severity']} | {issue['code']} | {issue['message']} | {', '.join(issue['affected_ids']) or 'N/A'}"
                for issue in issue_rows
            )
        else:
            lines.append("- None detected by implemented checks.")
        lines.extend(
            [
                "",
                "## 15. Recommended Next Actions",
                "- Resolve all BLOCKER/HIGH data issues before retrieval, model encoding, or index construction.",
                "- Verify BTC CLIP feature ordering with official mapping metadata before enabling the BTC baseline.",
            ]
        )
        return "\n".join(lines) + "\n"


def analyze_data(options: AnalysisOptions) -> dict[str, Any]:
    return DataAnalyzer(options).run()
