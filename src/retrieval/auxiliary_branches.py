"""Configuration-driven construction of optional OCR, ASR, and metadata branches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from data.text_artifacts import (
    load_asr_records_jsonl,
    load_metadata_records_jsonl,
    load_ocr_records_jsonl,
)
from domain.models import KeyframeRecord
from domain.protocols import QueryCandidateRetriever
from retrieval.text_retriever import (
    ASRTextRetriever,
    KeyframeCandidateMapper,
    MetadataTextRetriever,
    OCRTextRetriever,
)


class AuxiliaryBranchConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AuxiliaryBranchRuntime:
    branches: Mapping[str, QueryCandidateRetriever]
    metadata: dict[str, object]


def build_auxiliary_text_branches(
    config: Mapping[str, object],
    data_root: Path,
    keyframes: Sequence[KeyframeRecord],
) -> AuxiliaryBranchRuntime:
    """Load only enabled text artifacts; missing enabled input is a clear failure."""

    mapper = KeyframeCandidateMapper(keyframes)
    branches: dict[str, QueryCandidateRetriever] = {}
    metadata: dict[str, object] = {}
    for source in ("ocr", "asr", "metadata"):
        source_config = config.get(source, {})
        if not isinstance(source_config, Mapping):
            raise AuxiliaryBranchConfigError(f"auxiliary_retrieval.{source} must be a mapping")
        enabled = bool(source_config.get("enabled", False))
        metadata[source] = {"enabled": enabled}
        if not enabled:
            continue
        records_path = _artifact_path(source_config, data_root, source)
        if source == "ocr":
            records = load_ocr_records_jsonl(records_path)
            branches[source] = OCRTextRetriever(records, mapper)
        elif source == "asr":
            records = load_asr_records_jsonl(records_path)
            branches[source] = ASRTextRetriever(records, mapper)
        else:
            fields = source_config.get("fields", ())
            if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
                raise AuxiliaryBranchConfigError("auxiliary_retrieval.metadata.fields must be a list")
            records = load_metadata_records_jsonl(records_path, [str(field) for field in fields])
            branches[source] = MetadataTextRetriever(records, mapper)
        metadata[source] = {
            "enabled": True,
            "records_path": str(records_path),
            "record_count": len(records),
        }
        if source == "metadata":
            metadata[source]["fields"] = list(fields)
    return AuxiliaryBranchRuntime(branches=branches, metadata=metadata)


def _artifact_path(source_config: Mapping[str, object], data_root: Path, source: str) -> Path:
    value = source_config.get("records_path")
    if value is None or not str(value).strip():
        raise AuxiliaryBranchConfigError(f"auxiliary_retrieval.{source}.records_path must be configured")
    path = Path(str(value))
    return path if path.is_absolute() else data_root / path
