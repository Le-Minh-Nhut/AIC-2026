"""Source-selection helpers shared by API request handling and runtime wiring."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


VISUAL_SOURCES = frozenset({"fgclip2", "pecore"})
AUXILIARY_SOURCES = frozenset({"ocr", "asr", "metadata"})
SUPPORTED_SOURCES = VISUAL_SOURCES | AUXILIARY_SOURCES


def normalize_sources(sources: Iterable[str] | None) -> tuple[str, ...]:
    """Validate and canonicalize the user-selected retrieval branches."""

    selected = tuple(sorted({source.strip().lower() for source in sources or () if source.strip()}))
    if not selected:
        return ("fgclip2", "pecore")
    unknown = sorted(set(selected) - SUPPORTED_SOURCES)
    if unknown:
        raise ValueError("Unsupported retrieval sources: " + ", ".join(unknown))
    if not set(selected) & VISUAL_SOURCES:
        raise ValueError("Select FG-CLIP2 or PE-Core; text-only UI search is not configured")
    return selected


def encoder_for_sources(sources: Iterable[str]) -> str:
    """Map the selected visual branches to the existing M2--M4 mode name."""

    selected = set(normalize_sources(sources))
    if {"fgclip2", "pecore"} <= selected:
        return "fg_pe_fusion"
    if "fgclip2" in selected:
        return "fgclip2_large"
    return "pecore_g14_448"


def apply_source_selection(
    retrieval_config: Mapping[str, Any], sources: Iterable[str]
) -> dict[str, Any]:
    """Enable only selected auxiliary branches without changing any ranker logic."""

    selected = set(normalize_sources(sources))
    configured = deepcopy(dict(retrieval_config))
    auxiliary = configured.get("auxiliary_retrieval", {})
    if not isinstance(auxiliary, dict):
        raise ValueError("auxiliary_retrieval must be a mapping")
    for source in AUXILIARY_SOURCES:
        branch = auxiliary.get(source)
        if not isinstance(branch, dict):
            continue
        branch["enabled"] = source in selected
    configured["auxiliary_retrieval"] = auxiliary
    return configured
