"""Shared feature-store validation data contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureValidationReport:
    count: int
    dimension: int
    dtypes: tuple[str, ...]
    nan_count: int
    inf_count: int
    zero_vector_count: int
    min_norm: float
    max_norm: float
    vectors_are_l2_normalized: bool
    uses_mmap: bool

