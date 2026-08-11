"""Configured storage cleanup plans that require an explicit delete decision."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from hardening.profiling import storage_size


class StorageCleanupError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StorageCleanupTarget:
    name: str
    paths: tuple[Path, ...]
    bytes_on_disk: int


@dataclass(frozen=True, slots=True)
class StorageCleanupPlan:
    targets: tuple[StorageCleanupTarget, ...]

    @property
    def total_bytes(self) -> int:
        return sum(target.bytes_on_disk for target in self.targets)


def plan_storage_cleanup(
    data_root: Path,
    configured_targets: Mapping[str, Sequence[object]],
    selected_names: Sequence[str],
) -> StorageCleanupPlan:
    names = tuple(selected_names) or tuple(sorted(configured_targets))
    if len(set(names)) != len(names):
        raise StorageCleanupError("Cleanup target names must be unique")
    targets: list[StorageCleanupTarget] = []
    for name in names:
        raw_paths = configured_targets.get(name)
        if raw_paths is None:
            raise StorageCleanupError(f"Unknown cleanup target: {name}")
        if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes)):
            raise StorageCleanupError(f"Cleanup target {name} must contain a path list")
        paths = tuple(_safe_target_path(data_root, str(path)) for path in raw_paths)
        targets.append(
            StorageCleanupTarget(name=name, paths=paths, bytes_on_disk=sum(storage_size(path) for path in paths))
        )
    return StorageCleanupPlan(targets=tuple(targets))


def execute_storage_cleanup(plan: StorageCleanupPlan, delete: bool = False) -> tuple[Path, ...]:
    if not delete:
        raise StorageCleanupError("Refusing cleanup without explicit delete=True")
    removed: list[Path] = []
    for target in plan.targets:
        for path in target.paths:
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
    return tuple(removed)


def _safe_target_path(data_root: Path, configured_path: str) -> Path:
    if not configured_path.strip():
        raise StorageCleanupError("Cleanup paths must be non-empty")
    root = data_root.resolve()
    candidate = (root / configured_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise StorageCleanupError(f"Cleanup path escapes data root: {configured_path}") from error
    if candidate == root:
        raise StorageCleanupError("Cleanup target cannot be the data root")
    return candidate
