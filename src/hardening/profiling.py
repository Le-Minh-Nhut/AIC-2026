"""Low-overhead latency, RAM, optional VRAM, and storage profiling utilities."""

from __future__ import annotations

import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypeVar


Value = TypeVar("Value")


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    latency_ms: float
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None
    storage_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_callable(
    operation: Callable[[], Value],
    storage_paths: tuple[Path, ...] = (),
) -> tuple[Value, ResourceProfile]:
    peak_before = _peak_rss_bytes()
    vram_before = _reset_vram_peak()
    started = time.perf_counter()
    value = operation()
    latency_ms = (time.perf_counter() - started) * 1000
    peak_after = _peak_rss_bytes()
    return value, ResourceProfile(
        latency_ms=latency_ms,
        peak_rss_bytes=max(peak_before or 0, peak_after or 0) or None,
        peak_vram_bytes=_peak_vram_bytes(vram_before),
        storage_bytes=sum(storage_size(path) for path in storage_paths),
    )


def profile_subprocess(
    command: tuple[str, ...],
    storage_paths: tuple[Path, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], ResourceProfile]:
    if not command:
        raise ValueError("Profile command must be non-empty")
    started = time.perf_counter()
    process = subprocess.Popen(command, text=True)
    peak_vram_bytes: int | None = None
    while process.poll() is None:
        observed_vram = _nvidia_smi_vram_bytes()
        peak_vram_bytes = max(peak_vram_bytes or 0, observed_vram or 0) or None
        time.sleep(0.1)
    result = subprocess.CompletedProcess(command, process.wait())
    latency_ms = (time.perf_counter() - started) * 1000
    child_rss = _peak_child_rss_bytes()
    return result, ResourceProfile(
        latency_ms=latency_ms,
        peak_rss_bytes=child_rss,
        peak_vram_bytes=peak_vram_bytes,
        storage_bytes=sum(storage_size(path) for path in storage_paths),
    )


def storage_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def _peak_rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    return int(value if platform.system() == "Darwin" else value * 1024)


def _peak_child_rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (AttributeError, OSError):
        return None
    return int(value if platform.system() == "Darwin" else value * 1024)


def _reset_vram_peak() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except (ImportError, RuntimeError):
        pass
    return False


def _peak_vram_bytes(enabled: bool) -> int | None:
    if not enabled:
        return None
    try:
        import torch

        return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None


def _nvidia_smi_vram_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    try:
        return sum(int(value) * 1024 * 1024 for value in values)
    except ValueError:
        return None
