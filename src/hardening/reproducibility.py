"""Best-effort deterministic setup without making Torch a base dependency."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    seed: int
    torch_configured: bool
    torch_error: str | None = None


def configure_determinism(seed: int) -> DeterminismReport:
    if seed < 0:
        raise ValueError("Deterministic seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        return DeterminismReport(seed=seed, torch_configured=False)
    except Exception as error:
        return DeterminismReport(seed=seed, torch_configured=False, torch_error=str(error))
    return DeterminismReport(seed=seed, torch_configured=True)
