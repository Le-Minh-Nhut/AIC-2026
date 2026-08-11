"""Encoder utilities shared by retrieval adapters."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class EncoderUnavailableError(RuntimeError):
    pass


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in {1, 2}:
        raise ValueError(f"Expected one or two dimensions, received shape {array.shape}")
    norms = np.linalg.norm(array, axis=-1, keepdims=array.ndim == 2)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("Cannot normalize vectors with NaN, Inf, or zero norm")
    return array / norms


class TextEncoderAdapter:
    """Base class for adapters that expose normalized text embeddings."""

    @property
    def embedding_dimension(self) -> int | None:
        return None

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError
