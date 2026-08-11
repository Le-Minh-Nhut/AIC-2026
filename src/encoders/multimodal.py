"""Reusable batching and validation for image-text embedding adapters."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, TypeVar

import numpy as np
from PIL import Image

from encoders.base import l2_normalize


class ImageTextEmbeddingBackend(Protocol):
    @property
    def embedding_dimension(self) -> int | None: ...

    @property
    def preprocessing_config(self) -> dict[str, object]: ...

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


Item = TypeVar("Item")


class BatchedImageTextEncoder:
    """Normalizes backend outputs while keeping model loading separate."""

    def __init__(
        self,
        backend: ImageTextEmbeddingBackend,
        batch_size: int,
        encoder_label: str,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._backend = backend
        self._batch_size = batch_size
        self._encoder_label = encoder_label
        self._embedding_dimension = backend.embedding_dimension

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    @property
    def preprocessing_config(self) -> dict[str, object]:
        return self._backend.preprocessing_config

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("At least one image is required")
        vectors = self._encode_batched(images, self._backend.encode_images, "image")
        return self._normalize_and_validate(vectors, len(images), "image")

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("At least one non-empty query text is required")
        vectors = self._encode_batched(texts, self._backend.encode_texts, "text")
        return self._normalize_and_validate(vectors, len(texts), "text")

    def _encode_batched(
        self,
        items: Sequence[Item],
        method: Callable[[Sequence[Item]], np.ndarray],
        modality: str,
    ) -> np.ndarray:
        batches = [
            np.asarray(method(items[start : start + self._batch_size]), dtype=np.float32)
            for start in range(0, len(items), self._batch_size)
        ]
        for batch in batches:
            if batch.ndim != 2:
                raise ValueError(
                    f"{self._encoder_label} {modality} backend must return a two-dimensional array"
                )
        return np.concatenate(batches, axis=0)

    def _normalize_and_validate(
        self,
        vectors: np.ndarray,
        expected_rows: int,
        modality: str,
    ) -> np.ndarray:
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ValueError(
                f"{self._encoder_label} {modality} encoder returned shape {vectors.shape}; "
                f"expected [{expected_rows}, dimension]"
            )
        dimension = int(vectors.shape[1])
        if self._embedding_dimension is None:
            self._embedding_dimension = dimension
        if dimension != self._embedding_dimension:
            raise ValueError(
                f"{self._encoder_label} embedding dimension changed from "
                f"{self._embedding_dimension} to {dimension}"
            )
        return l2_normalize(vectors)
