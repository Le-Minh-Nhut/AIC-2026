"""BTC CLIP text-encoder adapter with no implicit model download."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from encoders.base import EncoderUnavailableError, TextEncoderAdapter, l2_normalize


class ClipTextBackend(Protocol):
    @property
    def embedding_dimension(self) -> int | None: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class BtcClipTextEncoder(TextEncoderAdapter):
    """Normalizes embeddings produced by a BTC-compatible CLIP backend."""

    def __init__(self, backend: ClipTextBackend) -> None:
        self._backend = backend
        self._embedding_dimension = backend.embedding_dimension

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("At least one non-empty query text is required")
        vectors = np.asarray(self._backend.encode_texts(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(
                "CLIP backend must return a two-dimensional array with one embedding per text; "
                f"received {vectors.shape} for {len(texts)} text(s)"
            )
        if self._embedding_dimension is None:
            self._embedding_dimension = int(vectors.shape[1])
        if vectors.shape[1] != self._embedding_dimension:
            raise ValueError(
                f"CLIP embedding dimension changed from {self._embedding_dimension} to {vectors.shape[1]}"
            )
        return l2_normalize(vectors)


class OpenClipTextBackend:
    """Optional local-checkpoint adapter for OpenCLIP's ViT-B/32 text encoder."""

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._embedding_dimension: int | None = None

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint: Path,
        model_name: str = "ViT-B-32",
        device: str = "cpu",
    ) -> "OpenClipTextBackend":
        if not checkpoint.is_file():
            raise EncoderUnavailableError(f"BTC CLIP checkpoint does not exist: {checkpoint}")
        try:
            import open_clip
            import torch
        except ImportError as error:
            raise EncoderUnavailableError(
                "BTC CLIP text encoding requires optional dependencies open_clip_torch and torch"
            ) from error
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=str(checkpoint),
            device=device,
        )
        model.eval()
        backend = cls(model=model, tokenizer=open_clip.get_tokenizer(model_name), device=device)
        backend._torch = torch
        return backend

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            tokens = self._tokenizer(list(texts)).to(self._device)
            vectors = self._model.encode_text(tokens)
        result = vectors.detach().float().cpu().numpy()
        self._embedding_dimension = int(result.shape[1])
        return result
