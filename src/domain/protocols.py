"""Dependency-inversion contracts for encoders, stores, indexes, and retrieval."""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence

import numpy as np
from PIL import Image

from domain.models import Candidate, KeyframeRecord, SearchHit


class TextEncoder(Protocol):
    @property
    def embedding_dimension(self) -> int | None: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class ImageTextEncoder(TextEncoder, Protocol):
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...


class FeatureStore(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def count(self) -> int: ...

    def iter_batches(self, batch_size: int) -> Iterator[tuple[tuple[str, ...], np.ndarray]]: ...

    def metadata_for_id(self, item_id: str) -> KeyframeRecord: ...


class VectorIndex(Protocol):
    @property
    def dimension(self) -> int: ...

    def search(self, query: np.ndarray, top_k: int) -> Sequence[SearchHit]: ...


class Retriever(Protocol):
    def retrieve(self, query_vector: np.ndarray, top_k: int) -> Sequence[Candidate]: ...


class QueryCandidateRetriever(Protocol):
    """A named branch that ranks mapped candidates directly from query text."""

    source: str

    def retrieve(self, query: str, top_k: int) -> Sequence[Candidate]: ...


class VisualAnswerer(Protocol):
    """Dependency-inversion boundary for multi-frame visual question answering."""

    def answer(
        self,
        images: Sequence[Image.Image],
        event_description: str,
        question: str,
    ) -> str: ...
