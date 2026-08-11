"""Chunked exact cosine/inner-product search over a generic feature store."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from domain.models import SearchHit
from domain.protocols import FeatureStore
from encoders.base import l2_normalize


class ExactCosineIndex:
    def __init__(self, feature_store: FeatureStore, batch_size: int = 8192) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._feature_store = feature_store
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._feature_store.dimension

    def search(self, query: np.ndarray, top_k: int) -> Sequence[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError(f"Query vector must have shape [dimension], received {vector.shape}")
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {vector.shape[0]} does not match feature dimension {self.dimension}"
            )
        normalized_query = l2_normalize(vector)
        best: list[SearchHit] = []
        for ids, vectors in self._feature_store.iter_batches(self._batch_size):
            normalized_vectors = l2_normalize(vectors)
            scores = normalized_vectors @ normalized_query
            take = min(top_k, len(ids))
            selected = np.argpartition(scores, -take)[-take:]
            best.extend(SearchHit(ids[index], float(scores[index])) for index in selected)
        return tuple(sorted(best, key=lambda hit: (-hit.score, hit.item_id))[:top_k])
