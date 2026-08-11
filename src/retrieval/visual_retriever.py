"""Generic vector retriever that keeps index and metadata responsibilities separate."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from domain.models import Candidate
from domain.protocols import FeatureStore, VectorIndex
from retrieval.candidate import candidate_from_hit


class VectorRetriever:
    def __init__(self, index: VectorIndex, feature_store: FeatureStore, source: str) -> None:
        self._index = index
        self._feature_store = feature_store
        self._source = source

    def retrieve(self, query_vector: np.ndarray, top_k: int) -> Sequence[Candidate]:
        hits = self._index.search(query_vector, top_k)
        return tuple(
            candidate_from_hit(
                hit,
                self._feature_store.metadata_for_id(hit.item_id),
                rank=rank,
                source=self._source,
            )
            for rank, hit in enumerate(hits, start=1)
        )
