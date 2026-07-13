from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class SequentialRetriever(nn.Module, ABC):
    """Shared contract for target-aware session retrieval models."""

    @abstractmethod
    def encode(
        self,
        aid_ids: Tensor,
        event_types: Tensor,
        time_buckets: Tensor,
        padding_mask: Tensor,
        target_types: Tensor,
    ) -> Tensor:
        """Return one normalized query vector per session."""

    @abstractmethod
    def item_vectors(self) -> Tensor:
        """Return normalized vectors for all real items."""

    def forward(
        self,
        aid_ids: Tensor,
        event_types: Tensor,
        time_buckets: Tensor,
        padding_mask: Tensor,
        target_types: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        queries = self.encode(aid_ids, event_types, time_buckets, padding_mask, target_types)
        candidates = self.item_vectors()[candidate_ids]
        return torch.einsum("bd,bkd->bk", queries, candidates)
