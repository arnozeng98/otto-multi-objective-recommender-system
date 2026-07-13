from __future__ import annotations

import torch
from torch import Tensor, nn


class EventEmbedding(nn.Module):
    def __init__(self, item_count: int, dimension: int, time_bucket_count: int = 128) -> None:
        super().__init__()
        self.items = nn.Embedding(item_count + 1, dimension, padding_idx=0)
        self.events = nn.Embedding(4, dimension, padding_idx=0)
        self.times = nn.Embedding(time_bucket_count, dimension, padding_idx=0)
        self.targets = nn.Embedding(3, dimension)
        self.norm = nn.LayerNorm(dimension)

    def sequence(self, aids: Tensor, event_types: Tensor, time_buckets: Tensor) -> Tensor:
        sequence: Tensor = self.norm(
            self.items(aids) + self.events(event_types) + self.times(time_buckets)
        )
        return sequence

    def item_vectors(self) -> Tensor:
        vectors: Tensor = torch.nn.functional.normalize(self.items.weight[1:], dim=-1)
        return vectors


def last_valid(sequence: Tensor, padding_mask: Tensor) -> Tensor:
    lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
    rows = torch.arange(sequence.shape[0], device=sequence.device)
    selected: Tensor = sequence[rows, lengths]
    return selected
