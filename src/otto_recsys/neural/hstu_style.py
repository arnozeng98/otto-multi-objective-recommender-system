from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from otto_recsys.neural.base import SequentialRetriever
from otto_recsys.neural.embeddings import EventEmbedding, last_valid


class HSTUStyleBlock(nn.Module):
    """Small gated temporal block inspired by HSTU, not Meta's production kernel."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(dimension, dimension * 2)
        self.output = nn.Linear(dimension, dimension)

    def forward(self, values: Tensor, padding_mask: Tensor, causal_mask: Tensor) -> Tensor:
        normalized = self.norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )
        gate, content = self.gate(attended).chunk(2, dim=-1)
        return cast(Tensor, values + self.output(torch.nn.functional.silu(gate) * content))


class HSTUStyleRetriever(SequentialRetriever):
    def __init__(
        self,
        item_count: int,
        dimension: int = 48,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = EventEmbedding(item_count, dimension)
        self.blocks = nn.ModuleList(HSTUStyleBlock(dimension, heads, 0.1) for _ in range(layers))
        self.output_norm = nn.LayerNorm(dimension)

    def encode(
        self,
        aid_ids: Tensor,
        event_types: Tensor,
        time_buckets: Tensor,
        padding_mask: Tensor,
        target_types: Tensor,
    ) -> Tensor:
        values = self.embedding.sequence(aid_ids, event_types, time_buckets)
        length = aid_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=aid_ids.device, dtype=torch.bool), diagonal=1
        )
        for block in self.blocks:
            values = block(values, padding_mask, causal_mask)
        query = last_valid(values, padding_mask) + self.embedding.targets(target_types)
        return torch.nn.functional.normalize(self.output_norm(query), dim=-1)

    def item_vectors(self) -> Tensor:
        return self.embedding.item_vectors()
