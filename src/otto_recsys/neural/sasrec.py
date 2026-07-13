from __future__ import annotations

import torch
from torch import Tensor, nn

from otto_recsys.neural.base import SequentialRetriever
from otto_recsys.neural.embeddings import EventEmbedding, last_valid


class SASRecRetriever(SequentialRetriever):
    def __init__(
        self,
        item_count: int,
        dimension: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = EventEmbedding(item_count, dimension)
        block = nn.TransformerEncoderLayer(
            dimension,
            heads,
            dimension * 4,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dimension)

    def encode(
        self,
        aid_ids: Tensor,
        event_types: Tensor,
        time_buckets: Tensor,
        padding_mask: Tensor,
        target_types: Tensor,
    ) -> Tensor:
        sequence = self.embedding.sequence(aid_ids, event_types, time_buckets)
        length = aid_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=aid_ids.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.encoder(sequence, mask=causal_mask, src_key_padding_mask=padding_mask)
        query = last_valid(encoded, padding_mask) + self.embedding.targets(target_types)
        return torch.nn.functional.normalize(self.output_norm(query), dim=-1)

    def item_vectors(self) -> Tensor:
        return self.embedding.item_vectors()
