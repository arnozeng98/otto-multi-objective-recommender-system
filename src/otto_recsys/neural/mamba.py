from __future__ import annotations

import torch
from torch import Tensor, nn

from otto_recsys.neural.base import SequentialRetriever
from otto_recsys.neural.embeddings import EventEmbedding, last_valid


class MambaRetriever(SequentialRetriever):
    def __init__(self, item_count: int, dimension: int = 64, layers: int = 2) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise RuntimeError(
                "Mamba requires the optional Linux CUDA dependency: "
                "install the 'mamba' extra after installing CUDA-enabled PyTorch."
            ) from error
        self.embedding = EventEmbedding(item_count, dimension)
        self.blocks = nn.ModuleList(
            Mamba(d_model=dimension, d_state=16, d_conv=4, expand=2) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dimension) for _ in range(layers))
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
        for block, norm in zip(self.blocks, self.norms, strict=True):
            values = values + block(norm(values))
        query = last_valid(values, padding_mask) + self.embedding.targets(target_types)
        return torch.nn.functional.normalize(self.output_norm(query), dim=-1)

    def item_vectors(self) -> Tensor:
        return self.embedding.item_vectors()
