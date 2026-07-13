from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SemanticIdModel:
    codebooks: tuple[NDArray[np.float32], ...]

    def encode(self, vectors: NDArray[np.float32]) -> NDArray[np.int32]:
        residual = np.asarray(vectors, dtype=np.float32).copy()
        codes: list[NDArray[np.int32]] = []
        for codebook in self.codebooks:
            distances = ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
            code = np.argmin(distances, axis=1).astype(np.int32)
            codes.append(code)
            residual -= codebook[code]
        return np.stack(codes, axis=1)


def _kmeans(
    values: NDArray[np.float32], clusters: int, iterations: int, rng: np.random.Generator
) -> NDArray[np.float32]:
    if len(values) < clusters:
        raise ValueError("clusters cannot exceed the number of item vectors")
    centroids = values[rng.choice(len(values), size=clusters, replace=False)].copy()
    for _ in range(iterations):
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = np.argmin(distances, axis=1)
        for cluster in range(clusters):
            members = values[assignments == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return np.asarray(centroids, dtype=np.float32)


def train_residual_quantizer(
    vectors: NDArray[np.float32],
    *,
    levels: int = 3,
    clusters: int = 256,
    iterations: int = 10,
    seed: int = 2026,
) -> SemanticIdModel:
    """Learn collaborative residual-quantization codes for TIGER-style retrieval."""
    residual = np.asarray(vectors, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    codebooks: list[NDArray[np.float32]] = []
    for _ in range(levels):
        codebook = _kmeans(residual, clusters, iterations, rng)
        codebooks.append(codebook)
        distances = ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
        residual -= codebook[np.argmin(distances, axis=1)]
    return SemanticIdModel(tuple(codebooks))
