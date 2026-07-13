from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class VectorIndex:
    """FAISS-backed inner-product index with an exact NumPy fallback."""

    def __init__(self, vectors: NDArray[np.float32], *, prefer_faiss: bool = True) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        self.vectors = np.asarray(vectors / norms, dtype=np.float32)
        self._index: object | None = None
        if prefer_faiss:
            try:
                import faiss

                index = faiss.IndexFlatIP(self.vectors.shape[1])
                index.add(self.vectors)
                self._index = index
            except ImportError:
                pass

    def search(
        self, queries: NDArray[np.float32], top_k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        normalized = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-12)
        normalized = np.asarray(normalized, dtype=np.float32)
        if self._index is not None:
            scores, indices = self._index.search(normalized, top_k)  # type: ignore[attr-defined]
            return scores, indices
        similarities = normalized @ self.vectors.T
        indices = np.argsort(-similarities, axis=1)[:, :top_k]
        scores = np.take_along_axis(similarities, indices, axis=1)
        return np.asarray(scores, dtype=np.float32), np.asarray(indices, dtype=np.int64)
