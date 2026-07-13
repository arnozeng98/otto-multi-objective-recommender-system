import numpy as np

from otto_recsys.neural.index import VectorIndex
from otto_recsys.neural.semantic_ids import train_residual_quantizer


def test_exact_vector_index_returns_nearest_item() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    index = VectorIndex(vectors, prefer_faiss=False)

    _, neighbors = index.search(np.asarray([[0.9, 0.1]], dtype=np.float32), top_k=2)

    assert neighbors[0, 0] == 0


def test_residual_quantizer_is_deterministic() -> None:
    vectors = np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]], dtype=np.float32)
    first = train_residual_quantizer(vectors, levels=2, clusters=2, iterations=3)
    second = train_residual_quantizer(vectors, levels=2, clusters=2, iterations=3)

    np.testing.assert_array_equal(first.encode(vectors), second.encode(vectors))
