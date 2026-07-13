import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("model_name", ["sasrec", "hstu_style"])
def test_retriever_shapes_and_normalization(model_name: str) -> None:
    if model_name == "sasrec":
        from otto_recsys.neural.sasrec import SASRecRetriever

        model_class = SASRecRetriever
    else:
        from otto_recsys.neural.hstu_style import HSTUStyleRetriever

        model_class = HSTUStyleRetriever
    model = model_class(item_count=20, dimension=16, layers=1, heads=4)
    aids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    event_types = torch.tensor([[1, 1, 0], [1, 2, 3]])
    time_buckets = torch.tensor([[1, 2, 0], [1, 2, 3]])
    padding_mask = aids == 0
    targets = torch.tensor([0, 2])

    queries = model.encode(aids, event_types, time_buckets, padding_mask, targets)

    assert queries.shape == (2, 16)
    torch.testing.assert_close(torch.linalg.vector_norm(queries, dim=1), torch.ones(2))
