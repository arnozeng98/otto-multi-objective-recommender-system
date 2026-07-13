import pytest

from otto_recsys.constants import EventType
from otto_recsys.metrics import weighted_recall_at_k


def test_weighted_recall_uses_corpus_denominator_and_deduplicates_predictions() -> None:
    truth = {
        (1, EventType.CLICKS): [10],
        (1, EventType.CARTS): [20, 21],
        (1, EventType.ORDERS): [30, 31],
        (2, EventType.ORDERS): [40],
    }
    predictions = {
        (1, EventType.CLICKS): [10, 10],
        (1, EventType.CARTS): [20],
        (1, EventType.ORDERS): [30, 999],
        (2, EventType.ORDERS): [999],
    }

    result = weighted_recall_at_k(predictions, truth)

    assert result.per_type[EventType.CLICKS] == 1.0
    assert result.per_type[EventType.CARTS] == 0.5
    assert result.per_type[EventType.ORDERS] == pytest.approx(1 / 3)
    assert result.weighted == pytest.approx(0.45)


def test_recall_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        weighted_recall_at_k({}, {}, k=21)
