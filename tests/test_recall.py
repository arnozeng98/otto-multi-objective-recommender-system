from pathlib import Path

import polars as pl
import pytest

from otto_recsys.constants import EventType
from otto_recsys.metrics import evaluate_candidate_recall, weighted_recall_at_k


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


def test_candidate_recall_reports_cutoffs_sources_and_marginal_gain(tmp_path: Path) -> None:
    labels = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "session": [1, 1, 2, 1, 1, 1],
            "type": ["clicks", "clicks", "clicks", "carts", "orders", "orders"],
            "aid": [10, 10, 20, 30, 40, 41],
        }
    ).write_parquet(labels)
    candidates = tmp_path / "candidates"
    rows = {
        "clicks": {
            "session": [1, 1, 2],
            "aid": [10, 99, 20],
            "label": [1, 0, 1],
            "candidate_rank": [1, 2, 50],
            "session_length": [1, 1, 1],
            "source_history_present": [1, 0, 0],
            "source_covisit_present": [0, 1, 1],
        },
        "carts": {
            "session": [1],
            "aid": [99],
            "label": [0],
            "candidate_rank": [1],
            "session_length": [1],
            "source_history_present": [1],
            "source_covisit_present": [0],
        },
        "orders": {
            "session": [1],
            "aid": [40],
            "label": [1],
            "candidate_rank": [20],
            "session_length": [1],
            "source_history_present": [1],
            "source_covisit_present": [1],
        },
    }
    for target, data in rows.items():
        directory = candidates / target
        directory.mkdir(parents=True)
        pl.DataFrame(data).write_parquet(directory / "part-000000.parquet")

    report = evaluate_candidate_recall(candidates, labels, cutoffs=(20, 50, 100))

    assert report.targets["clicks"].denominator == 2
    assert report.targets["clicks"].recall_at == {20: 0.5, 50: 1.0, 100: 1.0}
    assert report.targets["clicks"].source_recall["history"] == 0.5
    assert report.targets["clicks"].source_marginal_recall["covisit"] == 0.5
    assert report.targets["clicks"].source_union_recall == 1.0
    assert report.targets["clicks"].session_length_recall_at_20 == {"1": 0.5}
    assert report.targets["orders"].recall_at[20] == 0.5
    assert report.weighted_recall_at_20 == pytest.approx(0.35)
    assert report.weighted_recall_at_100 == pytest.approx(0.4)
