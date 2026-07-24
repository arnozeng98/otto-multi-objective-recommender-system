import csv
from pathlib import Path

import polars as pl

from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.submission import write_submission, write_submission_from_shards


def test_submission_has_three_rows_and_unique_labels(tmp_path: Path) -> None:
    destination = tmp_path / "submission.csv"
    predictions = {(10, EventType.CLICKS): [1, 1, 2]}
    backfill = {event_type: [2, 3] for event_type in EVENT_TYPES}

    rows = write_submission(destination, [10], predictions, backfill)

    with destination.open(newline="", encoding="utf-8") as source:
        content = list(csv.DictReader(source))
    assert rows == 3
    assert content[0] == {"session_type": "10_clicks", "labels": "1 2 3"}


def test_streaming_submission_uses_sample_order_and_backfill(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    sample.write_text(
        "session_type,labels\n"
        "10_clicks,1\n10_carts,1\n10_orders,1\n"
        "20_clicks,1\n20_carts,1\n20_orders,1\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions"
    for target in EVENT_TYPES:
        directory = predictions / target.value
        directory.mkdir(parents=True)
        pl.DataFrame(
            {
                "session": [10, 20],
                "aid": [100 + len(target.value), 200 + len(target.value)],
                "rank": [1, 1],
            }
        ).write_parquet(directory / "part-000000.parquet")
    destination = tmp_path / "submission" / "submission.csv"
    backfill = {target: [999] for target in EVENT_TYPES}

    result = write_submission_from_shards(destination, sample, predictions, backfill)

    with destination.open(newline="", encoding="utf-8") as source:
        content = list(csv.DictReader(source))
    assert result.sessions == 2
    assert result.rows == 6
    assert content[0]["session_type"] == "10_clicks"
    assert content[0]["labels"].endswith("999")
    assert (destination.parent / "manifest.json").exists()
