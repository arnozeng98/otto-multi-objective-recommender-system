import csv
from pathlib import Path

from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.submission import write_submission


def test_submission_has_three_rows_and_unique_labels(tmp_path: Path) -> None:
    destination = tmp_path / "submission.csv"
    predictions = {(10, EventType.CLICKS): [1, 1, 2]}
    backfill = {event_type: [2, 3] for event_type in EVENT_TYPES}

    rows = write_submission(destination, [10], predictions, backfill)

    with destination.open(newline="", encoding="utf-8") as source:
        content = list(csv.DictReader(source))
    assert rows == 3
    assert content[0] == {"session_type": "10_clicks", "labels": "1 2 3"}
