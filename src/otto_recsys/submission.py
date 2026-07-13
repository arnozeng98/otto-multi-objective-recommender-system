from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from otto_recsys.constants import EVENT_TYPES, EventType


def write_submission(
    destination: Path,
    sessions: Sequence[int],
    predictions: Mapping[tuple[int, EventType], Sequence[int]],
    backfill: Mapping[EventType, Sequence[int]],
) -> int:
    """Write and validate the Kaggle three-row-per-session submission contract."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["session_type", "labels"])
        for session_id in sessions:
            for event_type in EVENT_TYPES:
                aids = list(dict.fromkeys(predictions.get((session_id, event_type), ())))
                aids.extend(aid for aid in backfill[event_type] if aid not in aids)
                labels = aids[:20]
                if not labels:
                    row_id = f"{session_id}_{event_type.value}"
                    raise ValueError(f"No predictions available for {row_id}")
                writer.writerow([f"{session_id}_{event_type.value}", " ".join(map(str, labels))])
                rows += 1
    expected = len(sessions) * len(EVENT_TYPES)
    if rows != expected:
        raise RuntimeError(f"Wrote {rows} rows; expected {expected}")
    return rows
