from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from otto_recsys.constants import EVENT_TYPES, EventType


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    sessions: int
    rows: int
    bytes: int
    sha256: str


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


def _iter_target_predictions(path: Path) -> Iterator[tuple[int, list[int]]]:
    current_session: int | None = None
    aids: list[int] = []
    for part in sorted(path.glob("part-*.parquet")):
        frame = pl.read_parquet(part, columns=["session", "aid", "rank"]).sort("session", "rank")
        for session, aid, _rank in frame.iter_rows():
            session = int(session)
            if current_session is not None and session != current_session:
                yield current_session, aids
                aids = []
            current_session = session
            aids.append(int(aid))
    if current_session is not None:
        yield current_session, aids


def write_submission_from_shards(
    destination: Path,
    sample_submission: Path,
    prediction_shards: Path,
    backfill: Mapping[EventType, Sequence[int]],
) -> SubmissionResult:
    """Stream ranked prediction shards into an atomically validated Kaggle CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    iterators = {
        target: iter(_iter_target_predictions(prediction_shards / target.value))
        for target in EVENT_TYPES
    }
    current = {target: next(iterators[target], None) for target in EVENT_TYPES}
    rows = 0
    sessions = 0
    previous_session: int | None = None
    session_targets: set[EventType] = set()
    digest = hashlib.sha256()
    try:
        with (
            sample_submission.open(newline="", encoding="utf-8") as sample,
            temporary.open("w", newline="", encoding="utf-8") as output,
        ):
            reader = csv.DictReader(sample)
            if reader.fieldnames != ["session_type", "labels"]:
                raise ValueError("Sample submission must contain session_type,labels")
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(["session_type", "labels"])
            for row in reader:
                row_id = row["session_type"]
                session_text, target_text = row_id.rsplit("_", 1)
                session = int(session_text)
                target = EventType(target_text)
                if previous_session is not None and session != previous_session:
                    if session_targets != set(EVENT_TYPES):
                        raise ValueError(
                            f"Sample submission has incomplete session {previous_session}"
                        )
                    sessions += 1
                    session_targets.clear()
                if target in session_targets:
                    raise ValueError(f"Sample submission repeats {row_id}")
                session_targets.add(target)
                previous_session = session

                candidate = current[target]
                while candidate is not None and candidate[0] < session:
                    candidate = next(iterators[target], None)
                current[target] = candidate
                predicted = (
                    candidate[1] if candidate is not None and candidate[0] == session else []
                )
                aids = list(dict.fromkeys([*predicted, *backfill[target]]))[:20]
                if not aids:
                    raise ValueError(f"No predictions available for {row_id}")
                writer.writerow([row_id, " ".join(map(str, aids))])
                rows += 1
            if previous_session is not None:
                if session_targets != set(EVENT_TYPES):
                    raise ValueError(f"Sample submission has incomplete session {previous_session}")
                sessions += 1
        with temporary.open("rb") as output:
            while block := output.read(1024 * 1024):
                digest.update(block)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    result = SubmissionResult(
        sessions=sessions,
        rows=rows,
        bytes=destination.stat().st_size,
        sha256=digest.hexdigest(),
    )
    (destination.parent / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "submission",
                "inputs": {
                    "sample_submission": str(sample_submission),
                    "prediction_shards": str(prediction_shards),
                },
                "result": asdict(result),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result
