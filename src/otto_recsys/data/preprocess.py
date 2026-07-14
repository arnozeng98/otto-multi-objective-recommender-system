from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from otto_recsys.constants import EventType
from otto_recsys.data.schemas import EVENT_SCHEMA


def _flush(rows: list[dict[str, Any]], writer: pq.ParquetWriter) -> int:
    if not rows:
        return 0
    writer.write_table(pa.Table.from_pylist(rows, schema=EVENT_SCHEMA))
    count = len(rows)
    rows.clear()
    return count


def convert_jsonl_to_parquet(
    source: Path,
    destination: Path,
    *,
    batch_events: int = 250_000,
    max_sessions: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, int | str]:
    """Flatten session JSONL into Parquet without retaining the dataset in memory."""
    if batch_events < 1:
        raise ValueError("batch_events must be positive")
    if max_sessions is not None and max_sessions < 1:
        raise ValueError("max_sessions must be positive when provided")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    sessions = 0
    events = 0
    digest = hashlib.sha256()
    try:
        with (
            source.open("rb") as input_file,
            pq.ParquetWriter(temporary, EVENT_SCHEMA, compression="zstd") as writer,
        ):
            for line in input_file:
                if max_sessions is not None and sessions >= max_sessions:
                    break
                digest.update(line)
                if progress is not None:
                    progress(len(line))
                payload = orjson.loads(line)
                session_id = int(payload["session"])
                sessions += 1
                for event in payload["events"]:
                    event_type = EventType(event["type"])
                    rows.append(
                        {
                            "session": session_id,
                            "aid": int(event["aid"]),
                            "ts": int(event["ts"]),
                            "type": event_type.value,
                        }
                    )
                    if len(rows) >= batch_events:
                        events += _flush(rows, writer)
            events += _flush(rows, writer)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"sessions": sessions, "events": events, "sha256": digest.hexdigest()}
