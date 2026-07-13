from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from otto_recsys.artifacts import stable_hash
from otto_recsys.covisitation.partitioned import iter_parquet_sessions
from otto_recsys.data.schemas import EVENT_SCHEMA, Event
from otto_recsys.data.split import split_session

LABEL_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int64(), nullable=False),
        pa.field("type", pa.string(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class TemporalMaterializationResult:
    source_sessions: int
    matrix_events: int
    query_sessions: int
    query_context_events: int
    labels: int
    cutoff_timestamp_ms: int
    label_end_timestamp_ms: int | None


@dataclass(frozen=True, slots=True)
class ValidationViewsResult:
    ranker_train: TemporalMaterializationResult
    local_validation: TemporalMaterializationResult


def _event_row(session_id: int, event: Event) -> dict[str, int | str]:
    return {
        "session": session_id,
        "aid": event.aid,
        "ts": event.ts,
        "type": event.type.value,
    }


def _flush(
    rows: list[dict[str, Any]],
    writer: pq.ParquetWriter,
    schema: pa.Schema,
) -> int:
    if not rows:
        return 0
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    count = len(rows)
    rows.clear()
    return count


def materialize_temporal_split(
    source: Path,
    destination: Path,
    cutoff_timestamp_ms: int,
    *,
    label_end_timestamp_ms: int | None = None,
    batch_rows: int = 250_000,
    overwrite: bool = False,
) -> TemporalMaterializationResult:
    """Persist leakage-safe matrix events, query contexts, and labels."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    if label_end_timestamp_ms is not None and label_end_timestamp_ms <= cutoff_timestamp_ms:
        raise ValueError("label_end_timestamp_ms must be greater than cutoff_timestamp_ms")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")

    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    matrix_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    source_sessions = 0
    matrix_events = 0
    query_sessions = 0
    query_context_events = 0
    label_count = 0

    try:
        with (
            pq.ParquetWriter(
                temporary / "matrix_events.parquet", EVENT_SCHEMA, compression="zstd"
            ) as matrix_writer,
            pq.ParquetWriter(
                temporary / "query_contexts.parquet", EVENT_SCHEMA, compression="zstd"
            ) as context_writer,
            pq.ParquetWriter(
                temporary / "labels.parquet", LABEL_SCHEMA, compression="zstd"
            ) as label_writer,
        ):
            for session in iter_parquet_sessions(source, batch_rows=batch_rows):
                source_sessions += 1
                matrix_rows.extend(
                    _event_row(session.session, event)
                    for event in session.events
                    if event.ts < cutoff_timestamp_ms
                )
                split = split_session(
                    session,
                    cutoff_timestamp_ms,
                    label_end_timestamp_ms=label_end_timestamp_ms,
                )
                has_labels = any(split.labels.values())
                if split.context is not None and has_labels:
                    query_sessions += 1
                    context_rows.extend(
                        _event_row(session.session, event) for event in split.context.events
                    )
                    for event_type, aids in split.labels.items():
                        label_rows.extend(
                            {"session": session.session, "type": event_type.value, "aid": aid}
                            for aid in aids
                        )
                if len(matrix_rows) >= batch_rows:
                    matrix_events += _flush(matrix_rows, matrix_writer, EVENT_SCHEMA)
                if len(context_rows) >= batch_rows:
                    query_context_events += _flush(context_rows, context_writer, EVENT_SCHEMA)
                if len(label_rows) >= batch_rows:
                    label_count += _flush(label_rows, label_writer, LABEL_SCHEMA)
            matrix_events += _flush(matrix_rows, matrix_writer, EVENT_SCHEMA)
            query_context_events += _flush(context_rows, context_writer, EVENT_SCHEMA)
            label_count += _flush(label_rows, label_writer, LABEL_SCHEMA)

        result = TemporalMaterializationResult(
            source_sessions=source_sessions,
            matrix_events=matrix_events,
            query_sessions=query_sessions,
            query_context_events=query_context_events,
            labels=label_count,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
            label_end_timestamp_ms=label_end_timestamp_ms,
        )
        source_stat = source.stat()
        manifest = {
            "stage": "temporal_split",
            "config_hash": stable_hash(
                {
                    "cutoff_timestamp_ms": cutoff_timestamp_ms,
                    "label_end_timestamp_ms": label_end_timestamp_ms,
                }
            ),
            "input": {
                "path": str(source),
                "size": source_stat.st_size,
                "modified_ns": source_stat.st_mtime_ns,
            },
            "result": asdict(result),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temporary), str(destination))
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def materialize_validation_views(
    source: Path,
    destination: Path,
    training_cutoff_timestamp_ms: int,
    validation_cutoff_timestamp_ms: int,
    *,
    batch_rows: int = 250_000,
    overwrite: bool = False,
) -> ValidationViewsResult:
    """Atomically publish ranker-training and local-validation temporal views."""
    if training_cutoff_timestamp_ms >= validation_cutoff_timestamp_ms:
        raise ValueError("training cutoff must be earlier than validation cutoff")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")

    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        ranker_train = materialize_temporal_split(
            source,
            temporary / "ranker_train",
            training_cutoff_timestamp_ms,
            label_end_timestamp_ms=validation_cutoff_timestamp_ms,
            batch_rows=batch_rows,
        )
        local_validation = materialize_temporal_split(
            source,
            temporary / "local_validation",
            validation_cutoff_timestamp_ms,
            batch_rows=batch_rows,
        )
        result = ValidationViewsResult(
            ranker_train=ranker_train,
            local_validation=local_validation,
        )
        manifest = {
            "stage": "validation_views",
            "config_hash": stable_hash(
                {
                    "training_cutoff_timestamp_ms": training_cutoff_timestamp_ms,
                    "validation_cutoff_timestamp_ms": validation_cutoff_timestamp_ms,
                }
            ),
            "input": str(source),
            "result": asdict(result),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temporary), str(destination))
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
