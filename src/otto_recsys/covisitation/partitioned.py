from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from otto_recsys.artifacts import stable_hash
from otto_recsys.constants import EventType
from otto_recsys.covisitation.builder import CovisitationRule, iter_weighted_pairs
from otto_recsys.data.schemas import Event, Session


@dataclass(frozen=True, slots=True)
class PartitionedBuildResult:
    sessions: int
    pairs: int
    partitions: int
    output_files: int
    rule: str


@dataclass(frozen=True, slots=True)
class CovisitationSuiteResult:
    matrices: dict[str, PartitionedBuildResult]


def iter_parquet_sessions(path: Path, *, batch_rows: int = 250_000) -> Iterator[Session]:
    """Stream ordered sessions from flat Parquet while preserving batch boundaries."""
    parquet = pq.ParquetFile(path)
    current_id: int | None = None
    current_events: list[Event] = []
    for batch in parquet.iter_batches(
        batch_size=batch_rows,
        columns=["session", "aid", "ts", "type"],
    ):
        columns = batch.to_pydict()
        for session_id, aid, timestamp, event_type in zip(
            columns["session"],
            columns["aid"],
            columns["ts"],
            columns["type"],
            strict=True,
        ):
            session_id = int(session_id)
            if current_id is not None and session_id != current_id:
                yield Session(current_id, tuple(current_events))
                current_events.clear()
            current_id = session_id
            current_events.append(Event(int(aid), int(timestamp), EventType(event_type)))
    if current_id is not None:
        yield Session(current_id, tuple(current_events))


def _flush_pairs(
    buffers: list[list[tuple[int, int, float]]],
    fragments_dir: Path,
    counters: list[int],
) -> None:
    for partition, rows in enumerate(buffers):
        if not rows:
            continue
        directory = fragments_dir / f"partition={partition:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"part-{counters[partition]:06d}.parquet"
        pl.DataFrame(
            rows,
            schema={"source_aid": pl.Int32, "target_aid": pl.Int32, "weight": pl.Float32},
            orient="row",
        ).write_parquet(path, compression="zstd")
        counters[partition] += 1
        rows.clear()


def _reduce_partition(fragment_glob: str, destination: Path, max_neighbors: int) -> bool:
    reduced = (
        pl.scan_parquet(fragment_glob)
        .group_by("source_aid", "target_aid")
        .agg(pl.col("weight").sum())
        .sort(["source_aid", "weight", "target_aid"], descending=[False, True, False])
        .with_columns(pl.col("source_aid").cum_count().over("source_aid").alias("rank"))
        .filter(pl.col("rank") <= max_neighbors)
        .collect(engine="streaming")
    )
    if reduced.is_empty():
        return False
    reduced.write_parquet(destination, compression="zstd", statistics=True)
    return True


def build_partitioned_covisitation(
    source: Path,
    destination: Path,
    rule: CovisitationRule,
    *,
    partitions: int = 64,
    pair_buffer_size: int = 500_000,
    batch_rows: int = 250_000,
    max_events_per_session: int | None = None,
    overwrite: bool = False,
) -> PartitionedBuildResult:
    """Build a disk-partitioned top-K co-visitation matrix from flat event Parquet."""
    if partitions < 1 or pair_buffer_size < 1 or batch_rows < 1:
        raise ValueError("partitions, pair_buffer_size, and batch_rows must be positive")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")

    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    fragments_dir = temporary / "fragments"
    matrices_dir = temporary / "matrix"
    matrices_dir.mkdir(parents=True)
    buffers: list[list[tuple[int, int, float]]] = [[] for _ in range(partitions)]
    fragment_counters = [0] * partitions
    buffered_pairs = 0
    total_pairs = 0
    sessions = 0

    try:
        for session in iter_parquet_sessions(source, batch_rows=batch_rows):
            sessions += 1
            if max_events_per_session is not None:
                session = Session(session.session, session.events[-max_events_per_session:])
            for source_aid, target_aid, weight in iter_weighted_pairs(session, rule):
                buffers[source_aid % partitions].append((source_aid, target_aid, weight))
                buffered_pairs += 1
                total_pairs += 1
            if buffered_pairs >= pair_buffer_size:
                _flush_pairs(buffers, fragments_dir, fragment_counters)
                buffered_pairs = 0
        _flush_pairs(buffers, fragments_dir, fragment_counters)

        output_files = 0
        for partition in range(partitions):
            directory = fragments_dir / f"partition={partition:03d}"
            if not directory.exists():
                continue
            output = matrices_dir / f"partition-{partition:03d}.parquet"
            if _reduce_partition(str(directory / "*.parquet"), output, rule.max_neighbors):
                output_files += 1

        source_stat = source.stat()
        result = PartitionedBuildResult(
            sessions=sessions,
            pairs=total_pairs,
            partitions=partitions,
            output_files=output_files,
            rule=rule.name,
        )
        manifest = {
            "stage": "partitioned_covisitation",
            "config_hash": stable_hash(
                {
                    "rule": asdict(rule),
                    "partitions": partitions,
                    "max_events_per_session": max_events_per_session,
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
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        shutil.rmtree(fragments_dir, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temporary), str(destination))
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_covisitation_suite(
    source: Path,
    destination: Path,
    *,
    max_neighbors: int = 80,
    partitions: int = 64,
    pair_buffer_size: int = 500_000,
    batch_rows: int = 250_000,
    max_events_per_session: int | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> CovisitationSuiteResult:
    """Build and atomically publish all default co-visitation matrices."""
    from otto_recsys.covisitation.rules import default_rules

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    if overwrite and temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        matrices: dict[str, PartitionedBuildResult] = {}
        for name, rule in default_rules(max_neighbors).items():
            child = temporary / name
            if (child / "manifest.json").exists():
                payload = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
                matrices[name] = PartitionedBuildResult(**payload["result"])
            else:
                shutil.rmtree(child, ignore_errors=True)
                matrices[name] = build_partitioned_covisitation(
                    source,
                    child,
                    rule,
                    partitions=partitions,
                    pair_buffer_size=pair_buffer_size,
                    batch_rows=batch_rows,
                    max_events_per_session=max_events_per_session,
                )
            if progress is not None:
                progress(1)
        result = CovisitationSuiteResult(matrices=matrices)
        source_stat = source.stat()
        manifest = {
            "stage": "covisitation_suite",
            "config_hash": stable_hash(
                {
                    "max_neighbors": max_neighbors,
                    "partitions": partitions,
                    "pair_buffer_size": pair_buffer_size,
                    "batch_rows": batch_rows,
                    "max_events_per_session": max_events_per_session,
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
        raise
