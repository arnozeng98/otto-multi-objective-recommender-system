from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
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


def _reduce_partition_task(task: tuple[str, Path, int]) -> bool:
    return _reduce_partition(*task)


def _reduce_fragments(
    fragments_dir: Path,
    matrices_dir: Path,
    partitions: int,
    max_neighbors: int,
    reduction_workers: int,
) -> int:
    reduction_tasks: list[tuple[str, Path, int]] = []
    for partition in range(partitions):
        directory = fragments_dir / f"partition={partition:03d}"
        if not directory.exists():
            continue
        output = matrices_dir / f"partition-{partition:03d}.parquet"
        reduction_tasks.append((str(directory / "*.parquet"), output, max_neighbors))
    if reduction_workers == 1:
        reduced = [_reduce_partition(*task) for task in reduction_tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=reduction_workers,
            mp_context=get_context("spawn"),
        ) as executor:
            reduced = list(executor.map(_reduce_partition_task, reduction_tasks))
    return sum(reduced)


def _publish_partitioned_rule(
    source: Path,
    work: Path,
    destination: Path,
    rule: CovisitationRule,
    *,
    sessions: int,
    pairs: int,
    partitions: int,
    max_events_per_session: int | None,
    reduction_workers: int,
) -> PartitionedBuildResult:
    output_files = _reduce_fragments(
        work / "fragments",
        work / "matrix",
        partitions,
        rule.max_neighbors,
        reduction_workers,
    )
    result = PartitionedBuildResult(
        sessions=sessions,
        pairs=pairs,
        partitions=partitions,
        output_files=output_files,
        rule=rule.name,
    )
    source_stat = source.stat()
    manifest = {
        "stage": "partitioned_covisitation",
        "config_hash": stable_hash(
            {
                "rule": asdict(rule),
                "partitions": partitions,
                "max_events_per_session": max_events_per_session,
                "reduction_workers": reduction_workers,
            }
        ),
        "input": {
            "path": str(source),
            "size": source_stat.st_size,
            "modified_ns": source_stat.st_mtime_ns,
        },
        "result": asdict(result),
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    shutil.rmtree(work / "fragments", ignore_errors=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(work), str(destination))
    return result


def build_partitioned_covisitation(
    source: Path,
    destination: Path,
    rule: CovisitationRule,
    *,
    partitions: int = 64,
    pair_buffer_size: int = 500_000,
    batch_rows: int = 250_000,
    max_events_per_session: int | None = None,
    reduction_workers: int = 1,
    overwrite: bool = False,
) -> PartitionedBuildResult:
    """Build a disk-partitioned top-K co-visitation matrix from flat event Parquet."""
    if partitions < 1 or pair_buffer_size < 1 or batch_rows < 1 or reduction_workers < 1:
        raise ValueError(
            "partitions, pair_buffer_size, batch_rows, and reduction_workers must be positive"
        )
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

        return _publish_partitioned_rule(
            source,
            temporary,
            destination,
            rule,
            sessions=sessions,
            pairs=total_pairs,
            partitions=partitions,
            max_events_per_session=max_events_per_session,
            reduction_workers=reduction_workers,
        )
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
    reduction_workers: int = 1,
    profile: str = "legacy_five",
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> CovisitationSuiteResult:
    """Build and atomically publish all default co-visitation matrices."""
    from otto_recsys.covisitation.rules import default_rules, public_v575_rules

    rule_profiles = {
        "legacy_five": default_rules,
        "public_v575": public_v575_rules,
    }
    if profile not in rule_profiles:
        available = ", ".join(sorted(rule_profiles))
        raise ValueError(f"Unknown co-visitation profile '{profile}'; expected one of {available}")
    rules = rule_profiles[profile](max_neighbors)

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    if overwrite and temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        matrices: dict[str, PartitionedBuildResult] = {}
        missing: dict[str, CovisitationRule] = {}
        for name, rule in rules.items():
            child = temporary / name
            if (child / "manifest.json").exists():
                payload = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
                matrices[name] = PartitionedBuildResult(**payload["result"])
                if progress is not None:
                    progress(1)
            else:
                missing[name] = rule

        if missing:
            work_dirs: dict[str, Path] = {}
            buffers: dict[str, list[list[tuple[int, int, float]]]] = {}
            fragment_counters: dict[str, list[int]] = {}
            pair_counts = {name: 0 for name in missing}
            sessions = 0
            buffered_pairs = 0
            for name in missing:
                child = temporary / name
                work = temporary / f"{name}.building"
                shutil.rmtree(child, ignore_errors=True)
                shutil.rmtree(work, ignore_errors=True)
                (work / "matrix").mkdir(parents=True)
                work_dirs[name] = work
                buffers[name] = [[] for _ in range(partitions)]
                fragment_counters[name] = [0] * partitions

            def flush_all() -> None:
                nonlocal buffered_pairs
                for rule_name in missing:
                    _flush_pairs(
                        buffers[rule_name],
                        work_dirs[rule_name] / "fragments",
                        fragment_counters[rule_name],
                    )
                buffered_pairs = 0

            for session in iter_parquet_sessions(source, batch_rows=batch_rows):
                sessions += 1
                if max_events_per_session is not None:
                    session = Session(session.session, session.events[-max_events_per_session:])
                for name, rule in missing.items():
                    for source_aid, target_aid, weight in iter_weighted_pairs(session, rule):
                        buffers[name][source_aid % partitions].append(
                            (source_aid, target_aid, weight)
                        )
                        pair_counts[name] += 1
                        buffered_pairs += 1
                if buffered_pairs >= pair_buffer_size:
                    flush_all()
            flush_all()

            for name, rule in missing.items():
                matrices[name] = _publish_partitioned_rule(
                    source,
                    work_dirs[name],
                    temporary / name,
                    rule,
                    sessions=sessions,
                    pairs=pair_counts[name],
                    partitions=partitions,
                    max_events_per_session=max_events_per_session,
                    reduction_workers=reduction_workers,
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
                    "profile": profile,
                    "reduction_workers": reduction_workers,
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
