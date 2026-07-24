from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow.parquet as pq

from otto_recsys.artifacts import stable_hash
from otto_recsys.candidates.targeted import TARGET_MATRIX_SOURCES, target_candidates
from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.covisitation import MatrixStore
from otto_recsys.covisitation.partitioned import iter_parquet_sessions
from otto_recsys.data.schemas import Session
from otto_recsys.features import CandidateFeatureContext, build_candidate_features


@dataclass(frozen=True, slots=True)
class CandidateMaterializationResult:
    sessions: int
    candidates: dict[str, int]
    positives: dict[str, int]


@dataclass(frozen=True, slots=True)
class CandidateCheckpoint:
    version: int
    config_hash: str
    last_session: int
    sessions: int
    next_parts: dict[str, int]
    candidates: dict[str, int]
    positives: dict[str, int]


@dataclass(frozen=True, slots=True)
class CandidateTaskSettings:
    budget: int
    history_budget: int | None
    covisitation_budget: int | None
    max_events_per_session: int | None
    retention_limit: int | None


@dataclass(frozen=True, slots=True)
class CandidateTaskResult:
    task_id: int
    sessions: int
    last_session: int
    candidates: dict[str, int]
    positives: dict[str, int]


_WORKER_MATRICES: dict[str, MatrixStore] | None = None
_WORKER_POPULARITY: dict[EventType, tuple[int, ...]] | None = None
_WORKER_SETTINGS: CandidateTaskSettings | None = None


def _initialize_candidate_worker(
    stores: Path,
    source_names: tuple[str, ...],
    popularity: dict[EventType, tuple[int, ...]],
    settings: CandidateTaskSettings,
) -> None:
    global _WORKER_MATRICES, _WORKER_POPULARITY, _WORKER_SETTINGS
    _WORKER_MATRICES = {name: MatrixStore(stores / name) for name in source_names}
    _WORKER_POPULARITY = popularity
    _WORKER_SETTINGS = settings


def _popular_aids(matrix_events: Path, budget: int) -> dict[EventType, tuple[int, ...]]:
    counts = pl.scan_parquet(matrix_events).group_by("type", "aid").len(name="count")
    overall = (
        counts.group_by("aid")
        .agg(pl.col("count").sum())
        .sort(["count", "aid"], descending=[True, False])
        .select("aid")
        .head(budget)
        .collect(engine="streaming")["aid"]
        .to_list()
    )
    by_type = (
        counts.sort(["type", "count", "aid"], descending=[False, True, False])
        .group_by("type", maintain_order=True)
        .head(budget)
        .collect(engine="streaming")
    )
    result: dict[EventType, tuple[int, ...]] = {}
    for target in EVENT_TYPES:
        target_aids = by_type.filter(pl.col("type") == target.value)["aid"].to_list()
        result[target] = tuple(dict.fromkeys([*target_aids, *overall]))[:budget]
    return result


def _load_labels(path: Path) -> dict[tuple[int, EventType], set[int]]:
    labels: dict[tuple[int, EventType], set[int]] = defaultdict(set)
    table = pq.read_table(path, columns=["session", "type", "aid"])
    columns = table.to_pydict()
    for session, event_type, aid in zip(
        columns["session"],
        columns["type"],
        columns["aid"],
        strict=True,
    ):
        labels[(int(session), EventType(event_type))].add(int(aid))
    return labels


def _iter_session_labels(path: Path) -> Iterator[tuple[int, dict[EventType, set[int]]]]:
    current_session: int | None = None
    current: dict[EventType, set[int]] = defaultdict(set)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["session", "type", "aid"]):
        columns = batch.to_pydict()
        for session, event_type, aid in zip(
            columns["session"], columns["type"], columns["aid"], strict=True
        ):
            session = int(session)
            if current_session is not None and session != current_session:
                yield current_session, dict(current)
                current.clear()
            current_session = session
            current[EventType(event_type)].add(int(aid))
    if current_session is not None:
        yield current_session, dict(current)


def _flush_rows(rows: list[dict[str, Any]], directory: Path, part: int) -> int:
    if not rows:
        return part
    destination = directory / f"part-{part:06d}.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    pl.DataFrame(rows).write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )
    os.replace(temporary, destination)
    rows.clear()
    return part + 1


def _candidate_rows_for_session(
    session: Session,
    matrices: dict[str, MatrixStore],
    popularity: dict[EventType, tuple[int, ...]],
    labels: dict[tuple[int, EventType], set[int]],
    settings: CandidateTaskSettings,
) -> tuple[dict[EventType, list[dict[str, Any]]], dict[str, int]]:
    rows: dict[EventType, list[dict[str, Any]]] = {target: [] for target in EVENT_TYPES}
    positives = {target.value: 0 for target in EVENT_TYPES}
    feature_context = CandidateFeatureContext.from_session(session)
    for target in EVENT_TYPES:
        candidates = target_candidates(
            session,
            target,
            matrices,
            popularity[target],
            budget=settings.budget,
            history_budget=settings.history_budget,
            covisitation_budget=settings.covisitation_budget,
            max_events_per_session=settings.max_events_per_session,
        )
        true_aids = labels.get((session.session, target), set())
        for candidate_rank, candidate in enumerate(candidates, start=1):
            label = int(candidate.aid in true_aids)
            if (
                settings.retention_limit is not None
                and candidate_rank > settings.retention_limit
                and not label
            ):
                continue
            rows[target].append(
                {
                    "session": session.session,
                    "target": target.value,
                    "aid": candidate.aid,
                    "label": label,
                    "candidate_rank": candidate_rank,
                    **build_candidate_features(session, candidate, context=feature_context),
                }
            )
            positives[target.value] += label
    return rows, positives


def _write_candidate_task(
    task_id: int,
    sessions: tuple[Session, ...],
    labels: dict[tuple[int, EventType], set[int]],
    temporary: Path,
) -> CandidateTaskResult:
    if _WORKER_MATRICES is None or _WORKER_POPULARITY is None or _WORKER_SETTINGS is None:
        raise RuntimeError("Candidate worker was not initialized")
    buffers: dict[EventType, list[dict[str, Any]]] = {target: [] for target in EVENT_TYPES}
    positives = {target.value: 0 for target in EVENT_TYPES}
    for session in sessions:
        session_rows, session_positives = _candidate_rows_for_session(
            session,
            _WORKER_MATRICES,
            _WORKER_POPULARITY,
            labels,
            _WORKER_SETTINGS,
        )
        for target in EVENT_TYPES:
            buffers[target].extend(session_rows[target])
            positives[target.value] += session_positives[target.value]
    counts = {target.value: len(buffers[target]) for target in EVENT_TYPES}
    for target in EVENT_TYPES:
        _flush_rows(buffers[target], temporary / target.value, task_id)
    return CandidateTaskResult(
        task_id=task_id,
        sessions=len(sessions),
        last_session=sessions[-1].session,
        candidates=counts,
        positives=positives,
    )


def _write_checkpoint(path: Path, checkpoint: CandidateCheckpoint) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(asdict(checkpoint), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_part(path: Path) -> pl.DataFrame | None:
    try:
        return pl.read_parquet(path, columns=["session"])
    except Exception:
        return None


def _recover_legacy_checkpoint(
    temporary: Path,
    config_hash: str,
) -> CandidateCheckpoint | None:
    target_files: dict[EventType, list[tuple[Path, int, int]]] = {}
    for target in EVENT_TYPES:
        valid: list[tuple[Path, int, int]] = []
        invalid_tail = False
        for path in sorted((temporary / target.value).glob("part-*.parquet")):
            frame = None if invalid_tail else _read_part(path)
            if frame is None or frame.is_empty():
                invalid_tail = True
                path.unlink(missing_ok=True)
                continue
            minimum = cast(int, frame.select(pl.col("session").min()).item())
            maximum = cast(int, frame.select(pl.col("session").max()).item())
            valid.append((path, minimum, maximum))
        target_files[target] = valid
    if not all(target_files.values()):
        return None

    last_session = min(files[-1][2] for files in target_files.values())
    for files in target_files.values():
        for path, minimum, maximum in files:
            if minimum > last_session:
                path.unlink(missing_ok=True)
            elif maximum > last_session:
                frame = pl.read_parquet(path).filter(pl.col("session") <= last_session)
                if frame.is_empty():
                    path.unlink(missing_ok=True)
                else:
                    replacement = path.with_suffix(".parquet.tmp")
                    frame.write_parquet(replacement, compression="zstd", statistics=True)
                    os.replace(replacement, path)

    counts: dict[str, int] = {}
    positives: dict[str, int] = {}
    next_parts: dict[str, int] = {}
    for target in EVENT_TYPES:
        paths = sorted((temporary / target.value).glob("part-*.parquet"))
        lazy_frame = pl.scan_parquet(paths)
        summary = lazy_frame.select(
            pl.len().alias("rows"),
            pl.col("label").sum().alias("positives"),
        ).collect(engine="streaming")
        counts[target.value] = int(summary["rows"][0])
        positives[target.value] = int(summary["positives"][0])
        next_parts[target.value] = len(paths)
    sessions = int(
        pl.scan_parquet(sorted((temporary / EventType.CLICKS.value).glob("part-*.parquet")))
        .select(pl.col("session").n_unique())
        .collect(engine="streaming")["session"][0]
    )
    checkpoint = CandidateCheckpoint(
        version=1,
        config_hash=config_hash,
        last_session=last_session,
        sessions=sessions,
        next_parts=next_parts,
        candidates=counts,
        positives=positives,
    )
    _write_checkpoint(temporary / "checkpoint.json", checkpoint)
    return checkpoint


def _load_checkpoint(temporary: Path, config_hash: str) -> CandidateCheckpoint | None:
    path = temporary / "checkpoint.json"
    if not path.exists():
        return _recover_legacy_checkpoint(temporary, config_hash)
    checkpoint = CandidateCheckpoint(**json.loads(path.read_text(encoding="utf-8")))
    if checkpoint.version != 1 or checkpoint.config_hash != config_hash:
        raise ValueError("Candidate checkpoint is incompatible; use --overwrite")
    for target in EVENT_TYPES:
        next_part = checkpoint.next_parts[target.value]
        for candidate in (temporary / target.value).glob("part-*.parquet"):
            part = int(candidate.stem.removeprefix("part-"))
            if part >= next_part:
                candidate.unlink()
    return checkpoint


def materialize_candidates(
    view: Path,
    stores: Path,
    destination: Path,
    *,
    budget: int,
    popularity_budget: int,
    history_budget: int | None = None,
    covisitation_budget: int | None = None,
    max_events_per_session: int | None = None,
    query_contexts: Path | None = None,
    popularity_events: Path | None = None,
    include_labels: bool = True,
    retention_limit: int | None = None,
    workers: int = 1,
    chunk_sessions: int = 5_000,
    batch_rows: int = 250_000,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> CandidateMaterializationResult:
    """Persist target-aware candidates and context-only features."""
    if (
        budget < 1
        or popularity_budget < 0
        or history_budget is not None
        and history_budget < 0
        or covisitation_budget is not None
        and covisitation_budget < 0
        or max_events_per_session is not None
        and max_events_per_session < 1
        or retention_limit is not None
        and retention_limit < 1
        or workers < 1
        or chunk_sessions < 1
        or batch_rows < 1
    ):
        raise ValueError(
            "budgets cannot be negative; budget, batch_rows, and max_events_per_session "
            "must be positive"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    if overwrite and temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    query_contexts = query_contexts or view / "query_contexts.parquet"
    popularity_events = popularity_events or view / "matrix_events.parquet"
    config_hash = stable_hash(
        {
            "budget": budget,
            "popularity_budget": popularity_budget,
            "history_budget": history_budget,
            "covisitation_budget": covisitation_budget,
            "max_events_per_session": max_events_per_session,
            "view": str(view.resolve()),
            "stores": str(stores.resolve()),
            "query_contexts": str(query_contexts.resolve()),
            "popularity_events": str(popularity_events.resolve()),
            "include_labels": include_labels,
            "retention_limit": retention_limit,
            "workers": workers,
            "chunk_sessions": chunk_sessions,
        }
    )
    checkpoint = _load_checkpoint(temporary, config_hash)
    if checkpoint is not None and progress is not None:
        progress(checkpoint.sessions)
    buffers: dict[EventType, list[dict[str, Any]]] = {target: [] for target in EVENT_TYPES}
    parts = {
        target: checkpoint.next_parts[target.value] if checkpoint is not None else 0
        for target in EVENT_TYPES
    }
    counts = {
        target.value: checkpoint.candidates[target.value] if checkpoint is not None else 0
        for target in EVENT_TYPES
    }
    positives = {
        target.value: checkpoint.positives[target.value] if checkpoint is not None else 0
        for target in EVENT_TYPES
    }
    for target in EVENT_TYPES:
        (temporary / target.value).mkdir(exist_ok=True)

    try:
        source_names = sorted(
            name
            for name in {name for names in TARGET_MATRIX_SOURCES.values() for name in names}
            if (stores / name / "manifest.json").exists()
        )
        matrices = {name: MatrixStore(stores / name) for name in source_names}
        popularity = _popular_aids(popularity_events, popularity_budget)
        (temporary / "popularity.json").write_text(
            json.dumps(
                {target.value: list(popularity[target]) for target in EVENT_TYPES},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        labels = _load_labels(view / "labels.parquet") if include_labels and workers == 1 else {}
        sessions = checkpoint.sessions if checkpoint is not None else 0
        last_session = checkpoint.last_session if checkpoint is not None else -1
        settings = CandidateTaskSettings(
            budget=budget,
            history_budget=history_budget,
            covisitation_budget=covisitation_budget,
            max_events_per_session=max_events_per_session,
            retention_limit=retention_limit,
        )
        if workers > 1:
            if len(set(parts.values())) != 1:
                raise ValueError("Parallel candidate checkpoints require synchronized target parts")
            next_task = next(iter(parts.values()))
            next_commit = next_task
            completed: dict[int, CandidateTaskResult] = {}
            pending: dict[Future[CandidateTaskResult], int] = {}
            label_iterator = (
                iter(_iter_session_labels(view / "labels.parquet")) if include_labels else iter(())
            )
            next_labels = next(label_iterator, None)

            def labels_for_chunk(
                session_chunk: list[Session],
            ) -> dict[tuple[int, EventType], set[int]]:
                nonlocal next_labels
                first_session = session_chunk[0].session
                last_chunk_session = session_chunk[-1].session
                chunk_labels: dict[tuple[int, EventType], set[int]] = {}
                while next_labels is not None and next_labels[0] <= last_chunk_session:
                    label_session, target_labels = next_labels
                    if label_session >= first_session:
                        for target, aids in target_labels.items():
                            chunk_labels[(label_session, target)] = aids
                    next_labels = next(label_iterator, None)
                return chunk_labels

            def commit_ready() -> None:
                nonlocal last_session, next_commit, sessions
                while next_commit in completed:
                    result = completed.pop(next_commit)
                    sessions += result.sessions
                    last_session = result.last_session
                    for target in EVENT_TYPES:
                        counts[target.value] += result.candidates[target.value]
                        positives[target.value] += result.positives[target.value]
                        parts[target] = next_commit + 1
                    _write_checkpoint(
                        temporary / "checkpoint.json",
                        CandidateCheckpoint(
                            version=1,
                            config_hash=config_hash,
                            last_session=last_session,
                            sessions=sessions,
                            next_parts={target.value: parts[target] for target in EVENT_TYPES},
                            candidates=counts,
                            positives=positives,
                        ),
                    )
                    if progress is not None:
                        progress(result.sessions)
                    next_commit += 1

            def collect_one() -> None:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    task_id = pending.pop(future)
                    result = future.result()
                    if result.task_id != task_id:
                        raise RuntimeError("Candidate worker returned a mismatched task ID")
                    completed[task_id] = result
                commit_ready()

            source_names_tuple = tuple(source_names)
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_candidate_worker,
                initargs=(stores, source_names_tuple, popularity, settings),
            ) as executor:
                chunk: list[Session] = []
                for session in iter_parquet_sessions(query_contexts, batch_rows=batch_rows):
                    if session.session <= last_session:
                        continue
                    chunk.append(session)
                    if len(chunk) < chunk_sessions:
                        continue
                    future = executor.submit(
                        _write_candidate_task,
                        next_task,
                        tuple(chunk),
                        labels_for_chunk(chunk),
                        temporary,
                    )
                    pending[future] = next_task
                    next_task += 1
                    chunk.clear()
                    if len(pending) >= workers * 2:
                        collect_one()
                if chunk:
                    future = executor.submit(
                        _write_candidate_task,
                        next_task,
                        tuple(chunk),
                        labels_for_chunk(chunk),
                        temporary,
                    )
                    pending[future] = next_task
                while pending:
                    collect_one()
            if completed:
                raise RuntimeError("Candidate tasks completed with a non-contiguous gap")
            result = CandidateMaterializationResult(
                sessions=sessions,
                candidates=counts,
                positives=positives,
            )
            manifest = {
                "stage": "candidates",
                "config_hash": config_hash,
                "inputs": {
                    "view": str(view),
                    "stores": str(stores),
                    "query_contexts": str(query_contexts),
                    "popularity_events": str(popularity_events),
                    "include_labels": include_labels,
                    "retention_limit": retention_limit,
                    "workers": workers,
                    "chunk_sessions": chunk_sessions,
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

        processed_since_flush = 0
        for session in iter_parquet_sessions(
            query_contexts,
            batch_rows=batch_rows,
        ):
            if session.session <= last_session:
                continue
            sessions += 1
            processed_since_flush += 1
            session_rows, session_positives = _candidate_rows_for_session(
                session,
                matrices,
                popularity,
                labels,
                settings,
            )
            for target in EVENT_TYPES:
                buffers[target].extend(session_rows[target])
                counts[target.value] += len(session_rows[target])
                positives[target.value] += session_positives[target.value]
            if max(len(buffer) for buffer in buffers.values()) >= batch_rows:
                for target in EVENT_TYPES:
                    parts[target] = _flush_rows(
                        buffers[target], temporary / target.value, parts[target]
                    )
                last_session = session.session
                _write_checkpoint(
                    temporary / "checkpoint.json",
                    CandidateCheckpoint(
                        version=1,
                        config_hash=config_hash,
                        last_session=last_session,
                        sessions=sessions,
                        next_parts={target.value: parts[target] for target in EVENT_TYPES},
                        candidates=counts,
                        positives=positives,
                    ),
                )
                if progress is not None:
                    progress(processed_since_flush)
                processed_since_flush = 0
        for target in EVENT_TYPES:
            parts[target] = _flush_rows(
                buffers[target],
                temporary / target.value,
                parts[target],
            )
        if processed_since_flush:
            if progress is not None:
                progress(processed_since_flush)
            last_session = session.session
        _write_checkpoint(
            temporary / "checkpoint.json",
            CandidateCheckpoint(
                version=1,
                config_hash=config_hash,
                last_session=last_session,
                sessions=sessions,
                next_parts={target.value: parts[target] for target in EVENT_TYPES},
                candidates=counts,
                positives=positives,
            ),
        )
        result = CandidateMaterializationResult(
            sessions=sessions,
            candidates=counts,
            positives=positives,
        )
        manifest = {
            "stage": "candidates",
            "config_hash": config_hash,
            "inputs": {
                "view": str(view),
                "stores": str(stores),
                "query_contexts": str(query_contexts),
                "popularity_events": str(popularity_events),
                "include_labels": include_labels,
                "retention_limit": retention_limit,
                "workers": workers,
                "chunk_sessions": chunk_sessions,
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
