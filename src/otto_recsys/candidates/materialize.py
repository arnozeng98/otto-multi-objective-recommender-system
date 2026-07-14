from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow.parquet as pq

from otto_recsys.artifacts import stable_hash
from otto_recsys.candidates.targeted import TARGET_MATRIX_SOURCES, target_candidates
from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.covisitation import MatrixStore
from otto_recsys.covisitation.partitioned import iter_parquet_sessions
from otto_recsys.features import build_candidate_features


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


def _popular_aids(matrix_events: Path, budget: int) -> dict[EventType, tuple[int, ...]]:
    counts = (
        pl.scan_parquet(matrix_events)
        .group_by("type", "aid")
        .len(name="count")
        .sort(["type", "count", "aid"], descending=[False, True, False])
        .collect(engine="streaming")
    )
    overall = (
        counts.group_by("aid")
        .agg(pl.col("count").sum())
        .sort(["count", "aid"], descending=[True, False])["aid"]
        .to_list()
    )
    result: dict[EventType, tuple[int, ...]] = {}
    for target in EVENT_TYPES:
        target_aids = counts.filter(pl.col("type") == target.value)["aid"].to_list()
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
    batch_rows: int = 250_000,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> CandidateMaterializationResult:
    """Persist labeled target-aware candidates and context-only features."""
    if budget < 1 or popularity_budget < 0 or batch_rows < 1:
        raise ValueError(
            "budget and batch_rows must be positive; popularity_budget cannot be negative"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    if overwrite and temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    config_hash = stable_hash(
        {
            "budget": budget,
            "popularity_budget": popularity_budget,
            "view": str(view.resolve()),
            "stores": str(stores.resolve()),
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
        source_names = sorted({name for names in TARGET_MATRIX_SOURCES.values() for name in names})
        matrices = {name: MatrixStore(stores / name) for name in source_names}
        popularity = _popular_aids(view / "matrix_events.parquet", popularity_budget)
        labels = _load_labels(view / "labels.parquet")
        sessions = checkpoint.sessions if checkpoint is not None else 0
        last_session = checkpoint.last_session if checkpoint is not None else -1
        chunk_sessions = 0
        for session in iter_parquet_sessions(
            view / "query_contexts.parquet",
            batch_rows=batch_rows,
        ):
            if session.session <= last_session:
                continue
            sessions += 1
            chunk_sessions += 1
            for target in EVENT_TYPES:
                candidates = target_candidates(
                    session,
                    target,
                    matrices,
                    popularity[target],
                    budget=budget,
                )
                true_aids = labels.get((session.session, target), set())
                for candidate_rank, candidate in enumerate(candidates, start=1):
                    label = int(candidate.aid in true_aids)
                    row: dict[str, Any] = {
                        "session": session.session,
                        "target": target.value,
                        "aid": candidate.aid,
                        "label": label,
                        "candidate_rank": candidate_rank,
                        **build_candidate_features(session, candidate),
                    }
                    buffers[target].append(row)
                    counts[target.value] += 1
                    positives[target.value] += label
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
                    progress(chunk_sessions)
                chunk_sessions = 0
        for target in EVENT_TYPES:
            parts[target] = _flush_rows(
                buffers[target],
                temporary / target.value,
                parts[target],
            )
        if chunk_sessions:
            if progress is not None:
                progress(chunk_sessions)
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
            "inputs": {"view": str(view), "stores": str(stores)},
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
