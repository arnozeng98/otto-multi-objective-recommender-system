from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    pl.DataFrame(rows).write_parquet(
        directory / f"part-{part:06d}.parquet",
        compression="zstd",
        statistics=True,
    )
    rows.clear()
    return part + 1


def materialize_candidates(
    view: Path,
    stores: Path,
    destination: Path,
    *,
    budget: int,
    popularity_budget: int,
    batch_rows: int = 250_000,
    overwrite: bool = False,
) -> CandidateMaterializationResult:
    """Persist labeled target-aware candidates and context-only features."""
    if budget < 1 or popularity_budget < 0 or batch_rows < 1:
        raise ValueError(
            "budget and batch_rows must be positive; popularity_budget cannot be negative"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    buffers: dict[EventType, list[dict[str, Any]]] = {target: [] for target in EVENT_TYPES}
    parts = {target: 0 for target in EVENT_TYPES}
    counts = {target.value: 0 for target in EVENT_TYPES}
    positives = {target.value: 0 for target in EVENT_TYPES}
    for target in EVENT_TYPES:
        (temporary / target.value).mkdir()

    try:
        source_names = sorted({name for names in TARGET_MATRIX_SOURCES.values() for name in names})
        matrices = {name: MatrixStore(stores / name) for name in source_names}
        popularity = _popular_aids(view / "matrix_events.parquet", popularity_budget)
        labels = _load_labels(view / "labels.parquet")
        sessions = 0
        for session in iter_parquet_sessions(
            view / "query_contexts.parquet",
            batch_rows=batch_rows,
        ):
            sessions += 1
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
                if len(buffers[target]) >= batch_rows:
                    parts[target] = _flush_rows(
                        buffers[target],
                        temporary / target.value,
                        parts[target],
                    )
        for target in EVENT_TYPES:
            parts[target] = _flush_rows(
                buffers[target],
                temporary / target.value,
                parts[target],
            )
        result = CandidateMaterializationResult(
            sessions=sessions,
            candidates=counts,
            positives=positives,
        )
        manifest = {
            "stage": "candidates",
            "config_hash": stable_hash(
                {
                    "budget": budget,
                    "popularity_budget": popularity_budget,
                }
            ),
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
        shutil.rmtree(temporary, ignore_errors=True)
        raise
