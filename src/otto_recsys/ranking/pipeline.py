from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from otto_recsys.artifacts import stable_hash
from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.metrics import weighted_recall_at_k
from otto_recsys.ranking.xgboost_ranker import train_ranker

IDENTIFIER_COLUMNS = frozenset(("session", "target", "aid", "label"))


@dataclass(frozen=True, slots=True)
class RankerSuiteResult:
    training_rows: dict[str, int]
    validation_rows: dict[str, int]
    recall_at_20: dict[str, float]
    weighted_recall_at_20: float


def _read_target(
    path: Path,
    target: EventType,
    limit: int,
    *,
    keep_positives: bool,
) -> pl.DataFrame:
    frame = pl.scan_parquet(path / target.value / "*.parquet")
    if "candidate_rank" in frame.collect_schema().names():
        predicate = pl.col("candidate_rank") <= limit
        if keep_positives:
            predicate = predicate | (pl.col("label") > 0)
        frame = frame.filter(predicate)
    return frame.sort("session").collect(engine="streaming")


def _positive_groups(frame: pl.DataFrame) -> pl.DataFrame:
    sessions = (
        frame.group_by("session")
        .agg(pl.col("label").sum().alias("positives"))
        .filter(pl.col("positives") > 0)
        .select("session")
    )
    return frame.join(sessions, on="session", how="inner").sort("session")


def _group_sizes(frame: pl.DataFrame) -> list[int]:
    return frame.group_by("session", maintain_order=True).len()["len"].to_list()


def _ground_truth(path: Path) -> dict[tuple[int, EventType], list[int]]:
    result: dict[tuple[int, EventType], list[int]] = {}
    table = pq.read_table(path, columns=["session", "type", "aid"])
    columns = table.to_pydict()
    for session, event_type, aid in zip(
        columns["session"],
        columns["type"],
        columns["aid"],
        strict=True,
    ):
        result.setdefault((int(session), EventType(event_type)), []).append(int(aid))
    return result


def train_ranker_suite(
    training_candidates: Path,
    validation_candidates: Path,
    validation_labels: Path,
    destination: Path,
    *,
    device: str = "cpu",
    rounds: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.08,
    seed: int = 2026,
    early_stopping_rounds: int = 30,
    training_candidate_limit: int = 80,
    validation_candidate_limit: int = 120,
    overwrite: bool = False,
) -> RankerSuiteResult:
    """Train one LambdaMART model per target and score local validation candidates."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    training_rows: dict[str, int] = {}
    validation_rows: dict[str, int] = {}
    predictions: dict[tuple[int, EventType], list[int]] = {}
    try:
        for target in EVENT_TYPES:
            training = _positive_groups(
                _read_target(
                    training_candidates,
                    target,
                    training_candidate_limit,
                    keep_positives=True,
                )
            )
            validation = _read_target(
                validation_candidates,
                target,
                validation_candidate_limit,
                keep_positives=False,
            )
            if training.is_empty() or validation.is_empty():
                raise ValueError(f"Target {target.value} has no usable ranking rows")
            feature_names = tuple(
                column for column in training.columns if column not in IDENTIFIER_COLUMNS
            )
            if (
                tuple(column for column in validation.columns if column not in IDENTIFIER_COLUMNS)
                != feature_names
            ):
                raise ValueError(f"Feature contract differs for target {target.value}")
            training_features = training.select(feature_names).to_numpy().astype(np.float32)
            training_labels = training["label"].to_numpy().astype(np.float32)
            validation_features = validation.select(feature_names).to_numpy().astype(np.float32)
            validation_target = validation["label"].to_numpy().astype(np.float32)
            model = train_ranker(
                training_features,
                training_labels,
                _group_sizes(training),
                feature_names,
                validation_features=validation_features,
                validation_labels=validation_target,
                validation_group_sizes=_group_sizes(validation),
                device=device,
                rounds=rounds,
                seed=seed,
                max_depth=max_depth,
                learning_rate=learning_rate,
                early_stopping_rounds=early_stopping_rounds,
            )
            model.save(temporary / f"{target.value}.json")
            scores = model.predict(validation_features).astype(np.float32)
            scored = validation.select("session", "target", "aid", "label").with_columns(
                pl.Series("score", scores)
            )
            scored.write_parquet(temporary / f"{target.value}-predictions.parquet")
            ranked = scored.sort(
                ["session", "score", "aid"],
                descending=[False, True, False],
            )
            for session_frame in ranked.partition_by("session", maintain_order=True):
                session_id = int(session_frame["session"][0])
                predictions[(session_id, target)] = session_frame["aid"].head(20).to_list()
            training_rows[target.value] = len(training)
            validation_rows[target.value] = len(validation)

        recall = weighted_recall_at_k(predictions, _ground_truth(validation_labels), k=20)
        result = RankerSuiteResult(
            training_rows=training_rows,
            validation_rows=validation_rows,
            recall_at_20={target.value: recall.per_type[target] for target in EVENT_TYPES},
            weighted_recall_at_20=recall.weighted,
        )
        manifest = {
            "stage": "ranker_suite",
            "config_hash": stable_hash(
                {
                    "device": device,
                    "rounds": rounds,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "training_candidate_limit": training_candidate_limit,
                    "validation_candidate_limit": validation_candidate_limit,
                }
            ),
            "inputs": {
                "training_candidates": str(training_candidates),
                "validation_candidates": str(validation_candidates),
                "validation_labels": str(validation_labels),
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
