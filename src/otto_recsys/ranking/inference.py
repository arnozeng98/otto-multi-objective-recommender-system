from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from otto_recsys.artifacts import stable_hash
from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.ranking.pipeline import IDENTIFIER_COLUMNS
from otto_recsys.ranking.xgboost_ranker import RankerModel


@dataclass(frozen=True, slots=True)
class InferenceResult:
    sessions: dict[str, int]
    predictions: dict[str, int]


def _score_part(
    source: Path,
    destination: Path,
    model: RankerModel,
    target: EventType,
    device: str,
) -> tuple[int, int]:
    frame = pl.read_parquet(source)
    feature_names = tuple(column for column in frame.columns if column not in IDENTIFIER_COLUMNS)
    if feature_names != model.feature_names:
        raise ValueError(f"Feature contract differs for target {target.value}")
    features = frame.select(feature_names).to_numpy().astype(np.float32)
    scores = model.predict(features, device=device).astype(np.float32)
    ranked = (
        frame.select("session", "aid")
        .with_columns(pl.Series("score", scores))
        .sort(["session", "score", "aid"], descending=[False, True, False])
        .with_columns(pl.col("session").cum_count().over("session").alias("rank"))
        .filter(pl.col("rank") <= 20)
        .with_columns(pl.lit(target.value).alias("target"))
        .select("session", "target", "aid", "score", "rank")
    )
    temporary = destination.with_suffix(".parquet.tmp")
    ranked.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, destination)
    return ranked["session"].n_unique(), len(ranked)


def score_candidate_suite(
    candidates: Path,
    models: Path,
    destination: Path,
    *,
    device: str = "cuda",
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> InferenceResult:
    """Score candidate shards with bounded memory and resumable atomic outputs."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    if overwrite and temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    config_hash = stable_hash(
        {
            "candidates": str(candidates.resolve()),
            "models": str(models.resolve()),
            "device": device,
            "top_k": 20,
        }
    )
    sessions: dict[str, int] = {}
    predictions: dict[str, int] = {}
    for target in EVENT_TYPES:
        target_dir = temporary / target.value
        target_dir.mkdir(exist_ok=True)
        model = RankerModel.load(models / f"{target.value}.json")
        target_sessions = 0
        target_predictions = 0
        for source in sorted((candidates / target.value).glob("part-*.parquet")):
            output = target_dir / source.name
            if output.exists():
                try:
                    completed = pl.read_parquet(output, columns=["session"])
                except Exception:
                    output.unlink(missing_ok=True)
                else:
                    target_sessions += completed["session"].n_unique()
                    target_predictions += len(completed)
                    if progress is not None:
                        progress(1)
                    continue
            part_sessions, part_predictions = _score_part(source, output, model, target, device)
            target_sessions += part_sessions
            target_predictions += part_predictions
            if progress is not None:
                progress(1)
        sessions[target.value] = target_sessions
        predictions[target.value] = target_predictions
    result = InferenceResult(sessions=sessions, predictions=predictions)
    manifest = {
        "stage": "ranker_inference",
        "config_hash": config_hash,
        "inputs": {"candidates": str(candidates), "models": str(models)},
        "result": asdict(result),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(temporary), str(destination))
    return result
