from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RankerModel:
    booster: Any
    feature_names: tuple[str, ...]

    def predict(
        self,
        features: np.ndarray[Any, np.dtype[np.float32]],
        *,
        device: str | None = None,
    ) -> np.ndarray[Any, Any]:
        import xgboost as xgb

        if device is not None:
            self.booster.set_param({"device": device})
        matrix = xgb.DMatrix(features, feature_names=list(self.feature_names))
        return np.asarray(self.booster.predict(matrix))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path)
        path.with_suffix(f"{path.suffix}.features.json").write_text(
            json.dumps(self.feature_names),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RankerModel:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(path)
        feature_names = tuple(
            json.loads(path.with_suffix(f"{path.suffix}.features.json").read_text(encoding="utf-8"))
        )
        return cls(booster=booster, feature_names=feature_names)


def train_ranker(
    features: np.ndarray[Any, np.dtype[np.float32]],
    labels: np.ndarray[Any, np.dtype[np.float32]],
    group_sizes: Sequence[int],
    feature_names: Sequence[str],
    *,
    validation_features: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    validation_labels: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    validation_group_sizes: Sequence[int] | None = None,
    device: str = "cpu",
    rounds: int = 100,
    seed: int = 2026,
    max_depth: int = 8,
    learning_rate: float = 0.08,
    early_stopping_rounds: int | None = None,
) -> RankerModel:
    """Train a LambdaMART ranker with explicit, validated session groups."""
    import xgboost as xgb

    if sum(group_sizes) != len(labels) or len(features) != len(labels):
        raise ValueError("Group sizes, features, and labels must describe the same rows")
    matrix = xgb.DMatrix(features, label=labels, feature_names=list(feature_names))
    matrix.set_group(group_sizes)
    evals: list[tuple[Any, str]] = []
    if validation_features is not None:
        if validation_labels is None or validation_group_sizes is None:
            raise ValueError("Validation features require labels and group sizes")
        if sum(validation_group_sizes) != len(validation_labels) or len(validation_features) != len(
            validation_labels
        ):
            raise ValueError("Validation groups, features, and labels must describe the same rows")
        validation_matrix = xgb.DMatrix(
            validation_features,
            label=validation_labels,
            feature_names=list(feature_names),
        )
        validation_matrix.set_group(validation_group_sizes)
        evals.append((validation_matrix, "validation"))
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@20",
        "tree_method": "hist",
        "device": device,
        "max_depth": max_depth,
        "eta": learning_rate,
        "seed": seed,
    }
    booster = xgb.train(
        params,
        matrix,
        num_boost_round=rounds,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return RankerModel(booster=booster, feature_names=tuple(feature_names))
