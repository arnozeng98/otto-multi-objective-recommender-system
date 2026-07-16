from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    seed: int = 2026
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")


class ValidationConfig(StrictModel):
    training_cutoff_timestamp_ms: int
    cutoff_timestamp_ms: int
    top_k: int = Field(default=20, ge=1, le=20)

    @model_validator(mode="after")
    def validate_cutoff_order(self) -> Self:
        if self.training_cutoff_timestamp_ms >= self.cutoff_timestamp_ms:
            raise ValueError("training cutoff must be earlier than validation cutoff")
        return self


class CandidateConfig(StrictModel):
    total_budget: int = Field(default=250, ge=20)
    history_budget: int = Field(default=50, ge=0)
    popularity_budget: int = Field(default=40, ge=0)
    covisitation_budget: int = Field(default=180, ge=0)


class CovisitationConfig(StrictModel):
    max_events_per_session: int = Field(default=30, ge=2)
    max_neighbors: int = Field(default=80, ge=1)
    partitions: int = Field(default=64, ge=1)
    pair_buffer_size: int = Field(default=500_000, ge=1)
    batch_rows: int = Field(default=250_000, ge=1)


class RankingConfig(StrictModel):
    device: str = "cuda"
    max_depth: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=0.08, gt=0)
    rounds: int = Field(default=500, ge=1)
    training_candidate_limit: int = Field(default=80, ge=20)
    validation_candidate_limit: int = Field(default=120, ge=20)
    training_query_limit: int = Field(default=50_000, ge=1)
    validation_query_limit: int = Field(default=50_000, ge=1)
    model_strategy: Literal["validated", "refit"] = "validated"


class NeuralConfig(StrictModel):
    embedding_dim: int = Field(default=64, ge=8)
    max_sequence_length: int = Field(default=64, ge=2)
    batch_size: int = Field(default=128, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    precision: str = "bf16"


class AppConfig(StrictModel):
    project: ProjectConfig
    validation: ValidationConfig
    candidates: CandidateConfig
    covisitation: CovisitationConfig
    ranking: RankingConfig
    neural: NeuralConfig


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> AppConfig:
    """Load a validated YAML config with an optional one-level `extends` reference."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = path.parent / parent
        parent_raw = yaml.safe_load(parent_path.read_text(encoding="utf-8")) or {}
        parent_raw.pop("extends", None)
        raw = _merge(parent_raw, raw)
    return AppConfig.model_validate(raw)
