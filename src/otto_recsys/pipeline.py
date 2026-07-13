from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from otto_recsys.artifacts import stable_hash
from otto_recsys.candidates.materialize import materialize_candidates
from otto_recsys.config import AppConfig
from otto_recsys.covisitation import build_covisitation_suite, build_matrix_store_suite
from otto_recsys.data.materialize import materialize_validation_views
from otto_recsys.data.preprocess import convert_jsonl_to_parquet
from otto_recsys.ranking import RankerSuiteResult, train_ranker_suite


@dataclass(frozen=True, slots=True)
class ValidationPipelineResult:
    reused: bool
    stages: dict[str, str]
    rankers: RankerSuiteResult


def _complete(path: Path) -> bool:
    return (path / "manifest.json").exists()


def run_validation_pipeline(
    source: Path,
    destination: Path,
    config: AppConfig,
    *,
    max_sessions: int | None = None,
    overwrite: bool = False,
) -> ValidationPipelineResult:
    """Run or resume the complete classical local-validation pipeline."""
    source_stat = source.stat()
    spec = {
        "source": {
            "path": str(source.resolve()),
            "size": source_stat.st_size,
            "modified_ns": source_stat.st_mtime_ns,
        },
        "config": config.model_dump(mode="json"),
        "max_sessions": max_sessions,
    }
    spec_hash = stable_hash(spec)
    if overwrite and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spec_path = destination / "run-spec.json"
    if spec_path.exists():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing["spec_hash"] != spec_hash:
            raise ValueError("Existing run uses different input or configuration; use --overwrite")
    else:
        spec_path.write_text(
            json.dumps({"spec_hash": spec_hash, "spec": spec}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    final_manifest = destination / "manifest.json"
    if final_manifest.exists():
        payload = json.loads(final_manifest.read_text(encoding="utf-8"))
        return ValidationPipelineResult(
            reused=True,
            stages={name: "reused" for name in payload["result"]["stages"]},
            rankers=RankerSuiteResult(**payload["result"]["rankers"]),
        )

    stages: dict[str, str] = {}
    events = destination / "events.parquet"
    preprocess_report = destination / "preprocess.json"
    if events.exists() and preprocess_report.exists():
        stages["preprocess"] = "reused"
    else:
        events.unlink(missing_ok=True)
        report = convert_jsonl_to_parquet(source, events, max_sessions=max_sessions)
        preprocess_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        stages["preprocess"] = "completed"

    views = destination / "views"
    if _complete(views):
        stages["views"] = "reused"
    else:
        shutil.rmtree(views, ignore_errors=True)
        materialize_validation_views(
            events,
            views,
            config.validation.training_cutoff_timestamp_ms,
            config.validation.cutoff_timestamp_ms,
        )
        stages["views"] = "completed"

    for view_name in ("ranker_train", "local_validation"):
        matrices = destination / "matrices" / view_name
        matrix_stage = f"matrices_{view_name}"
        if _complete(matrices):
            stages[matrix_stage] = "reused"
        else:
            shutil.rmtree(matrices, ignore_errors=True)
            build_covisitation_suite(
                views / view_name / "matrix_events.parquet",
                matrices,
                max_neighbors=config.covisitation.max_neighbors,
                partitions=config.covisitation.partitions,
            )
            stages[matrix_stage] = "completed"

        stores = destination / "stores" / view_name
        store_stage = f"stores_{view_name}"
        if _complete(stores):
            stages[store_stage] = "reused"
        else:
            shutil.rmtree(stores, ignore_errors=True)
            build_matrix_store_suite(matrices, stores)
            stages[store_stage] = "completed"

        candidates = destination / "candidates" / view_name
        candidate_stage = f"candidates_{view_name}"
        if _complete(candidates):
            stages[candidate_stage] = "reused"
        else:
            shutil.rmtree(candidates, ignore_errors=True)
            materialize_candidates(
                views / view_name,
                stores,
                candidates,
                budget=config.candidates.total_budget,
                popularity_budget=config.candidates.popularity_budget,
            )
            stages[candidate_stage] = "completed"

    ranker_path = destination / "rankers"
    if _complete(ranker_path):
        ranker_payload = json.loads((ranker_path / "manifest.json").read_text(encoding="utf-8"))
        rankers = RankerSuiteResult(**ranker_payload["result"])
        stages["rankers"] = "reused"
    else:
        shutil.rmtree(ranker_path, ignore_errors=True)
        rankers = train_ranker_suite(
            destination / "candidates" / "ranker_train",
            destination / "candidates" / "local_validation",
            views / "local_validation" / "labels.parquet",
            ranker_path,
            device=config.ranking.device,
            rounds=config.ranking.rounds,
            max_depth=config.ranking.max_depth,
            learning_rate=config.ranking.learning_rate,
            seed=config.project.seed,
            training_candidate_limit=config.ranking.training_candidate_limit,
            validation_candidate_limit=config.ranking.validation_candidate_limit,
        )
        stages["rankers"] = "completed"

    result = ValidationPipelineResult(reused=False, stages=stages, rankers=rankers)
    final_manifest.write_text(
        json.dumps(
            {"stage": "validation_pipeline", "spec_hash": spec_hash, "result": asdict(result)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result
