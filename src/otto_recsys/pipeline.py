from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from otto_recsys.artifacts import stable_hash
from otto_recsys.candidates.materialize import materialize_candidates
from otto_recsys.config import AppConfig
from otto_recsys.constants import EventType
from otto_recsys.covisitation import build_covisitation_suite, build_matrix_store_suite
from otto_recsys.data.materialize import (
    materialize_official_validation_views,
    materialize_validation_views,
)
from otto_recsys.data.preprocess import convert_jsonl_to_parquet
from otto_recsys.metrics import CandidateRecallReport, evaluate_candidate_recall
from otto_recsys.progress import create_progress
from otto_recsys.ranking import RankerSuiteResult, train_ranker_suite


@dataclass(frozen=True, slots=True)
class ValidationPipelineResult:
    reused: bool
    stages: dict[str, str]
    rankers: RankerSuiteResult


def _complete(path: Path) -> bool:
    return (path / "manifest.json").exists()


def _check_candidate_quality(report: CandidateRecallReport, config: AppConfig) -> None:
    failures = []
    if report.weighted_recall_at_20 < config.validation.minimum_rules_recall_at_20:
        failures.append(
            f"rules Recall@20 {report.weighted_recall_at_20:.6f} < "
            f"{config.validation.minimum_rules_recall_at_20:.6f}"
        )
    if report.weighted_recall_at_100 < config.validation.minimum_candidate_recall_at_100:
        failures.append(
            f"candidate Recall@100 {report.weighted_recall_at_100:.6f} < "
            f"{config.validation.minimum_candidate_recall_at_100:.6f}"
        )
    if failures:
        raise RuntimeError(
            "Candidate quality gate failed: "
            + "; ".join(failures)
            + ". Inspect candidate-recall.json before training or submission."
        )


def run_validation_pipeline(
    source: Path,
    destination: Path,
    config: AppConfig,
    *,
    max_sessions: int | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
    allow_low_score: bool = False,
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
    stage_names = (
        "Preprocess events",
        "Build temporal views",
        "Build training matrices",
        "Build training stores",
        "Materialize training candidates",
        "Build validation matrices",
        "Build validation stores",
        "Materialize validation candidates",
        "Train and evaluate rankers",
    )
    total_steps = len(stage_names)
    final_manifest = destination / "manifest.json"
    if final_manifest.exists():
        payload = json.loads(final_manifest.read_text(encoding="utf-8"))
        if show_progress:
            with create_progress(True) as progress:
                for step, name in enumerate(stage_names, start=1):
                    progress.start_stage(step, total_steps, name, total=1, unit="stage")
                    progress.finish("reused")
        return ValidationPipelineResult(
            reused=True,
            stages={name: "reused" for name in payload["result"]["stages"]},
            rankers=RankerSuiteResult(**payload["result"]["rankers"]),
        )

    stages: dict[str, str] = {}
    events = destination / "events.parquet"
    preprocess_report = destination / "preprocess.json"
    views = destination / "views"
    with create_progress(show_progress) as progress:
        progress.start_stage(
            1,
            total_steps,
            stage_names[0],
            total=source_stat.st_size,
            unit="bytes",
        )
        if events.exists() and preprocess_report.exists():
            stages["preprocess"] = "reused"
            progress.finish("reused")
        else:
            events.unlink(missing_ok=True)
            report = convert_jsonl_to_parquet(
                source,
                events,
                max_sessions=max_sessions,
                progress=progress.advance,
            )
            preprocess_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            stages["preprocess"] = "completed"
            progress.finish("completed")
        preprocess_payload = json.loads(preprocess_report.read_text(encoding="utf-8"))

        progress.start_stage(
            2,
            total_steps,
            stage_names[1],
            total=int(preprocess_payload["sessions"]) * 2,
            unit="sessions",
        )
        if _complete(views):
            stages["views"] = "reused"
            progress.finish("reused")
        else:
            shutil.rmtree(views, ignore_errors=True)
            if config.validation.strategy == "official_random_event":
                materialize_official_validation_views(
                    events,
                    views,
                    validation_days=config.validation.days,
                    seed=config.validation.seed,
                    progress=progress.advance,
                )
            else:
                materialize_validation_views(
                    events,
                    views,
                    config.validation.training_cutoff_timestamp_ms,
                    config.validation.cutoff_timestamp_ms,
                    progress=progress.advance,
                )
            stages["views"] = "completed"
            progress.finish("completed")
        views_payload = json.loads((views / "manifest.json").read_text(encoding="utf-8"))

        for view_index, view_name in enumerate(("ranker_train", "local_validation")):
            step_base = 3 + view_index * 3
            matrices = destination / "matrices" / view_name
            matrix_stage = f"matrices_{view_name}"
            matrix_count = 3 if config.covisitation.profile == "public_v575" else 5
            progress.start_stage(
                step_base,
                total_steps,
                stage_names[step_base - 1],
                total=matrix_count,
                unit="rules",
            )
            if _complete(matrices):
                stages[matrix_stage] = "reused"
                progress.finish("reused")
            else:
                build_covisitation_suite(
                    views / view_name / "matrix_events.parquet",
                    matrices,
                    max_neighbors=config.covisitation.max_neighbors,
                    partitions=config.covisitation.partitions,
                    pair_buffer_size=config.covisitation.pair_buffer_size,
                    batch_rows=config.covisitation.batch_rows,
                    max_events_per_session=config.covisitation.max_events_per_session,
                    profile=config.covisitation.profile,
                    reduction_workers=config.covisitation.reduction_workers,
                    progress=progress.advance,
                )
                stages[matrix_stage] = "completed"
                progress.finish("completed")

            stores = destination / "stores" / view_name
            store_stage = f"stores_{view_name}"
            progress.start_stage(
                step_base + 1,
                total_steps,
                stage_names[step_base],
                total=matrix_count,
                unit="stores",
            )
            if _complete(stores):
                stages[store_stage] = "reused"
                progress.finish("reused")
            else:
                build_matrix_store_suite(matrices, stores, progress=progress.advance)
                stages[store_stage] = "completed"
                progress.finish("completed")

            candidates = destination / "candidates" / view_name
            candidate_stage = f"candidates_{view_name}"
            query_sessions = int(views_payload["result"][view_name]["query_sessions"])
            progress.start_stage(
                step_base + 2,
                total_steps,
                stage_names[step_base + 1],
                total=query_sessions,
                unit="sessions",
            )
            if _complete(candidates):
                stages[candidate_stage] = "reused"
                progress.finish("reused")
            else:
                materialize_candidates(
                    views / view_name,
                    stores,
                    candidates,
                    budget=config.candidates.total_budget,
                    popularity_budget=config.candidates.popularity_budget,
                    history_budget=config.candidates.history_budget,
                    covisitation_budget=config.candidates.covisitation_budget,
                    max_events_per_session=config.covisitation.max_events_per_session,
                    retention_limit=(
                        config.ranking.training_candidate_limit
                        if view_name == "ranker_train"
                        else max(config.ranking.validation_candidate_limit, 150)
                    ),
                    workers=config.candidates.workers,
                    chunk_sessions=config.candidates.chunk_sessions,
                    progress=progress.advance,
                )
                stages[candidate_stage] = "completed"
                progress.finish("completed")

        recall_path = destination / "candidate-recall.json"
        candidate_recall = evaluate_candidate_recall(
            destination / "candidates" / "local_validation",
            views / "local_validation" / "labels.parquet",
        )
        recall_path.write_text(
            json.dumps(asdict(candidate_recall), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if (
            config.validation.strategy == "official_random_event"
            and config.validation.enforce_quality_gate
            and not allow_low_score
        ):
            _check_candidate_quality(candidate_recall, config)

        ranker_path = destination / "rankers"
        progress.start_stage(9, total_steps, stage_names[8], total=3, unit="targets")
        if _complete(ranker_path):
            ranker_payload = json.loads((ranker_path / "manifest.json").read_text(encoding="utf-8"))
            rankers = RankerSuiteResult(**ranker_payload["result"])
            stages["rankers"] = "reused"
            progress.finish("reused")
        else:
            rankers = train_ranker_suite(
                destination / "candidates" / "ranker_train",
                destination / "candidates" / "local_validation",
                views / "local_validation" / "labels.parquet",
                ranker_path,
                device=config.ranking.device,
                rounds=config.ranking.rounds,
                max_depth=config.ranking.max_depth,
                learning_rate=config.ranking.learning_rate,
                nthread=config.ranking.nthread,
                seed=config.project.seed,
                training_candidate_limit=config.ranking.training_candidate_limit,
                validation_candidate_limit=config.ranking.validation_candidate_limit,
                training_query_limit=config.ranking.training_query_limit,
                validation_query_limit=config.ranking.validation_query_limit,
                negative_sample_rates={
                    EventType.CLICKS: config.ranking.clicks_negative_sample_rate,
                    EventType.CARTS: config.ranking.carts_negative_sample_rate,
                    EventType.ORDERS: config.ranking.orders_negative_sample_rate,
                },
                progress=progress.advance,
            )
            stages["rankers"] = "completed"
            progress.finish("completed")

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
