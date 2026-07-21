from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from otto_recsys.artifacts import stable_hash
from otto_recsys.candidates.materialize import materialize_candidates
from otto_recsys.config import AppConfig
from otto_recsys.constants import EVENT_TYPES
from otto_recsys.covisitation import build_covisitation_suite, build_matrix_store_suite
from otto_recsys.data.materialize import concatenate_event_parquets
from otto_recsys.data.preprocess import convert_jsonl_to_parquet
from otto_recsys.pipeline import run_validation_pipeline
from otto_recsys.progress import create_progress
from otto_recsys.ranking import refit_ranker_suite, score_candidate_suite
from otto_recsys.submission import SubmissionResult, write_submission_from_shards


@dataclass(frozen=True, slots=True)
class FullPipelineResult:
    reused: bool
    stages: dict[str, str]
    validation_run: str
    model_strategy: str
    submission: str
    submission_result: SubmissionResult


def _complete(path: Path) -> bool:
    return (path / "manifest.json").exists()


def _source_spec(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            key in actual and _is_subset(value, actual[key]) for key, value in expected.items()
        )
    return bool(expected == actual)


def _compatible_validation_run(path: Path, train: Path, config: AppConfig) -> bool:
    spec_path = path / "run-spec.json"
    if not _complete(path) or not spec_path.exists() or not _complete(path / "rankers"):
        return False
    payload = json.loads(spec_path.read_text(encoding="utf-8"))["spec"]
    source = payload.get("source", {})
    current_source = _source_spec(train)
    if source.get("size") != current_source["size"]:
        return False
    validation_config = payload.get("config", {}).get("validation", {})
    if validation_config.get("strategy") != config.validation.strategy:
        return False
    return _is_subset(payload.get("config", {}), config.model_dump(mode="json"))


def _write_spec(destination: Path, spec: dict[str, Any], overwrite: bool) -> str:
    spec_hash = stable_hash(spec)
    if overwrite and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "run-spec.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["spec_hash"] != spec_hash:
            raise ValueError(
                "Existing full run uses different inputs or configuration; use --overwrite"
            )
    else:
        path.write_text(
            json.dumps({"spec_hash": spec_hash, "spec": spec}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return spec_hash


def run_full_pipeline(
    destination: Path,
    config: AppConfig,
    *,
    train: Path = Path("data/train.jsonl"),
    test: Path = Path("data/test.jsonl"),
    sample_submission: Path = Path("data/sample_submission.csv"),
    validation_run: Path | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    allow_low_score: bool = False,
) -> FullPipelineResult:
    """Run or resume training, test inference, and validated Kaggle submission output."""
    validation_run = validation_run or config.project.artifacts_dir / "full-validation"
    spec = {
        "train": _source_spec(train),
        "test": _source_spec(test),
        "sample_submission": _source_spec(sample_submission),
        "config": config.model_dump(mode="json"),
        "validation_run": str(validation_run.resolve()),
        "model_strategy": config.ranking.model_strategy,
        "allow_low_score": allow_low_score,
    }
    spec_hash = _write_spec(destination, spec, overwrite)
    final_manifest = destination / "manifest.json"
    stage_names = (
        "Validate or train rankers",
        "Preprocess test sessions",
        "Build full-history matrices",
        "Build full-history stores",
        "Materialize test candidates",
        "Prepare final rankers",
        "Score test candidates",
        "Write and validate submission",
    )
    if final_manifest.exists():
        payload = json.loads(final_manifest.read_text(encoding="utf-8"))["result"]
        result = FullPipelineResult(
            reused=True,
            stages={name: "reused" for name in payload["stages"]},
            validation_run=payload["validation_run"],
            model_strategy=payload["model_strategy"],
            submission=payload["submission"],
            submission_result=SubmissionResult(**payload["submission_result"]),
        )
        if show_progress:
            with create_progress(True) as progress:
                for step, name in enumerate(stage_names, start=1):
                    progress.start_stage(step, len(stage_names), name, total=1, unit="stage")
                    progress.finish("reused")
        return result

    stages: dict[str, str] = {}
    with create_progress(show_progress) as progress:
        progress.start_stage(1, 8, stage_names[0], total=1, unit="pipeline")
        if _compatible_validation_run(validation_run, train, config):
            stages["validation"] = "reused"
            progress.finish("reused")
        else:
            run_validation_pipeline(
                train,
                validation_run,
                config,
                show_progress=show_progress,
                allow_low_score=allow_low_score,
            )
            stages["validation"] = "completed"
            progress.finish("completed")

        test_events = destination / "test-events.parquet"
        test_report = destination / "test-preprocess.json"
        progress.start_stage(2, 8, stage_names[1], total=test.stat().st_size, unit="bytes")
        if test_events.exists() and test_report.exists():
            stages["test_preprocess"] = "reused"
            progress.finish("reused")
        else:
            report = convert_jsonl_to_parquet(test, test_events, progress=progress.advance)
            test_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            stages["test_preprocess"] = "completed"
            progress.finish("completed")
        test_payload = json.loads(test_report.read_text(encoding="utf-8"))

        train_events = validation_run / "events.parquet"
        inference_events = destination / "full-history-events.parquet"
        inference_events_report = destination / "full-history-events.json"
        if not inference_events.exists() or not inference_events_report.exists():
            event_rows = concatenate_event_parquets((train_events, test_events), inference_events)
            inference_events_report.write_text(
                json.dumps(
                    {
                        "stage": "transductive_inference_events",
                        "inputs": {
                            "train": _source_spec(train_events),
                            "test_context": _source_spec(test_events),
                        },
                        "result": {"events": event_rows},
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        matrices = destination / "full-history-matrices"
        matrix_count = 3 if config.covisitation.profile == "public_v575" else 5
        progress.start_stage(3, 8, stage_names[2], total=matrix_count, unit="rules")
        if _complete(matrices):
            stages["full_history_matrices"] = "reused"
            progress.finish("reused")
        else:
            build_covisitation_suite(
                inference_events,
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
            stages["full_history_matrices"] = "completed"
            progress.finish("completed")

        stores = destination / "full-history-stores"
        progress.start_stage(4, 8, stage_names[3], total=matrix_count, unit="stores")
        if _complete(stores):
            stages["full_history_stores"] = "reused"
            progress.finish("reused")
        else:
            build_matrix_store_suite(matrices, stores, progress=progress.advance)
            stages["full_history_stores"] = "completed"
            progress.finish("completed")

        inference_view = destination / "test-view"
        inference_view.mkdir(exist_ok=True)
        (inference_view / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "inference_view",
                    "inputs": {
                        "query_contexts": str(test_events),
                        "popularity_events": str(inference_events),
                    },
                    "result": {"query_sessions": int(test_payload["sessions"])},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        candidates = destination / "test-candidates"
        progress.start_stage(
            5, 8, stage_names[4], total=int(test_payload["sessions"]), unit="sessions"
        )
        if _complete(candidates):
            stages["test_candidates"] = "reused"
            progress.finish("reused")
        else:
            materialize_candidates(
                inference_view,
                stores,
                candidates,
                budget=config.candidates.total_budget,
                popularity_budget=config.candidates.popularity_budget,
                history_budget=config.candidates.history_budget,
                covisitation_budget=config.candidates.covisitation_budget,
                max_events_per_session=config.covisitation.max_events_per_session,
                query_contexts=test_events,
                popularity_events=inference_events,
                include_labels=False,
                retention_limit=config.ranking.validation_candidate_limit,
                workers=config.candidates.workers,
                chunk_sessions=config.candidates.chunk_sessions,
                progress=progress.advance,
            )
            stages["test_candidates"] = "completed"
            progress.finish("completed")

        progress.start_stage(6, 8, stage_names[5], total=3, unit="targets")
        if config.ranking.model_strategy == "validated":
            models = validation_run / "rankers"
            stages["final_rankers"] = "reused"
            progress.finish("reused")
        else:
            models = destination / "final-rankers"
            if _complete(models):
                stages["final_rankers"] = "reused"
                progress.finish("reused")
            else:
                refit_ranker_suite(
                    (
                        validation_run / "candidates" / "ranker_train",
                        validation_run / "candidates" / "local_validation",
                    ),
                    models,
                    device=config.ranking.device,
                    rounds=config.ranking.rounds,
                    max_depth=config.ranking.max_depth,
                    learning_rate=config.ranking.learning_rate,
                    nthread=config.ranking.nthread,
                    seed=config.project.seed,
                    candidate_limit=config.ranking.training_candidate_limit,
                    query_limit=config.ranking.training_query_limit,
                    progress=progress.advance,
                )
                stages["final_rankers"] = "completed"
                progress.finish("completed")
        predictions = destination / "test-predictions"
        prediction_parts = sum(
            len(list((candidates / target.value).glob("part-*.parquet"))) for target in EVENT_TYPES
        )
        progress.start_stage(7, 8, stage_names[6], total=prediction_parts, unit="shards")
        if _complete(predictions):
            stages["test_predictions"] = "reused"
            progress.finish("reused")
        else:
            score_candidate_suite(
                candidates,
                models,
                predictions,
                device=config.ranking.device,
                progress=progress.advance,
            )
            stages["test_predictions"] = "completed"
            progress.finish("completed")

        submission_dir = destination / "submission"
        submission_path = submission_dir / "submission.csv"
        progress.start_stage(
            8, 8, stage_names[7], total=int(test_payload["sessions"]) * 3, unit="rows"
        )
        if _complete(submission_dir) and submission_path.exists():
            submission_payload = json.loads(
                (submission_dir / "manifest.json").read_text(encoding="utf-8")
            )
            submission_result = SubmissionResult(**submission_payload["result"])
            stages["submission"] = "reused"
            progress.finish("reused")
        else:
            popularity_payload = json.loads(
                (candidates / "popularity.json").read_text(encoding="utf-8")
            )
            backfill = {target: tuple(popularity_payload[target.value]) for target in EVENT_TYPES}
            submission_result = write_submission_from_shards(
                submission_path, sample_submission, predictions, backfill
            )
            stages["submission"] = "completed"
            progress.advance(submission_result.rows)
            progress.finish("completed")

    result = FullPipelineResult(
        reused=False,
        stages=stages,
        validation_run=str(validation_run),
        model_strategy=config.ranking.model_strategy,
        submission=str(submission_path),
        submission_result=submission_result,
    )
    final_manifest.write_text(
        json.dumps(
            {"stage": "full_pipeline", "spec_hash": spec_hash, "result": asdict(result)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result
