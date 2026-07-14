from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated

import typer

from otto_recsys.candidates.materialize import materialize_candidates
from otto_recsys.config import load_config
from otto_recsys.covisitation import (
    build_covisitation_suite,
    build_matrix_store_suite,
    build_partitioned_covisitation,
    default_rules,
)
from otto_recsys.data.materialize import (
    materialize_temporal_split,
    materialize_validation_views,
)
from otto_recsys.data.preprocess import convert_jsonl_to_parquet
from otto_recsys.pipeline import run_validation_pipeline
from otto_recsys.ranking import train_ranker_suite
from otto_recsys.smoke import run_smoke

app = typer.Typer(no_args_is_help=True, help="Resource-aware OTTO recommendation pipeline.")


@app.command()
def diagnostics() -> None:
    """Print runtime and optional accelerator capabilities."""
    python_version = platform.python_version()
    report: dict[str, object] = {
        "expected_python": "3.12",
        "python": python_version,
        "python_matches_project": python_version.startswith("3.12."),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "faiss_available": find_spec("faiss") is not None,
        "mamba_ssm_available": find_spec("mamba_ssm") is not None,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_wheel"] = "cuda" if torch.version.cuda else "cpu"
        report["torch_cuda_runtime"] = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        report["cuda_available"] = cuda_available
        if cuda_available:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            report["compute_capability"] = list(torch.cuda.get_device_capability(0))
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
        else:
            report["bf16_supported"] = False
    except ImportError:
        report["torch"] = "not installed"
    typer.echo(json.dumps(report, indent=2))


@app.command("check-config")
def check_config(
    path: Annotated[Path, typer.Argument()] = Path("configs/single_gpu.yaml"),
) -> None:
    """Validate and print an experiment configuration."""
    config = load_config(path)
    typer.echo(config.model_dump_json(indent=2))


@app.command()
def preprocess(
    source: Path,
    destination: Path,
    batch_events: int = typer.Option(250_000, min=1),
    max_sessions: int | None = typer.Option(None, min=1),
) -> None:
    """Stream a Kaggle session JSONL file into flat Parquet."""
    report = convert_jsonl_to_parquet(
        source,
        destination,
        batch_events=batch_events,
        max_sessions=max_sessions,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("split")
def split_command(
    source: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    cutoff_timestamp_ms: int = typer.Option(..., "--cutoff", min=0),
    label_end_timestamp_ms: int | None = typer.Option(None, "--label-end", min=1),
    batch_rows: int = typer.Option(250_000, min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Persist leakage-safe matrix events, query contexts, and labels."""
    result = materialize_temporal_split(
        source,
        destination,
        cutoff_timestamp_ms,
        label_end_timestamp_ms=label_end_timestamp_ms,
        batch_rows=batch_rows,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("prepare-validation")
def prepare_validation_command(
    source: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/single_gpu.yaml"),
    batch_rows: int = typer.Option(250_000, min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Materialize ranker-training and local-validation temporal views."""
    config = load_config(config_path)
    result = materialize_validation_views(
        source,
        destination,
        config.validation.training_cutoff_timestamp_ms,
        config.validation.cutoff_timestamp_ms,
        batch_rows=batch_rows,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("build-covisitation")
def build_covisitation_command(
    source: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    rule_name: str = typer.Option("time_decay", "--rule"),
    partitions: int = typer.Option(64, min=1),
    pair_buffer_size: int = typer.Option(500_000, min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Build a bounded-memory, disk-partitioned co-visitation matrix."""
    rules = default_rules()
    if rule_name not in rules:
        available = ", ".join(sorted(rules))
        raise typer.BadParameter(f"Unknown rule '{rule_name}'. Available rules: {available}")
    result = build_partitioned_covisitation(
        source,
        destination,
        rules[rule_name],
        partitions=partitions,
        pair_buffer_size=pair_buffer_size,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("build-covisitation-suite")
def build_covisitation_suite_command(
    source: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    max_neighbors: int = typer.Option(80, min=1),
    partitions: int = typer.Option(64, min=1),
    pair_buffer_size: int = typer.Option(500_000, min=1),
    batch_rows: int = typer.Option(250_000, min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Build all five default co-visitation matrices."""
    result = build_covisitation_suite(
        source,
        destination,
        max_neighbors=max_neighbors,
        partitions=partitions,
        pair_buffer_size=pair_buffer_size,
        batch_rows=batch_rows,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("build-matrix-store")
def build_matrix_store_command(
    source: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Convert a five-matrix suite to lazy memory-mapped lookup stores."""
    result = build_matrix_store_suite(source, destination, overwrite=overwrite)
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("materialize-candidates")
def materialize_candidates_command(
    view: Annotated[Path, typer.Argument()],
    stores: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/single_gpu.yaml"),
    batch_rows: int = typer.Option(250_000, min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Persist target-aware candidate rows and ranking features."""
    config = load_config(config_path)
    result = materialize_candidates(
        view,
        stores,
        destination,
        budget=config.candidates.total_budget,
        popularity_budget=config.candidates.popularity_budget,
        batch_rows=batch_rows,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("train-rankers")
def train_rankers_command(
    training_candidates: Annotated[Path, typer.Argument()],
    validation_candidates: Annotated[Path, typer.Argument()],
    validation_labels: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/single_gpu.yaml"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Train and evaluate three target-specific LambdaMART rankers."""
    config = load_config(config_path)
    result = train_ranker_suite(
        training_candidates,
        validation_candidates,
        validation_labels,
        destination,
        device=config.ranking.device,
        rounds=config.ranking.rounds,
        max_depth=config.ranking.max_depth,
        learning_rate=config.ranking.learning_rate,
        seed=config.project.seed,
        training_candidate_limit=config.ranking.training_candidate_limit,
        validation_candidate_limit=config.ranking.validation_candidate_limit,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("run-validation")
def run_validation_command(
    source: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/single_gpu.yaml"),
    max_sessions: int | None = typer.Option(None, min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_progress: bool = typer.Option(False, "--no-progress"),
) -> None:
    """Run or resume the complete classical local-validation pipeline."""
    result = run_validation_pipeline(
        source,
        destination,
        load_config(config_path),
        max_sessions=max_sessions,
        overwrite=overwrite,
        show_progress=not no_progress,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("run")
def run_command(
    source: Annotated[Path, typer.Argument()],
    destination: Annotated[Path, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/single_gpu.yaml"),
    max_sessions: int | None = typer.Option(None, min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_progress: bool = typer.Option(False, "--no-progress"),
) -> None:
    """Run or resume the complete nine-stage recommendation pipeline."""
    result = run_validation_pipeline(
        source,
        destination,
        load_config(config_path),
        max_sessions=max_sessions,
        overwrite=overwrite,
        show_progress=not no_progress,
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command()
def smoke(
    output_dir: Annotated[Path, typer.Argument()] = Path("artifacts/smoke"),
) -> None:
    """Run a deterministic synthetic end-to-end baseline."""
    typer.echo(json.dumps(run_smoke(output_dir), indent=2))


if __name__ == "__main__":
    app()
