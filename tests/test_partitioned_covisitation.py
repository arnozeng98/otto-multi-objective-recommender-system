import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from otto_recsys.cli import app
from otto_recsys.covisitation import (
    CovisitationRule,
    MatrixStore,
    build_covisitation,
    build_covisitation_suite,
    build_matrix_store,
    build_matrix_store_suite,
    build_partitioned_covisitation,
    iter_parquet_sessions,
)
from otto_recsys.data.preprocess import convert_jsonl_to_parquet


def _write_sessions(path: Path) -> None:
    records = [
        {
            "session": 1,
            "events": [
                {"aid": 10, "ts": 100, "type": "clicks"},
                {"aid": 20, "ts": 200, "type": "clicks"},
                {"aid": 30, "ts": 300, "type": "orders"},
            ],
        },
        {
            "session": 2,
            "events": [
                {"aid": 10, "ts": 400, "type": "clicks"},
                {"aid": 20, "ts": 500, "type": "clicks"},
            ],
        },
    ]
    path.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")


def test_parquet_sessions_preserve_sessions_across_batches(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    parquet = tmp_path / "events.parquet"
    _write_sessions(source)
    convert_jsonl_to_parquet(source, parquet, batch_events=2)

    sessions = list(iter_parquet_sessions(parquet, batch_rows=2))

    assert [session.session for session in sessions] == [1, 2]
    assert [[event.aid for event in session.events] for session in sessions] == [
        [10, 20, 30],
        [10, 20],
    ]


def test_partitioned_matrix_matches_in_memory_reference(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    parquet = tmp_path / "events.parquet"
    destination = tmp_path / "matrix"
    _write_sessions(source)
    convert_jsonl_to_parquet(source, parquet, batch_events=2)
    rule = CovisitationRule(
        "test",
        forward_only=True,
        half_life_seconds=None,
        max_neighbors=2,
    )
    sessions = list(iter_parquet_sessions(parquet, batch_rows=2))
    expected = build_covisitation(sessions, rule)

    result = build_partitioned_covisitation(
        parquet,
        destination,
        rule,
        partitions=2,
        pair_buffer_size=2,
        batch_rows=2,
        reduction_workers=2,
    )

    frames = [pl.read_parquet(path) for path in sorted((destination / "matrix").glob("*.parquet"))]
    actual: dict[int, tuple[tuple[int, float], ...]] = {}
    for row in pl.concat(frames).sort(["source_aid", "rank"]).iter_rows(named=True):
        actual.setdefault(row["source_aid"], ())
        actual[row["source_aid"]] += ((row["target_aid"], row["weight"]),)
    assert actual == expected
    assert result.sessions == 2
    assert result.output_files == 1
    assert (destination / "manifest.json").exists()
    assert not (destination / "fragments").exists()


def test_build_covisitation_cli(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    parquet = tmp_path / "events.parquet"
    destination = tmp_path / "matrix"
    _write_sessions(source)
    convert_jsonl_to_parquet(source, parquet, batch_events=2)

    result = CliRunner().invoke(
        app,
        [
            "build-covisitation",
            str(parquet),
            str(destination),
            "--rule",
            "adjacent_clicks",
            "--partitions",
            "2",
            "--pair-buffer-size",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["rule"] == "adjacent_clicks"
    assert report["sessions"] == 2


def test_covisitation_suite_builds_all_default_rules(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    parquet = tmp_path / "events.parquet"
    destination = tmp_path / "matrices"
    _write_sessions(source)
    convert_jsonl_to_parquet(source, parquet, batch_events=2)

    result = build_covisitation_suite(
        parquet,
        destination,
        max_neighbors=2,
        partitions=2,
        pair_buffer_size=2,
        batch_rows=2,
    )

    expected = {"adjacent_clicks", "time_decay", "all_to_buy", "buy_to_buy", "recent_trend"}
    assert set(result.matrices) == expected
    assert {path.name for path in destination.iterdir() if path.is_dir()} == expected
    assert all((destination / name / "manifest.json").exists() for name in expected)
    assert (destination / "manifest.json").exists()
    assert not destination.with_name("matrices.tmp").exists()


def test_matrix_store_matches_partitioned_rows(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    parquet = tmp_path / "events.parquet"
    matrix = tmp_path / "matrix"
    store_path = tmp_path / "store"
    _write_sessions(source)
    convert_jsonl_to_parquet(source, parquet, batch_events=2)
    build_partitioned_covisitation(
        parquet,
        matrix,
        CovisitationRule("test", half_life_seconds=None, max_neighbors=3),
        partitions=2,
        pair_buffer_size=2,
        batch_rows=2,
    )

    result = build_matrix_store(matrix, store_path)
    store = MatrixStore(store_path)

    rows = pl.concat(
        [pl.read_parquet(path) for path in sorted((matrix / "matrix").glob("*.parquet"))]
    ).filter(pl.col("source_aid") == 10)
    expected = tuple(
        (row["target_aid"], row["weight"], row["rank"])
        for row in rows.sort("rank").iter_rows(named=True)
    )
    assert store.neighbors(10) == expected
    assert store.neighbors(999) == ()
    assert result.neighbors > 0


def test_matrix_store_suite_converts_every_rule(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    parquet = tmp_path / "events.parquet"
    matrices = tmp_path / "matrices"
    stores = tmp_path / "stores"
    _write_sessions(source)
    convert_jsonl_to_parquet(source, parquet, batch_events=2)
    matrix_result = build_covisitation_suite(
        parquet,
        matrices,
        max_neighbors=2,
        partitions=2,
        pair_buffer_size=2,
        batch_rows=2,
    )

    result = build_matrix_store_suite(matrices, stores)

    assert set(result.stores) == set(matrix_result.matrices)
    assert all((stores / name / "manifest.json").exists() for name in result.stores)
    assert (stores / "manifest.json").exists()
