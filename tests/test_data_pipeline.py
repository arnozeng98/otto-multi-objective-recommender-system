import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from otto_recsys.cli import app
from otto_recsys.constants import EventType
from otto_recsys.data.materialize import (
    concatenate_event_parquets,
    materialize_official_temporal_split,
    materialize_temporal_split,
)
from otto_recsys.data.preprocess import convert_jsonl_to_parquet
from otto_recsys.data.schemas import Event, Session
from otto_recsys.data.split import split_official_session, split_session


def test_convert_jsonl_flattens_in_batches(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text(
        json.dumps(
            {
                "session": 7,
                "events": [
                    {"aid": 1, "ts": 100, "type": "clicks"},
                    {"aid": 2, "ts": 200, "type": "carts"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "events.parquet"

    manifest = convert_jsonl_to_parquet(source, destination, batch_events=1)

    assert manifest["sessions"] == 1
    assert manifest["events"] == 2
    assert pq.read_table(destination).to_pylist()[1]["aid"] == 2


def test_concatenate_event_parquets_preserves_all_context_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    destination = tmp_path / "combined.parquet"
    sources = [
        {"session": [1], "aid": [10], "ts": [100], "type": ["clicks"]},
        {"session": [2], "aid": [20], "ts": [200], "type": ["carts"]},
    ]
    for path, rows in zip((first, second), sources, strict=True):
        pq.write_table(pa.Table.from_pydict(rows), path)

    count = concatenate_event_parquets((first, second), destination, batch_rows=1)

    combined = pq.read_table(destination).to_pylist()
    assert count == 2
    assert [(row["session"], row["aid"]) for row in combined] == [(1, 10), (2, 20)]


def test_convert_jsonl_honors_session_limit(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    records = [
        {"session": session, "events": [{"aid": session, "ts": 100, "type": "clicks"}]}
        for session in range(3)
    ]
    source.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")

    manifest = convert_jsonl_to_parquet(
        source,
        tmp_path / "events.parquet",
        max_sessions=2,
    )

    assert manifest["sessions"] == 2
    assert manifest["events"] == 2


def test_temporal_split_keeps_future_out_of_context() -> None:
    session = Session(
        1,
        (
            Event(10, 100, EventType.CLICKS),
            Event(20, 300, EventType.CLICKS),
            Event(21, 400, EventType.CLICKS),
            Event(30, 500, EventType.ORDERS),
            Event(30, 600, EventType.ORDERS),
        ),
    )

    split = split_session(session, 250)

    assert split.context is not None
    assert [event.aid for event in split.context.events] == [10]
    assert split.labels[EventType.CLICKS] == (20,)
    assert split.labels[EventType.ORDERS] == (30,)


def test_temporal_split_bounds_labels_to_window() -> None:
    session = Session(
        1,
        (
            Event(10, 100, EventType.CLICKS),
            Event(20, 300, EventType.CLICKS),
            Event(21, 400, EventType.CLICKS),
            Event(30, 500, EventType.ORDERS),
        ),
    )

    split = split_session(session, 250, label_end_timestamp_ms=450)

    assert split.labels[EventType.CLICKS] == (20,)
    assert split.labels[EventType.ORDERS] == ()


def test_official_split_uses_future_next_click_and_deduplicated_buys() -> None:
    session = Session(
        1,
        (
            Event(10, 100, EventType.CLICKS),
            Event(20, 200, EventType.CARTS),
            Event(30, 300, EventType.CLICKS),
            Event(20, 400, EventType.CARTS),
            Event(40, 500, EventType.ORDERS),
            Event(40, 600, EventType.ORDERS),
        ),
    )

    split = split_official_session(session, context_length=2)

    assert [event.aid for event in split.context.events] == [10, 20]
    assert split.labels[EventType.CLICKS] == (30,)
    assert split.labels[EventType.CARTS] == (20,)
    assert split.labels[EventType.ORDERS] == (40,)


def test_official_split_requires_future_event() -> None:
    session = Session(
        1,
        (
            Event(10, 100, EventType.CLICKS),
            Event(20, 200, EventType.CLICKS),
        ),
    )

    with pytest.raises(ValueError, match="between 1 and 1"):
        split_official_session(session, context_length=2)


def test_official_materialization_filters_cohort_and_unknown_aids(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    records = [
        {
            "session": 1,
            "events": [
                {"aid": 10, "ts": 10, "type": "clicks"},
                {"aid": 20, "ts": 20, "type": "carts"},
                {"aid": 30, "ts": 120, "type": "orders"},
            ],
        },
        {
            "session": 2,
            "events": [
                {"aid": 10, "ts": 110, "type": "clicks"},
                {"aid": 999, "ts": 120, "type": "clicks"},
                {"aid": 20, "ts": 130, "type": "carts"},
            ],
        },
        {
            "session": 3,
            "events": [
                {"aid": 10, "ts": 210, "type": "clicks"},
                {"aid": 20, "ts": 220, "type": "orders"},
            ],
        },
    ]
    source.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    parquet = tmp_path / "events.parquet"
    destination = tmp_path / "official"
    convert_jsonl_to_parquet(source, parquet, batch_events=2)

    result = materialize_official_temporal_split(
        parquet,
        destination,
        100,
        cohort_end_timestamp_ms=200,
        seed=42,
        batch_rows=2,
    )

    contexts = pq.read_table(destination / "query_contexts.parquet").to_pylist()
    matrix_events = pq.read_table(destination / "matrix_events.parquet").to_pylist()
    labels = pq.read_table(destination / "labels.parquet").to_pylist()
    assert result.query_sessions == 1
    assert {row["session"] for row in contexts} == {2}
    assert all(row["aid"] != 999 for row in [*contexts, *matrix_events, *labels])
    assert {row["session"] for row in matrix_events} == {1, 2}
    assert {row["session"] for row in labels} == {2}


def test_materialized_split_is_bounded_and_leakage_safe(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    records = [
        {
            "session": 1,
            "events": [
                {"aid": 10, "ts": 100, "type": "clicks"},
                {"aid": 20, "ts": 300, "type": "clicks"},
                {"aid": 21, "ts": 400, "type": "clicks"},
                {"aid": 30, "ts": 500, "type": "orders"},
            ],
        },
        {
            "session": 2,
            "events": [{"aid": 11, "ts": 150, "type": "clicks"}],
        },
    ]
    source.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    parquet = tmp_path / "events.parquet"
    destination = tmp_path / "split"
    convert_jsonl_to_parquet(source, parquet, batch_events=2)

    result = materialize_temporal_split(
        parquet,
        destination,
        250,
        label_end_timestamp_ms=450,
        batch_rows=2,
    )

    matrix_events = pq.read_table(destination / "matrix_events.parquet").to_pylist()
    contexts = pq.read_table(destination / "query_contexts.parquet").to_pylist()
    labels = pq.read_table(destination / "labels.parquet").to_pylist()
    assert {row["session"] for row in matrix_events} == {1, 2}
    assert max(row["ts"] for row in matrix_events) < 250
    assert contexts == [{"session": 1, "aid": 10, "ts": 100, "type": "clicks"}]
    assert labels == [{"session": 1, "type": "clicks", "aid": 20}]
    assert result.query_sessions == 1
    assert (destination / "manifest.json").exists()
    assert not destination.with_name("split.tmp").exists()


def test_split_cli_materializes_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text(
        json.dumps(
            {
                "session": 7,
                "events": [
                    {"aid": 1, "ts": 100, "type": "clicks"},
                    {"aid": 2, "ts": 300, "type": "orders"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parquet = tmp_path / "events.parquet"
    destination = tmp_path / "split"
    convert_jsonl_to_parquet(source, parquet)

    result = CliRunner().invoke(
        app,
        ["split", str(parquet), str(destination), "--cutoff", "250"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["query_sessions"] == 1
    assert (destination / "query_contexts.parquet").exists()


def test_prepare_validation_cli_materializes_two_windows(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text(
        json.dumps(
            {
                "session": 8,
                "events": [
                    {"aid": 1, "ts": 100, "type": "clicks"},
                    {"aid": 2, "ts": 300, "type": "carts"},
                    {"aid": 3, "ts": 500, "type": "orders"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parquet = tmp_path / "events.parquet"
    destination = tmp_path / "views"
    config = tmp_path / "config.yaml"
    base_config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base_config["validation"]["strategy"] = "legacy_global_cutoff"
    base_config["validation"]["training_cutoff_timestamp_ms"] = 250
    base_config["validation"]["cutoff_timestamp_ms"] = 450
    config.write_text(yaml.safe_dump(base_config), encoding="utf-8")
    convert_jsonl_to_parquet(source, parquet, batch_events=1)

    result = CliRunner().invoke(
        app,
        [
            "prepare-validation",
            str(parquet),
            str(destination),
            "--config",
            str(config),
            "--batch-rows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    train_labels = pq.read_table(destination / "ranker_train" / "labels.parquet").to_pylist()
    validation_labels = pq.read_table(
        destination / "local_validation" / "labels.parquet"
    ).to_pylist()
    assert train_labels == [{"session": 8, "type": "carts", "aid": 2}]
    assert validation_labels == [{"session": 8, "type": "orders", "aid": 3}]
    assert (destination / "manifest.json").exists()
    assert not destination.with_name("views.tmp").exists()
