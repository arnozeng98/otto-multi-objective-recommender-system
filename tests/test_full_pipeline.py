import csv
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from otto_recsys.cli import app


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")


def test_run_command_writes_and_reuses_submission(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    sample = tmp_path / "sample_submission.csv"
    destination = tmp_path / "full-run"
    validation_root = tmp_path / "artifacts"
    records = []
    for session in range(1, 5):
        records.append(
            {
                "session": session,
                "events": [
                    {"aid": 20, "ts": 100, "type": "clicks"},
                    {"aid": 30, "ts": 110, "type": "carts"},
                    {"aid": 40, "ts": 120, "type": "orders"},
                    {"aid": 20, "ts": 200, "type": "clicks"},
                    {"aid": 30, "ts": 250, "type": "carts"},
                    {"aid": 40, "ts": 300, "type": "orders"},
                    {"aid": 20, "ts": 400, "type": "clicks"},
                    {"aid": 30, "ts": 450, "type": "carts"},
                    {"aid": 40, "ts": 500, "type": "orders"},
                ],
            }
        )
    _write_jsonl(train, records)
    _write_jsonl(
        test,
        [
            {
                "session": 100,
                "events": [
                    {"aid": 20, "ts": 600, "type": "clicks"},
                    {"aid": 30, "ts": 610, "type": "carts"},
                ],
            }
        ],
    )
    sample.write_text(
        "session_type,labels\n100_clicks,20\n100_carts,30\n100_orders,40\n",
        encoding="utf-8",
    )
    config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config["project"]["artifacts_dir"] = str(validation_root)
    config["validation"]["strategy"] = "legacy_global_cutoff"
    config["validation"]["training_cutoff_timestamp_ms"] = 150
    config["validation"]["cutoff_timestamp_ms"] = 350
    config["covisitation"]["partitions"] = 2
    config["covisitation"]["pair_buffer_size"] = 10
    config["covisitation"]["batch_rows"] = 10
    config["ranking"]["device"] = "cpu"
    config["ranking"]["rounds"] = 2
    config["ranking"]["training_query_limit"] = 2
    config["ranking"]["validation_query_limit"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = [
        "run",
        str(destination),
        "--train",
        str(train),
        "--test",
        str(test),
        "--sample-submission",
        str(sample),
        "--config",
        str(config_path),
        "--no-progress",
    ]

    first = CliRunner().invoke(app, args)

    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    submission = destination / "submission" / "submission.csv"
    first_hash = first_payload["submission_result"]["sha256"]
    with submission.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 3
    assert rows[0]["session_type"] == "100_clicks"

    second = CliRunner().invoke(app, args)

    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert second_payload["reused"] is True
    assert second_payload["submission_result"]["sha256"] == first_hash
