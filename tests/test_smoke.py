import json
from pathlib import Path

from otto_recsys.smoke import run_smoke


def test_smoke_pipeline_writes_metrics_and_submission(tmp_path: Path) -> None:
    report = run_smoke(tmp_path)

    assert report["weighted_recall_at_20"] == 1.0
    assert report["submission_rows"] == 9
    assert (tmp_path / "submission.csv").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["submission_rows"] == 9
