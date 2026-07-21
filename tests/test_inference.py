import json
from pathlib import Path

import polars as pl

from otto_recsys.constants import EVENT_TYPES
from otto_recsys.full_pipeline import _validated_strategy
from otto_recsys.ranking import rank_candidate_suite


def test_rules_inference_promotes_candidate_rank_top_20(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    for target in EVENT_TYPES:
        directory = candidates / target.value
        directory.mkdir(parents=True)
        pl.DataFrame(
            {
                "session": [10] * 21,
                "aid": list(range(101, 122)),
                "candidate_rank": list(range(1, 22)),
            }
        ).write_parquet(directory / "part-000000.parquet")

    result = rank_candidate_suite(candidates, tmp_path / "predictions")

    assert result.predictions == {target.value: 20 for target in EVENT_TYPES}
    for target in EVENT_TYPES:
        frame = pl.read_parquet(tmp_path / "predictions" / target.value / "*.parquet")
        assert frame["aid"].to_list() == list(range(101, 121))
        assert frame["rank"].to_list() == list(range(1, 21))


def test_validated_strategy_rejects_regressing_ranker(tmp_path: Path) -> None:
    (tmp_path / "rankers").mkdir()
    (tmp_path / "candidate-recall.json").write_text(
        json.dumps({"weighted_recall_at_20": 0.56}), encoding="utf-8"
    )
    manifest = tmp_path / "rankers" / "manifest.json"
    manifest.write_text(json.dumps({"result": {"weighted_recall_at_20": 0.05}}), encoding="utf-8")

    assert _validated_strategy(tmp_path, "validated") == "rules"

    manifest.write_text(json.dumps({"result": {"weighted_recall_at_20": 0.57}}), encoding="utf-8")
    assert _validated_strategy(tmp_path, "validated") == "validated"
    assert _validated_strategy(tmp_path, "refit") == "refit"
