from pathlib import Path

import numpy as np
import polars as pl
import pytest

xgboost = pytest.importorskip("xgboost")


def test_ranker_trains_and_predicts_grouped_candidates(tmp_path) -> None:
    from otto_recsys.ranking import RankerModel, train_ranker

    features = np.asarray([[2.0], [0.0], [1.5], [0.1]], dtype=np.float32)
    labels = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    model = train_ranker(
        features,
        labels,
        [2, 2],
        ["score"],
        validation_features=features,
        validation_labels=labels,
        validation_group_sizes=[2, 2],
        rounds=5,
        early_stopping_rounds=2,
    )
    predictions = model.predict(features)
    model_path = tmp_path / "ranker.json"
    model.save(model_path)
    loaded_predictions = RankerModel.load(model_path).predict(features)

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(loaded_predictions, predictions)


def test_ranker_rejects_inconsistent_groups() -> None:
    from otto_recsys.ranking import train_ranker

    features = np.asarray([[1.0], [0.0]], dtype=np.float32)
    labels = np.asarray([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="same rows"):
        train_ranker(features, labels, [1], ["score"], rounds=1)


def test_ranker_suite_writes_three_models_and_predictions(tmp_path: Path) -> None:
    from otto_recsys.constants import EVENT_TYPES
    from otto_recsys.ranking import train_ranker_suite

    training = tmp_path / "training"
    validation = tmp_path / "validation"
    for root in (training, validation):
        for target in EVENT_TYPES:
            directory = root / target.value
            directory.mkdir(parents=True)
            pl.DataFrame(
                {
                    "session": [1, 1, 2, 2],
                    "target": [target.value] * 4,
                    "aid": [10, 11, 20, 21],
                    "label": [1, 0, 1, 0],
                    "candidate_score": [2.0, 0.0, 2.0, 0.0],
                }
            ).write_parquet(directory / "part-000000.parquet")
    labels = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2, 2],
            "type": [target.value for _ in range(2) for target in EVENT_TYPES],
            "aid": [10, 10, 10, 20, 20, 20],
        }
    ).write_parquet(labels)
    destination = tmp_path / "rankers"

    result = train_ranker_suite(
        training,
        validation,
        labels,
        destination,
        rounds=3,
        early_stopping_rounds=1,
        training_query_limit=1,
        validation_query_limit=1,
    )

    assert 0.0 <= result.weighted_recall_at_20 <= 1.0
    assert result.training_rows == {target.value: 2 for target in EVENT_TYPES}
    assert result.validation_rows == {target.value: 2 for target in EVENT_TYPES}
    for target in EVENT_TYPES:
        assert (destination / f"{target.value}.json").exists()
        assert (destination / f"{target.value}-predictions.parquet").exists()
    assert (destination / "manifest.json").exists()


def test_inference_scores_candidate_shards_and_reuses_partial_outputs(tmp_path: Path) -> None:
    from otto_recsys.constants import EVENT_TYPES
    from otto_recsys.ranking import score_candidate_suite, train_ranker_suite

    training = tmp_path / "training"
    validation = tmp_path / "validation"
    for root in (training, validation):
        for target in EVENT_TYPES:
            directory = root / target.value
            directory.mkdir(parents=True)
            pl.DataFrame(
                {
                    "session": [1, 1, 2, 2],
                    "target": [target.value] * 4,
                    "aid": [10, 11, 20, 21],
                    "label": [1, 0, 1, 0],
                    "candidate_score": [2.0, 0.0, 2.0, 0.0],
                }
            ).write_parquet(directory / "part-000000.parquet")
    labels = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2, 2],
            "type": [target.value for _ in range(2) for target in EVENT_TYPES],
            "aid": [10, 10, 10, 20, 20, 20],
        }
    ).write_parquet(labels)
    models = tmp_path / "models"
    train_ranker_suite(training, validation, labels, models, rounds=2)

    destination = tmp_path / "predictions"
    result = score_candidate_suite(validation, models, destination, device="cpu")

    assert result.sessions == {target.value: 2 for target in EVENT_TYPES}
    for target in EVENT_TYPES:
        output = pl.read_parquet(destination / target.value / "part-000000.parquet")
        assert output.columns == ["session", "target", "aid", "score", "rank"]
        assert output.group_by("session").len()["len"].max() <= 20


def test_refit_combines_temporal_windows_as_distinct_queries(tmp_path: Path) -> None:
    from otto_recsys.constants import EVENT_TYPES
    from otto_recsys.ranking import refit_ranker_suite

    sources = (tmp_path / "first", tmp_path / "second")
    for source_index, root in enumerate(sources):
        for target in EVENT_TYPES:
            directory = root / target.value
            directory.mkdir(parents=True)
            pl.DataFrame(
                {
                    "session": [1, 1],
                    "target": [target.value] * 2,
                    "aid": [10 + source_index, 20 + source_index],
                    "label": [1, 0],
                    "candidate_score": [2.0, 0.0],
                }
            ).write_parquet(directory / "part-000000.parquet")

    destination = tmp_path / "refit"
    result = refit_ranker_suite(sources, destination, device="cpu", rounds=2, query_limit=2)

    assert result.training_rows == {target.value: 4 for target in EVENT_TYPES}
    for target in EVENT_TYPES:
        assert (destination / f"{target.value}.json").exists()
    assert (destination / "manifest.json").exists()
