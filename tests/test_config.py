from pathlib import Path

from otto_recsys.config import load_config


def test_smoke_config_inherits_base() -> None:
    config = load_config(Path("configs/smoke.yaml"))

    assert config.project.seed == 2026
    assert config.ranking.device == "cpu"
    assert config.ranking.rounds == 20
    assert config.candidates.total_budget == 250
    assert config.covisitation.profile == "public_v575"
    assert config.validation.strategy == "official_random_event"
    assert config.validation.minimum_rules_recall_at_20 == 0.54
    assert config.validation.training_cutoff_timestamp_ms < config.validation.cutoff_timestamp_ms
