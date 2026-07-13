import json
from pathlib import Path

import polars as pl

from otto_recsys.candidates import target_candidates
from otto_recsys.candidates.baseline import baseline_candidates
from otto_recsys.candidates.materialize import materialize_candidates
from otto_recsys.candidates.merge import CandidateInput, merge_candidates
from otto_recsys.constants import EventType
from otto_recsys.covisitation import (
    CovisitationRule,
    build_covisitation,
    build_covisitation_suite,
    build_matrix_store_suite,
)
from otto_recsys.data.materialize import materialize_temporal_split
from otto_recsys.data.preprocess import convert_jsonl_to_parquet
from otto_recsys.data.schemas import Event, Session


def test_covisitation_respects_direction_and_decay() -> None:
    sessions = [
        Session(
            1,
            (
                Event(1, 0, EventType.CLICKS),
                Event(2, 1_000, EventType.CARTS),
                Event(3, 3_000, EventType.ORDERS),
            ),
        )
    ]
    rule = CovisitationRule("forward", forward_only=True, half_life_seconds=1.0)

    matrix = build_covisitation(sessions, rule)

    assert 1 in matrix
    assert 1 not in dict(matrix[2])
    assert dict(matrix[1])[2] > dict(matrix[1])[3]


def test_candidate_merge_deduplicates_and_enforces_budget() -> None:
    merged = merge_candidates(
        [
            CandidateInput(1, "history", 1.0, 1),
            CandidateInput(1, "covisit", 2.0, 2),
            CandidateInput(2, "popular", 0.1, 1),
        ],
        budget=1,
    )

    assert len(merged) == 1
    assert merged[0].aid == 1
    assert merged[0].sources == ("covisit", "history")
    assert [(item.source, item.score, item.rank) for item in merged[0].evidence] == [
        ("covisit", 2.0, 2),
        ("history", 1.0, 1),
    ]


def test_baseline_candidates_include_history_and_neighbors() -> None:
    session = Session(1, (Event(1, 0, EventType.CLICKS),))
    candidates = baseline_candidates(session, {1: ((2, 4.0),)}, [3], budget=3)

    assert {candidate.aid for candidate in candidates} == {1, 2, 3}


class _Lookup:
    def __init__(self, target: int) -> None:
        self.target = target

    def neighbors(self, source_aid: int) -> tuple[tuple[int, float, int], ...]:
        return ((self.target, float(source_aid), 1),)


def test_target_candidates_use_target_specific_matrices() -> None:
    session = Session(1, (Event(10, 0, EventType.CLICKS),))
    matrices = {
        "adjacent_clicks": _Lookup(20),
        "time_decay": _Lookup(30),
        "all_to_buy": _Lookup(40),
        "buy_to_buy": _Lookup(50),
        "recent_trend": _Lookup(60),
    }

    clicks = target_candidates(session, EventType.CLICKS, matrices, (), budget=10)
    orders = target_candidates(session, EventType.ORDERS, matrices, (), budget=10)

    assert {candidate.aid for candidate in clicks} == {10, 20, 30}
    assert {candidate.aid for candidate in orders} == {10, 30, 40, 50, 60}
    order_sources = {source for candidate in orders for source in candidate.sources}
    assert "adjacent_clicks" not in order_sources


def test_materialize_candidates_writes_labeled_target_partitions(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    source.write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {
                        "session": 1,
                        "events": [
                            {"aid": 10, "ts": 100, "type": "clicks"},
                            {"aid": 20, "ts": 200, "type": "carts"},
                            {"aid": 30, "ts": 300, "type": "orders"},
                        ],
                    },
                    {
                        "session": 2,
                        "events": [
                            {"aid": 20, "ts": 110, "type": "carts"},
                            {"aid": 30, "ts": 120, "type": "orders"},
                        ],
                    },
                ],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    parquet = tmp_path / "events.parquet"
    view = tmp_path / "view"
    matrices = tmp_path / "matrices"
    stores = tmp_path / "stores"
    candidates = tmp_path / "candidates"
    convert_jsonl_to_parquet(source, parquet, batch_events=1)
    materialize_temporal_split(parquet, view, 150)
    build_covisitation_suite(
        view / "matrix_events.parquet",
        matrices,
        max_neighbors=2,
        partitions=2,
        pair_buffer_size=1,
        batch_rows=1,
    )
    build_matrix_store_suite(matrices, stores)

    result = materialize_candidates(
        view,
        stores,
        candidates,
        budget=5,
        popularity_budget=2,
        batch_rows=2,
    )

    assert result.sessions == 1
    for target in EventType:
        frame = pl.read_parquet(candidates / target.value / "*.parquet")
        assert frame["session"].unique().to_list() == [1]
        assert frame["candidate_rank"].to_list() == list(range(1, len(frame) + 1))
        assert "source_history_score" in frame.columns
    assert result.positives["carts"] == 1
    assert result.positives["orders"] == 1
    assert result.positives["clicks"] == 0
