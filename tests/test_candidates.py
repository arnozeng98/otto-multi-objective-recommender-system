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
    iter_weighted_pairs,
)
from otto_recsys.covisitation.rules import public_v575_rules
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


def test_public_rules_match_reference_pair_weights() -> None:
    start_ms = 1_659_304_800_000
    session = Session(
        1,
        (
            Event(10, start_ms, EventType.CLICKS),
            Event(20, start_ms + 1_000, EventType.CARTS),
            Event(20, start_ms + 2_000, EventType.CARTS),
            Event(30, start_ms + 3_000, EventType.ORDERS),
        ),
    )
    rules = public_v575_rules(max_neighbors=40)

    clicks = list(iter_weighted_pairs(session, rules["clicks_all_to_all"]))
    carts_orders = list(iter_weighted_pairs(session, rules["carts_orders"]))
    buy_to_buy = list(iter_weighted_pairs(session, rules["buy_to_buy"]))

    assert [(source, target) for source, target, _ in clicks] == [
        (10, 20),
        (10, 30),
        (20, 10),
        (20, 30),
        (30, 10),
        (30, 20),
    ]
    assert clicks[0][2] == 1.0
    assert clicks[-1][2] > clicks[0][2]
    assert carts_orders == [
        (10, 20, 6.0),
        (10, 30, 3.0),
        (20, 30, 3.0),
        (30, 20, 6.0),
    ]
    assert buy_to_buy == [(20, 30, 1.0), (30, 20, 1.0)]


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


class _RankedLookup:
    def __init__(self, targets: tuple[int, ...]) -> None:
        self.targets = targets

    def neighbors(self, source_aid: int) -> tuple[tuple[int, float, int], ...]:
        return tuple(
            (target, float(len(self.targets) - rank + 1), rank)
            for rank, target in enumerate(self.targets, start=1)
        )


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


def test_target_candidates_enforce_source_and_session_budgets() -> None:
    session = Session(
        1,
        (
            Event(10, 0, EventType.CLICKS),
            Event(20, 1, EventType.CLICKS),
            Event(30, 2, EventType.CLICKS),
        ),
    )
    matrices = {"adjacent_clicks": _Lookup(40), "time_decay": _Lookup(50)}

    candidates = target_candidates(
        session,
        EventType.CLICKS,
        matrices,
        (),
        budget=10,
        history_budget=1,
        covisitation_budget=1,
        max_events_per_session=2,
    )

    assert {candidate.aid for candidate in candidates} == {30, 40}
    assert all(candidate.aid != 10 for candidate in candidates)


def test_target_candidates_split_unique_budget_across_sources() -> None:
    session = Session(1, (Event(10, 0, EventType.CLICKS),))
    matrices = {
        "adjacent_clicks": _RankedLookup((20, 21)),
        "time_decay": _RankedLookup((30, 31)),
    }

    candidates = target_candidates(
        session,
        EventType.CLICKS,
        matrices,
        (),
        budget=10,
        history_budget=0,
        covisitation_budget=2,
    )

    assert {candidate.aid for candidate in candidates} == {20, 30}


def test_public_click_rules_preserve_reference_top_order() -> None:
    session = Session(
        1,
        (
            Event(10, 0, EventType.CLICKS),
            Event(20, 1, EventType.CLICKS),
        ),
    )
    matrices = {
        "clicks_all_to_all": _RankedLookup((30, 50)),
    }

    candidates = target_candidates(
        session,
        EventType.CLICKS,
        matrices,
        (40,),
        budget=5,
    )

    assert [candidate.aid for candidate in candidates] == [20, 10, 30, 50, 40]


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


def test_materialize_candidates_supports_unlabeled_inference(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    source.write_text(
        json.dumps(
            {
                "session": 1,
                "events": [
                    {"aid": 10, "ts": 100, "type": "clicks"},
                    {"aid": 20, "ts": 200, "type": "orders"},
                ],
            }
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
        budget=3,
        popularity_budget=2,
        query_contexts=parquet,
        popularity_events=view / "matrix_events.parquet",
        include_labels=False,
        batch_rows=2,
    )

    assert result.sessions == 1
    assert result.positives == {target.value: 0 for target in EventType}
    for target in EventType:
        frame = pl.read_parquet(candidates / target.value / "*.parquet")
        assert frame["session"].unique().to_list() == [1]
        assert frame["label"].sum() == 0


def test_candidate_retention_keeps_all_positives(tmp_path: Path) -> None:
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
                            {"aid": 20, "ts": 200, "type": "orders"},
                            {"aid": 30, "ts": 300, "type": "orders"},
                        ],
                    },
                    {
                        "session": 2,
                        "events": [
                            {"aid": 20, "ts": 110, "type": "orders"},
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
        max_neighbors=3,
        partitions=2,
        pair_buffer_size=1,
        batch_rows=1,
    )
    build_matrix_store_suite(matrices, stores)

    materialize_candidates(
        view,
        stores,
        candidates,
        budget=3,
        popularity_budget=3,
        retention_limit=1,
        batch_rows=2,
    )

    orders = pl.read_parquet(candidates / "orders" / "*.parquet")
    assert orders.filter(pl.col("label") == 0)["candidate_rank"].max() <= 1
    assert set(orders.filter(pl.col("label") > 0)["aid"]) == {20, 30}


def test_candidate_materialization_resumes_synchronized_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    records = [
        {
            "session": session,
            "events": [
                {"aid": 10, "ts": 100, "type": "clicks"},
                {"aid": 20 + session, "ts": 200, "type": "orders"},
            ],
        }
        for session in range(1, 5)
    ]
    source.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    parquet = tmp_path / "events.parquet"
    view = tmp_path / "view"
    matrices = tmp_path / "matrices"
    stores = tmp_path / "stores"
    destination = tmp_path / "candidates"
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

    materialize_candidates(
        view,
        stores,
        destination,
        budget=3,
        popularity_budget=2,
        batch_rows=3,
    )
    completed = {
        target.value: pl.read_parquet(destination / target.value / "*.parquet")
        for target in EventType
    }
    destination.rename(destination.with_name("candidates.tmp"))

    resumed = materialize_candidates(
        view,
        stores,
        destination,
        budget=3,
        popularity_budget=2,
        batch_rows=3,
    )

    assert resumed.sessions == 4
    for target in EventType:
        actual = pl.read_parquet(destination / target.value / "*.parquet")
        assert actual.equals(completed[target.value])


def test_parallel_candidate_materialization_matches_single_process(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    records = [
        {
            "session": session,
            "events": [
                {"aid": 10, "ts": 100, "type": "clicks"},
                {"aid": 20 + session, "ts": 200, "type": "orders"},
            ],
        }
        for session in range(1, 7)
    ]
    source.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    parquet = tmp_path / "events.parquet"
    view = tmp_path / "view"
    matrices = tmp_path / "matrices"
    stores = tmp_path / "stores"
    single = tmp_path / "single"
    parallel = tmp_path / "parallel"
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

    expected = materialize_candidates(
        view,
        stores,
        single,
        budget=3,
        popularity_budget=2,
        retention_limit=2,
        batch_rows=3,
    )
    actual = materialize_candidates(
        view,
        stores,
        parallel,
        budget=3,
        popularity_budget=2,
        retention_limit=2,
        workers=2,
        chunk_sessions=2,
        batch_rows=3,
    )

    assert actual == expected
    for target in EventType:
        expected_frame = pl.read_parquet(single / target.value / "*.parquet").sort(
            "session", "candidate_rank"
        )
        actual_frame = pl.read_parquet(parallel / target.value / "*.parquet").sort(
            "session", "candidate_rank"
        )
        assert actual_frame.equals(expected_frame)

    parallel.rename(parallel.with_name("parallel.tmp"))
    resumed = materialize_candidates(
        view,
        stores,
        parallel,
        budget=3,
        popularity_budget=2,
        retention_limit=2,
        workers=2,
        chunk_sessions=2,
        batch_rows=3,
    )
    assert resumed == expected


def test_candidate_materialization_salvages_legacy_corrupt_tail(tmp_path: Path) -> None:
    source = tmp_path / "sessions.jsonl"
    records = [
        {
            "session": session,
            "events": [
                {"aid": 10, "ts": 100, "type": "clicks"},
                {"aid": 30 + session, "ts": 200, "type": "orders"},
            ],
        }
        for session in range(1, 5)
    ]
    source.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    parquet = tmp_path / "events.parquet"
    view = tmp_path / "view"
    matrices = tmp_path / "matrices"
    stores = tmp_path / "stores"
    destination = tmp_path / "candidates"
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
    materialize_candidates(
        view,
        stores,
        destination,
        budget=3,
        popularity_budget=2,
        batch_rows=1,
    )
    expected = {
        target.value: pl.read_parquet(destination / target.value / "*.parquet")
        for target in EventType
    }
    temporary = destination.with_name("candidates.tmp")
    destination.rename(temporary)
    (temporary / "checkpoint.json").unlink()
    cart_tail = sorted((temporary / "carts").glob("part-*.parquet"))[-1]
    cart_tail.write_bytes(b"interrupted")

    result = materialize_candidates(
        view,
        stores,
        destination,
        budget=3,
        popularity_budget=2,
        batch_rows=1,
    )

    assert result.sessions == 4
    for target in EventType:
        actual = pl.read_parquet(destination / target.value / "*.parquet")
        assert actual.equals(expected[target.value])
