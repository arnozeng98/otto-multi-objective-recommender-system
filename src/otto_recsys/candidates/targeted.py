from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol

from otto_recsys.candidates.merge import Candidate, CandidateInput, merge_candidates
from otto_recsys.constants import EventType
from otto_recsys.data.schemas import Session

TARGET_MATRIX_SOURCES: dict[EventType, tuple[str, ...]] = {
    EventType.CLICKS: ("clicks_all_to_all", "adjacent_clicks", "time_decay"),
    EventType.CARTS: (
        "carts_orders",
        "buy_to_buy",
        "time_decay",
        "all_to_buy",
        "recent_trend",
    ),
    EventType.ORDERS: (
        "carts_orders",
        "buy_to_buy",
        "time_decay",
        "all_to_buy",
        "recent_trend",
    ),
}


class NeighborLookup(Protocol):
    def neighbors(self, source_aid: int) -> tuple[tuple[int, float, int], ...]: ...


def _rank_by_score(scores: Mapping[int, float], insertion: Mapping[int, int]) -> list[int]:
    return sorted(scores, key=lambda aid: (-scores[aid], insertion[aid], aid))


def _public_reference_inputs(
    session: Session,
    target: EventType,
    matrices: Mapping[str, NeighborLookup],
    popular_aids: Sequence[int],
    budget: int,
) -> list[CandidateInput]:
    events = sorted(session.events, key=lambda event: event.ts)
    recency = list(dict.fromkeys(event.aid for event in reversed(events)))
    insertion: dict[int, int] = {}
    scores: dict[int, float] = {}

    def add(aid: int, score: float) -> None:
        insertion.setdefault(aid, len(insertion))
        scores[aid] = scores.get(aid, 0.0) + score

    if len(recency) >= 20:
        start = 0.1 if target == EventType.CLICKS else 0.5
        denominator = max(1, len(events) - 1)
        type_weights = {
            EventType.CLICKS: 1.0,
            EventType.CARTS: 6.0,
            EventType.ORDERS: 3.0,
        }
        for index, event in enumerate(events):
            exponent = start + (1.0 - start) * index / denominator
            add(event.aid, (2.0**exponent - 1.0) * type_weights[event.type])
        history = _rank_by_score(scores, insertion)
    else:
        history = recency

    matrix_scores: dict[int, float] = {}
    matrix_insertion: dict[int, int] = {}
    matrix_sources: dict[int, list[str]] = {}
    sources: tuple[tuple[str, list[int]], ...]
    if target == EventType.CLICKS:
        sources = (("clicks_all_to_all", recency),)
    else:
        buys = list(
            dict.fromkeys(
                event.aid
                for event in reversed(events)
                if event.type in (EventType.CARTS, EventType.ORDERS)
            )
        )
        sources = (("buy_to_buy", buys), ("carts_orders", recency))
    for source, seeds in sources:
        matrix = matrices.get(source)
        if matrix is None:
            continue
        for seed in seeds:
            for aid, _score, _rank in matrix.neighbors(seed):
                matrix_insertion.setdefault(aid, len(matrix_insertion))
                matrix_scores[aid] = matrix_scores.get(aid, 0.0) + 1.0
                sources_for_aid = matrix_sources.setdefault(aid, [])
                if source not in sources_for_aid:
                    sources_for_aid.append(source)

    seen = set(history)
    neighbors = [aid for aid in _rank_by_score(matrix_scores, matrix_insertion) if aid not in seen]
    ordered = list(dict.fromkeys([*history, *neighbors, *popular_aids]))[:budget]
    history_set = set(history)
    neighbor_set = set(neighbors)
    inputs: list[CandidateInput] = []
    for rank, aid in enumerate(ordered, start=1):
        if aid in history_set:
            primary_source = "history"
        elif aid in neighbor_set:
            primary_source = matrix_sources[aid][0]
        else:
            primary_source = "popularity"
        inputs.append(CandidateInput(aid, primary_source, float(len(ordered) - rank + 1), rank))
        for source in matrix_sources.get(aid, []):
            if source != primary_source:
                inputs.append(CandidateInput(aid, source, 0.0, rank))
    return inputs


def target_candidates(
    session: Session,
    target: EventType,
    matrices: Mapping[str, NeighborLookup],
    popular_aids: Sequence[int],
    *,
    budget: int,
    history_budget: int | None = None,
    covisitation_budget: int | None = None,
    max_events_per_session: int | None = None,
) -> tuple[Candidate, ...]:
    """Generate target-aware candidates while preserving per-source evidence."""
    if "clicks_all_to_all" in matrices:
        return merge_candidates(
            _public_reference_inputs(session, target, matrices, popular_aids, budget),
            budget,
        )
    inputs: list[CandidateInput] = []
    events = (
        session.events[-max_events_per_session:]
        if max_events_per_session is not None
        else session.events
    )
    recency = list(dict.fromkeys(event.aid for event in reversed(events)))
    frequency = Counter(event.aid for event in events)
    history = recency[:history_budget] if history_budget is not None else recency
    for history_rank, aid in enumerate(history, start=1):
        inputs.append(CandidateInput(aid, "history", float(frequency[aid]), history_rank))

    available_sources = [source for source in TARGET_MATRIX_SOURCES[target] if source in matrices]
    source_quotas: dict[str, int | None] = {source: None for source in available_sources}
    if covisitation_budget is not None and available_sources:
        quotient, remainder = divmod(covisitation_budget, len(available_sources))
        source_quotas = {
            source: quotient + (source_index < remainder)
            for source_index, source in enumerate(available_sources)
        }

    for source in available_sources:
        matrix = matrices[source]
        source_scores: dict[int, float] = {}
        source_best_ranks: dict[int, int] = {}
        for seed_rank, aid in enumerate(recency, start=1):
            for neighbor, _raw_score, neighbor_rank in matrix.neighbors(aid):
                source_scores[neighbor] = source_scores.get(neighbor, 0.0) + 1.0 / (
                    seed_rank * (60 + neighbor_rank)
                )
                source_best_ranks[neighbor] = min(
                    source_best_ranks.get(neighbor, neighbor_rank), neighbor_rank
                )
        ranked = sorted(
            source_scores,
            key=lambda aid: (-source_scores[aid], source_best_ranks[aid], aid),
        )
        quota = source_quotas[source]
        if quota is not None:
            ranked = ranked[:quota]
        for source_rank, aid in enumerate(ranked, start=1):
            inputs.append(CandidateInput(aid, source, source_scores[aid], source_rank))
    for rank, aid in enumerate(popular_aids, start=1):
        inputs.append(CandidateInput(aid, "popularity", 0.01, rank))
    return merge_candidates(inputs, budget)
