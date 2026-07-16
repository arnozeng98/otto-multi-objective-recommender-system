from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol

from otto_recsys.candidates.merge import Candidate, CandidateInput, merge_candidates
from otto_recsys.constants import EventType
from otto_recsys.data.schemas import Session

TARGET_MATRIX_SOURCES: dict[EventType, tuple[str, ...]] = {
    EventType.CLICKS: ("adjacent_clicks", "time_decay"),
    EventType.CARTS: ("time_decay", "all_to_buy", "buy_to_buy", "recent_trend"),
    EventType.ORDERS: ("time_decay", "all_to_buy", "buy_to_buy", "recent_trend"),
}


class NeighborLookup(Protocol):
    def neighbors(self, source_aid: int) -> tuple[tuple[int, float, int], ...]: ...


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

    covisitation_inputs = 0
    for seed_rank, aid in enumerate(recency, start=1):
        for source in TARGET_MATRIX_SOURCES[target]:
            matrix = matrices.get(source)
            if matrix is None:
                continue
            for neighbor, score, neighbor_rank in matrix.neighbors(aid):
                if covisitation_budget is not None and covisitation_inputs >= covisitation_budget:
                    break
                inputs.append(
                    CandidateInput(
                        neighbor,
                        source,
                        score / seed_rank,
                        neighbor_rank,
                    )
                )
                covisitation_inputs += 1
            if covisitation_budget is not None and covisitation_inputs >= covisitation_budget:
                break
        if covisitation_budget is not None and covisitation_inputs >= covisitation_budget:
            break
    for rank, aid in enumerate(popular_aids, start=1):
        inputs.append(CandidateInput(aid, "popularity", 0.01, rank))
    return merge_candidates(inputs, budget)
