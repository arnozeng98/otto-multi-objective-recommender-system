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
) -> tuple[Candidate, ...]:
    """Generate target-aware candidates while preserving per-source evidence."""
    inputs: list[CandidateInput] = []
    recency = list(dict.fromkeys(event.aid for event in reversed(session.events)))
    frequency = Counter(event.aid for event in session.events)
    for seed_rank, aid in enumerate(recency, start=1):
        inputs.append(CandidateInput(aid, "history", float(frequency[aid]), seed_rank))
        for source in TARGET_MATRIX_SOURCES[target]:
            matrix = matrices.get(source)
            if matrix is None:
                continue
            for neighbor, score, neighbor_rank in matrix.neighbors(aid):
                inputs.append(
                    CandidateInput(
                        neighbor,
                        source,
                        score / seed_rank,
                        neighbor_rank,
                    )
                )
    for rank, aid in enumerate(popular_aids, start=1):
        inputs.append(CandidateInput(aid, "popularity", 0.01, rank))
    return merge_candidates(inputs, budget)
