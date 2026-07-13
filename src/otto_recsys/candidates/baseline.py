from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from otto_recsys.candidates.merge import Candidate, CandidateInput, merge_candidates
from otto_recsys.data.schemas import Session


def baseline_candidates(
    session: Session,
    covisitation: Mapping[int, Sequence[tuple[int, float]]],
    popular_aids: Sequence[int],
    *,
    budget: int,
) -> tuple[Candidate, ...]:
    inputs: list[CandidateInput] = []
    recency = list(dict.fromkeys(event.aid for event in reversed(session.events)))
    frequency = Counter(event.aid for event in session.events)
    for rank, aid in enumerate(recency, start=1):
        inputs.append(CandidateInput(aid, "history", float(frequency[aid]), rank))
        for neighbor_rank, (neighbor, score) in enumerate(covisitation.get(aid, ()), start=1):
            inputs.append(CandidateInput(neighbor, "covisitation", score / rank, neighbor_rank))
    for rank, aid in enumerate(popular_aids, start=1):
        inputs.append(CandidateInput(aid, "popularity", 0.01, rank))
    return merge_candidates(inputs, budget)
