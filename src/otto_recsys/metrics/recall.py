from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from otto_recsys.constants import EVENT_TYPES, TYPE_WEIGHTS, EventType

SessionTarget = tuple[int, EventType]


@dataclass(frozen=True)
class RecallResult:
    per_type: Mapping[EventType, float]
    weighted: float


def weighted_recall_at_k(
    predictions: Mapping[SessionTarget, Iterable[int]],
    ground_truth: Mapping[SessionTarget, Iterable[int]],
    k: int = 20,
) -> RecallResult:
    """Compute the competition's corpus-level weighted Recall@K exactly."""
    if not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")

    recalls: dict[EventType, float] = {}
    for event_type in EVENT_TYPES:
        hits = 0
        denominator = 0
        for key, truth_values in ground_truth.items():
            if key[1] != event_type:
                continue
            truth = set(truth_values)
            if not truth:
                continue
            predicted = list(dict.fromkeys(predictions.get(key, ())))[:k]
            hits += len(set(predicted).intersection(truth))
            denominator += min(k, len(truth))
        recalls[event_type] = hits / denominator if denominator else 0.0

    weighted = sum(TYPE_WEIGHTS[event_type] * recalls[event_type] for event_type in EVENT_TYPES)
    return RecallResult(per_type=recalls, weighted=weighted)
