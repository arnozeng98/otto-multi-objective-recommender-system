from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.data.schemas import Session


@dataclass(frozen=True, slots=True)
class CovisitationRule:
    name: str
    source_types: frozenset[EventType] = frozenset(EVENT_TYPES)
    target_types: frozenset[EventType] = frozenset(EVENT_TYPES)
    max_sequence_distance: int = 3
    max_time_seconds: float = 86_400.0
    half_life_seconds: float | None = 3_600.0
    forward_only: bool = False
    max_neighbors: int = 80
    deduplicate_pairs: bool = False
    target_type_weights: tuple[tuple[EventType, float], ...] = ()
    timestamp_weight_range: tuple[float, float, float, float] | None = None


def iter_weighted_pairs(
    session: Session, rule: CovisitationRule
) -> Iterable[tuple[int, int, float]]:
    """Yield weighted item pairs for one session and rule."""
    events = sorted(session.events, key=lambda event: event.ts)
    seen_pairs: set[tuple[int, int]] = set()
    target_type_weights = dict(rule.target_type_weights)
    for source_index, source in enumerate(events):
        if source.type not in rule.source_types:
            continue
        left = (
            source_index + 1
            if rule.forward_only
            else max(0, source_index - rule.max_sequence_distance)
        )
        right = min(len(events), source_index + rule.max_sequence_distance + 1)
        for target_index in range(left, right):
            if target_index == source_index:
                continue
            target = events[target_index]
            if target.type not in rule.target_types or target.aid == source.aid:
                continue
            pair = (source.aid, target.aid)
            if rule.deduplicate_pairs and pair in seen_pairs:
                continue
            time_delta_seconds = abs(target.ts - source.ts) / 1_000.0
            if time_delta_seconds > rule.max_time_seconds:
                continue
            weight = 1.0
            if rule.half_life_seconds is not None:
                weight = math.pow(0.5, time_delta_seconds / rule.half_life_seconds)
            if rule.timestamp_weight_range is not None:
                start, end, minimum, maximum = rule.timestamp_weight_range
                source_seconds = source.ts / 1_000.0
                progress = min(1.0, max(0.0, (source_seconds - start) / (end - start)))
                weight *= minimum + (maximum - minimum) * progress
            weight *= target_type_weights.get(target.type, 1.0)
            seen_pairs.add(pair)
            yield source.aid, target.aid, weight


def build_covisitation(
    sessions: Iterable[Session], rule: CovisitationRule
) -> dict[int, tuple[tuple[int, float], ...]]:
    """Build a deterministic top-K item-item matrix for one configurable rule."""
    scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for session in sessions:
        for source_aid, target_aid, weight in iter_weighted_pairs(session, rule):
            scores[source_aid][target_aid] += weight
    return {
        aid: tuple(
            sorted(neighbors.items(), key=lambda item: (-item[1], item[0]))[: rule.max_neighbors]
        )
        for aid, neighbors in scores.items()
    }
