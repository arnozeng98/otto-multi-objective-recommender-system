from __future__ import annotations

from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.covisitation.builder import CovisitationRule

BUY_TYPES = frozenset((EventType.CARTS, EventType.ORDERS))


def default_rules(max_neighbors: int = 80) -> dict[str, CovisitationRule]:
    """Return the five high-value, resource-bounded co-visitation rules."""
    return {
        "adjacent_clicks": CovisitationRule(
            name="adjacent_clicks",
            source_types=frozenset((EventType.CLICKS,)),
            target_types=frozenset((EventType.CLICKS,)),
            max_sequence_distance=1,
            max_time_seconds=86_400.0,
            half_life_seconds=None,
            forward_only=True,
            max_neighbors=max_neighbors,
        ),
        "time_decay": CovisitationRule(
            name="time_decay",
            source_types=frozenset(EVENT_TYPES),
            target_types=frozenset(EVENT_TYPES),
            max_sequence_distance=6,
            max_time_seconds=86_400.0,
            half_life_seconds=3_600.0,
            max_neighbors=max_neighbors,
        ),
        "all_to_buy": CovisitationRule(
            name="all_to_buy",
            source_types=frozenset(EVENT_TYPES),
            target_types=BUY_TYPES,
            max_sequence_distance=6,
            max_time_seconds=7_200.0,
            half_life_seconds=3_600.0,
            forward_only=True,
            max_neighbors=max_neighbors,
        ),
        "buy_to_buy": CovisitationRule(
            name="buy_to_buy",
            source_types=BUY_TYPES,
            target_types=BUY_TYPES,
            max_sequence_distance=3,
            max_time_seconds=86_400.0,
            half_life_seconds=7_200.0,
            max_neighbors=max_neighbors,
        ),
        "recent_trend": CovisitationRule(
            name="recent_trend",
            source_types=frozenset(EVENT_TYPES),
            target_types=BUY_TYPES,
            max_sequence_distance=3,
            max_time_seconds=86_400.0,
            half_life_seconds=1_800.0,
            forward_only=True,
            max_neighbors=max_neighbors,
        ),
    }
