from __future__ import annotations

from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.covisitation.builder import CovisitationRule

BUY_TYPES = frozenset((EventType.CARTS, EventType.ORDERS))
PUBLIC_TRAIN_START_SECONDS = 1_659_304_800.0
PUBLIC_TRAIN_END_SECONDS = 1_662_328_791.0


def public_v575_rules(max_neighbors: int = 40) -> dict[str, CovisitationRule]:
    """Return the three co-visitation matrices from the public 0.575 baseline."""
    return {
        "clicks_all_to_all": CovisitationRule(
            name="clicks_all_to_all",
            max_sequence_distance=29,
            max_time_seconds=86_400.0,
            half_life_seconds=None,
            max_neighbors=max_neighbors,
            deduplicate_pairs=True,
            timestamp_weight_range=(
                PUBLIC_TRAIN_START_SECONDS,
                PUBLIC_TRAIN_END_SECONDS,
                1.0,
                4.0,
            ),
        ),
        "carts_orders": CovisitationRule(
            name="carts_orders",
            target_types=BUY_TYPES,
            max_sequence_distance=29,
            max_time_seconds=86_400.0,
            half_life_seconds=None,
            max_neighbors=max_neighbors,
            deduplicate_pairs=True,
            target_type_weights=((EventType.CARTS, 6.0), (EventType.ORDERS, 3.0)),
        ),
        "buy_to_buy": CovisitationRule(
            name="buy_to_buy",
            source_types=BUY_TYPES,
            target_types=BUY_TYPES,
            max_sequence_distance=29,
            max_time_seconds=14 * 86_400.0,
            half_life_seconds=None,
            max_neighbors=max_neighbors,
            deduplicate_pairs=True,
        ),
    }


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
