from __future__ import annotations

from dataclasses import dataclass

from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.data.schemas import Session


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    context: Session | None
    labels: dict[EventType, tuple[int, ...]]


def split_session(
    session: Session,
    cutoff_timestamp_ms: int,
    *,
    label_end_timestamp_ms: int | None = None,
) -> TemporalSplit:
    """Split one session at a global timestamp with competition label semantics."""
    if label_end_timestamp_ms is not None and label_end_timestamp_ms <= cutoff_timestamp_ms:
        raise ValueError("label_end_timestamp_ms must be greater than cutoff_timestamp_ms")
    ordered = sorted(session.events, key=lambda event: event.ts)
    context_events = tuple(event for event in ordered if event.ts < cutoff_timestamp_ms)
    future_events = tuple(
        event
        for event in ordered
        if event.ts >= cutoff_timestamp_ms
        and (label_end_timestamp_ms is None or event.ts < label_end_timestamp_ms)
    )
    labels: dict[EventType, tuple[int, ...]] = {}
    for event_type in EVENT_TYPES:
        aids = [event.aid for event in future_events if event.type == event_type]
        if event_type == EventType.CLICKS:
            aids = aids[:1]
        labels[event_type] = tuple(dict.fromkeys(aids))
    context = Session(session.session, context_events) if context_events else None
    return TemporalSplit(context=context, labels=labels)


def split_official_session(session: Session, context_length: int) -> TemporalSplit:
    """Split a session with the organizer's random-event test-set semantics."""
    ordered = tuple(sorted(session.events, key=lambda event: event.ts))
    maximum = len(ordered) - 1
    if not 1 <= context_length <= maximum:
        raise ValueError(f"context_length must be between 1 and {maximum}")

    context_events = ordered[:context_length]
    future_events = ordered[context_length:]
    next_click = next(
        (event.aid for event in future_events if event.type == EventType.CLICKS),
        None,
    )
    labels: dict[EventType, tuple[int, ...]] = {
        EventType.CLICKS: (next_click,) if next_click is not None else (),
        EventType.CARTS: tuple(
            dict.fromkeys(event.aid for event in future_events if event.type == EventType.CARTS)
        ),
        EventType.ORDERS: tuple(
            dict.fromkeys(event.aid for event in future_events if event.type == EventType.ORDERS)
        ),
    }
    return TemporalSplit(
        context=Session(session.session, context_events),
        labels=labels,
    )
