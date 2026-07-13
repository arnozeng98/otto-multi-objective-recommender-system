from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from otto_recsys.constants import EventType

EVENT_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int64(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("type", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class Event:
    aid: int
    ts: int
    type: EventType


@dataclass(frozen=True, slots=True)
class Session:
    session: int
    events: tuple[Event, ...]
