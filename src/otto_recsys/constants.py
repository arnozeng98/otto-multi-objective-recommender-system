from enum import StrEnum


class EventType(StrEnum):
    CLICKS = "clicks"
    CARTS = "carts"
    ORDERS = "orders"


EVENT_TYPES = tuple(EventType)
TYPE_WEIGHTS = {
    EventType.CLICKS: 0.10,
    EventType.CARTS: 0.30,
    EventType.ORDERS: 0.60,
}
