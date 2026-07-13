from __future__ import annotations

from collections import Counter

from otto_recsys.candidates import Candidate
from otto_recsys.constants import EVENT_TYPES
from otto_recsys.data.schemas import Session

SOURCE_NAMES = (
    "history",
    "popularity",
    "adjacent_clicks",
    "time_decay",
    "all_to_buy",
    "buy_to_buy",
    "recent_trend",
)


def build_candidate_features(session: Session, candidate: Candidate) -> dict[str, float | int]:
    """Build compact session-item features without using future events."""
    counts = Counter(event.aid for event in session.events)
    type_counts = Counter(event.type for event in session.events if event.aid == candidate.aid)
    last_index = next(
        (
            index
            for index, event in enumerate(reversed(session.events))
            if event.aid == candidate.aid
        ),
        len(session.events),
    )
    features: dict[str, float | int] = {
        "candidate_score": candidate.score,
        "candidate_best_rank": candidate.best_rank,
        "candidate_source_count": len(candidate.sources),
        "session_length": len(session.events),
        "session_unique_aids": len(counts),
        "aid_frequency": counts[candidate.aid],
        "aid_recency_events": last_index,
    }
    for event_type in EVENT_TYPES:
        features[f"aid_{event_type.value}_count"] = type_counts[event_type]
    evidence = {item.source: item for item in candidate.evidence}
    for source in SOURCE_NAMES:
        item = evidence.get(source)
        features[f"source_{source}_present"] = int(item is not None)
        features[f"source_{source}_score"] = item.score if item is not None else 0.0
        features[f"source_{source}_rank"] = item.rank if item is not None else 0
    return features
