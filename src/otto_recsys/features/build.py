from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from otto_recsys.candidates import Candidate
from otto_recsys.constants import EVENT_TYPES
from otto_recsys.data.schemas import Session

SOURCE_NAMES = (
    "history",
    "popularity",
    "clicks_all_to_all",
    "carts_orders",
    "adjacent_clicks",
    "time_decay",
    "all_to_buy",
    "buy_to_buy",
    "recent_trend",
)


@dataclass(frozen=True, slots=True)
class CandidateFeatureContext:
    session_length: int
    session_duration_ms: int
    session_mean_gap_ms: float
    session_last_gap_ms: int
    session_type_counts: Counter[object]
    aid_counts: Counter[int]
    aid_type_counts: dict[int, Counter[object]]
    aid_recency: dict[int, int]
    aid_first_position: dict[int, int]
    aid_last_position: dict[int, int]
    aid_time_recency_ms: dict[int, int]

    @classmethod
    def from_session(cls, session: Session) -> CandidateFeatureContext:
        aid_counts = Counter(event.aid for event in session.events)
        aid_type_counts: dict[int, Counter[object]] = {}
        aid_recency: dict[int, int] = {}
        aid_first_position: dict[int, int] = {}
        aid_last_position: dict[int, int] = {}
        latest_timestamp = session.events[-1].ts
        for index, event in enumerate(reversed(session.events)):
            aid_type_counts.setdefault(event.aid, Counter())[event.type] += 1
            aid_recency.setdefault(event.aid, index)
        for index, event in enumerate(session.events):
            aid_first_position.setdefault(event.aid, index)
            aid_last_position[event.aid] = index
        gaps = [
            current.ts - previous.ts
            for previous, current in zip(session.events, session.events[1:], strict=False)
        ]
        return cls(
            session_length=len(session.events),
            session_duration_ms=latest_timestamp - session.events[0].ts,
            session_mean_gap_ms=sum(gaps) / len(gaps) if gaps else 0.0,
            session_last_gap_ms=gaps[-1] if gaps else 0,
            session_type_counts=Counter(event.type for event in session.events),
            aid_counts=aid_counts,
            aid_type_counts=aid_type_counts,
            aid_recency=aid_recency,
            aid_first_position=aid_first_position,
            aid_last_position=aid_last_position,
            aid_time_recency_ms={
                aid: latest_timestamp - session.events[position].ts
                for aid, position in aid_last_position.items()
            },
        )


def build_candidate_features(
    session: Session,
    candidate: Candidate,
    *,
    context: CandidateFeatureContext | None = None,
) -> dict[str, float | int]:
    """Build compact session-item features without using future events."""
    context = context or CandidateFeatureContext.from_session(session)
    type_counts = context.aid_type_counts.get(candidate.aid, Counter())
    features: dict[str, float | int] = {
        "candidate_score": candidate.score,
        "candidate_best_rank": candidate.best_rank,
        "candidate_source_count": len(candidate.sources),
        "session_length": context.session_length,
        "session_unique_aids": len(context.aid_counts),
        "session_duplicate_rate": 1.0 - len(context.aid_counts) / context.session_length,
        "session_duration_ms": context.session_duration_ms,
        "session_mean_gap_ms": context.session_mean_gap_ms,
        "session_last_gap_ms": context.session_last_gap_ms,
        "aid_frequency": context.aid_counts[candidate.aid],
        "aid_recency_events": context.aid_recency.get(candidate.aid, context.session_length),
        "aid_time_recency_ms": context.aid_time_recency_ms.get(
            candidate.aid, context.session_duration_ms
        ),
        "aid_first_position": context.aid_first_position.get(candidate.aid, context.session_length),
        "aid_last_position": context.aid_last_position.get(candidate.aid, context.session_length),
    }
    for event_type in EVENT_TYPES:
        features[f"session_{event_type.value}_count"] = context.session_type_counts[event_type]
        features[f"aid_{event_type.value}_count"] = type_counts[event_type]
    evidence = {item.source: item for item in candidate.evidence}
    for source in SOURCE_NAMES:
        item = evidence.get(source)
        features[f"source_{source}_present"] = int(item is not None)
        features[f"source_{source}_score"] = item.score if item is not None else 0.0
        features[f"source_{source}_rank"] = item.rank if item is not None else 0
    return features
