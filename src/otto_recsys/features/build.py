from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class CandidateFeatureContext:
    session_length: int
    aid_counts: Counter[int]
    aid_type_counts: dict[int, Counter[object]]
    aid_recency: dict[int, int]

    @classmethod
    def from_session(cls, session: Session) -> CandidateFeatureContext:
        aid_counts = Counter(event.aid for event in session.events)
        aid_type_counts: dict[int, Counter[object]] = {}
        aid_recency: dict[int, int] = {}
        for index, event in enumerate(reversed(session.events)):
            aid_type_counts.setdefault(event.aid, Counter())[event.type] += 1
            aid_recency.setdefault(event.aid, index)
        return cls(len(session.events), aid_counts, aid_type_counts, aid_recency)


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
        "aid_frequency": context.aid_counts[candidate.aid],
        "aid_recency_events": context.aid_recency.get(candidate.aid, context.session_length),
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
