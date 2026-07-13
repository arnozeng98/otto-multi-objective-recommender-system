from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from otto_recsys.candidates.baseline import baseline_candidates
from otto_recsys.constants import EVENT_TYPES, EventType
from otto_recsys.covisitation import CovisitationRule, build_covisitation
from otto_recsys.data.schemas import Event, Session
from otto_recsys.data.split import split_session
from otto_recsys.metrics import weighted_recall_at_k
from otto_recsys.submission import write_submission


def run_smoke(output_dir: Path) -> dict[str, object]:
    """Exercise the classical path deterministically on generated sessions."""
    sessions = [
        Session(1, (Event(10, 100, EventType.CLICKS), Event(20, 300, EventType.CLICKS))),
        Session(2, (Event(10, 110, EventType.CLICKS), Event(30, 310, EventType.ORDERS))),
        Session(3, (Event(11, 120, EventType.CLICKS), Event(20, 320, EventType.CARTS))),
    ]
    splits = [split_session(session, 250) for session in sessions]
    contexts = [split.context for split in splits if split.context is not None]
    matrix = build_covisitation(sessions, CovisitationRule("smoke", forward_only=True))
    popularity_counts = Counter(event.aid for session in sessions for event in session.events)
    popularity = [aid for aid, _ in popularity_counts.most_common()]

    predictions: dict[tuple[int, EventType], list[int]] = {}
    truth: dict[tuple[int, EventType], tuple[int, ...]] = {}
    for context, split in zip(contexts, splits, strict=True):
        candidates = baseline_candidates(context, matrix, popularity, budget=20)
        aids = [candidate.aid for candidate in candidates]
        for event_type in EVENT_TYPES:
            predictions[(context.session, event_type)] = aids
            truth[(context.session, event_type)] = split.labels[event_type]

    result = weighted_recall_at_k(predictions, truth)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = output_dir / "submission.csv"
    backfill = {event_type: popularity for event_type in EVENT_TYPES}
    session_ids = [session.session for session in contexts]
    rows = write_submission(submission, session_ids, predictions, backfill)
    report: dict[str, object] = {
        "weighted_recall_at_20": result.weighted,
        "per_type": {key.value: value for key, value in result.per_type.items()},
        "submission_rows": rows,
        "submission": str(submission),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
