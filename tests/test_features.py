from otto_recsys.candidates import Candidate, CandidateEvidence
from otto_recsys.constants import EventType
from otto_recsys.data.schemas import Event, Session
from otto_recsys.features import build_candidate_features


def test_features_use_only_supplied_context() -> None:
    session = Session(
        1,
        (Event(10, 100, EventType.CLICKS), Event(10, 200, EventType.CARTS)),
    )
    candidate = Candidate(10, 2.0, ("history",), 1, (CandidateEvidence("history", 2.0, 1),))

    features = build_candidate_features(session, candidate)

    assert features["aid_frequency"] == 2
    assert features["aid_clicks_count"] == 1
    assert features["aid_carts_count"] == 1
    assert features["aid_recency_events"] == 0
    assert features["source_history_present"] == 1
    assert features["source_history_score"] == 2.0
    assert features["source_all_to_buy_present"] == 0
