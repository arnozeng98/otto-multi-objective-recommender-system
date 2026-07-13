from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateInput:
    aid: int
    source: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    source: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class Candidate:
    aid: int
    score: float
    sources: tuple[str, ...]
    best_rank: int
    evidence: tuple[CandidateEvidence, ...] = ()


@dataclass(slots=True)
class _MergedState:
    score: float
    source_scores: dict[str, float]
    source_ranks: dict[str, int]
    best_rank: int


def merge_candidates(inputs: Iterable[CandidateInput], budget: int) -> tuple[Candidate, ...]:
    """Fuse candidate sources with reciprocal-rank normalization and deterministic ties."""
    if budget < 1:
        raise ValueError("budget must be positive")
    merged: dict[int, _MergedState] = {}
    for candidate in inputs:
        state = merged.setdefault(
            candidate.aid,
            _MergedState(
                score=0.0,
                source_scores={},
                source_ranks={},
                best_rank=candidate.rank,
            ),
        )
        state.score += candidate.score + 1.0 / (60 + candidate.rank)
        state.source_scores[candidate.source] = (
            state.source_scores.get(candidate.source, 0.0) + candidate.score
        )
        state.source_ranks[candidate.source] = min(
            state.source_ranks.get(candidate.source, candidate.rank),
            candidate.rank,
        )
        state.best_rank = min(state.best_rank, candidate.rank)
    result = [
        Candidate(
            aid=aid,
            score=state.score,
            sources=tuple(sorted(state.source_scores)),
            best_rank=state.best_rank,
            evidence=tuple(
                CandidateEvidence(
                    source=source,
                    score=state.source_scores[source],
                    rank=state.source_ranks[source],
                )
                for source in sorted(state.source_scores)
            ),
        )
        for aid, state in merged.items()
    ]
    return tuple(sorted(result, key=lambda item: (-item.score, item.best_rank, item.aid))[:budget])
