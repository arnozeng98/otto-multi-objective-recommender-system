from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from otto_recsys.constants import EVENT_TYPES


@dataclass(frozen=True, slots=True)
class TargetCandidateRecall:
    denominator: int
    recall_at: dict[int, float]
    source_recall: dict[str, float]
    source_marginal_recall: dict[str, float]
    source_union_recall: float
    session_length_recall_at_20: dict[str, float]
    positive_rank_histogram: dict[int, int]


@dataclass(frozen=True, slots=True)
class CandidateRecallReport:
    targets: dict[str, TargetCandidateRecall]
    weighted_recall_at_20: float
    weighted_recall_at_100: float


def _length_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length <= 5:
        return "2-5"
    if length <= 20:
        return "6-20"
    return "21+"


def _denominators(labels: Path) -> dict[str, int]:
    frame = (
        pl.scan_parquet(labels)
        .group_by("session", "type")
        .agg(pl.col("aid").n_unique().clip(upper_bound=20).alias("count"))
        .group_by("type")
        .agg(pl.col("count").sum())
        .collect(engine="streaming")
    )
    return {str(event_type): int(count) for event_type, count in frame.iter_rows()}


def evaluate_candidate_recall(
    candidates: Path,
    labels: Path,
    *,
    cutoffs: tuple[int, ...] = (20, 50, 80, 100, 150),
) -> CandidateRecallReport:
    """Measure fusion coverage and source-level positive recall from candidate shards."""
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain positive values")
    denominators = _denominators(labels)
    reports: dict[str, TargetCandidateRecall] = {}
    weights = {
        target.value: weight for target, weight in zip(EVENT_TYPES, (0.1, 0.3, 0.6), strict=True)
    }

    for target in EVENT_TYPES:
        frame = pl.scan_parquet(candidates / target.value / "*.parquet")
        schema = frame.collect_schema().names()
        sources = sorted(
            column.removeprefix("source_").removesuffix("_present")
            for column in schema
            if column.startswith("source_") and column.endswith("_present")
        )
        denominator = denominators.get(target.value, 0)
        positives = frame.filter(pl.col("label") > 0)
        recall_expressions = [
            (pl.col("candidate_rank") <= cutoff).sum().alias(f"at_{cutoff}") for cutoff in cutoffs
        ]
        source_expressions = [
            (pl.col(f"source_{source}_present") > 0).sum().alias(f"source_{source}")
            for source in sources
        ]
        marginal_expressions = []
        for source in sources:
            other_columns = [
                pl.col(f"source_{other}_present") for other in sources if other != source
            ]
            other_evidence = pl.sum_horizontal(other_columns) if other_columns else pl.lit(0)
            marginal_expressions.append(
                ((pl.col(f"source_{source}_present") > 0) & (other_evidence == 0))
                .sum()
                .alias(f"marginal_{source}")
            )
        source_columns = [pl.col(f"source_{source}_present") for source in sources]
        union_expression = (
            (pl.sum_horizontal(source_columns) > 0).sum().alias("source_union")
            if source_columns
            else pl.lit(0).alias("source_union")
        )
        summary = positives.select(
            *recall_expressions,
            *source_expressions,
            *marginal_expressions,
            union_expression,
        ).collect(engine="streaming")
        histogram_frame = (
            positives.group_by("candidate_rank")
            .len(name="count")
            .sort("candidate_rank")
            .collect(engine="streaming")
        )
        session_lengths = (
            frame.select("session", "session_length")
            .unique(subset=["session"])
            .collect(engine="streaming")
        )
        bucket_by_session = {
            int(session): _length_bucket(int(length))
            for session, length in session_lengths.iter_rows()
        }
        label_frame = pl.read_parquet(labels).filter(pl.col("type") == target.value)
        bucket_denominators: dict[str, int] = {}
        label_counts = label_frame.group_by("session").agg(pl.col("aid").n_unique())
        for session, aids in label_counts.iter_rows():
            bucket = bucket_by_session.get(int(session))
            if bucket is not None:
                bucket_denominators[bucket] = bucket_denominators.get(bucket, 0) + min(
                    int(aids), 20
                )
        bucket_hits: dict[str, int] = {}
        for session in (
            positives.filter(pl.col("candidate_rank") <= 20)
            .select("session")
            .collect(engine="streaming")["session"]
        ):
            bucket = bucket_by_session.get(int(session))
            if bucket is not None:
                bucket_hits[bucket] = bucket_hits.get(bucket, 0) + 1
        reports[target.value] = TargetCandidateRecall(
            denominator=denominator,
            recall_at={
                cutoff: int(summary[f"at_{cutoff}"][0]) / denominator if denominator else 0.0
                for cutoff in cutoffs
            },
            source_recall={
                source: int(summary[f"source_{source}"][0]) / denominator if denominator else 0.0
                for source in sources
            },
            source_marginal_recall={
                source: int(summary[f"marginal_{source}"][0]) / denominator if denominator else 0.0
                for source in sources
            },
            source_union_recall=(
                int(summary["source_union"][0]) / denominator if denominator else 0.0
            ),
            session_length_recall_at_20={
                bucket: bucket_hits.get(bucket, 0) / bucket_denominator
                for bucket, bucket_denominator in sorted(bucket_denominators.items())
            },
            positive_rank_histogram={
                int(rank): int(count) for rank, count in histogram_frame.iter_rows()
            },
        )

    weighted = sum(weights[target] * reports[target].recall_at.get(20, 0.0) for target in reports)
    weighted_at_100 = sum(
        weights[target] * reports[target].recall_at.get(100, 0.0) for target in reports
    )
    return CandidateRecallReport(reports, weighted, weighted_at_100)
