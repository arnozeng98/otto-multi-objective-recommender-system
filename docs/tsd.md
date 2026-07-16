# Technical Solution Design and Specification

## 1. Purpose

This document defines the architecture and implementation contract for a resource-aware solution to the Kaggle **OTTO – Multi-Objective Recommender System** competition. The system predicts the next clicked item and subsequent cart and order items from a truncated anonymous shopping session.

The design combines the strongest reproducible ideas from the 2023 winning solutions with optional sequential and generative retrieval experiments. It is deliberately bounded for one RTX 5070 with 12 GB VRAM, 32 GB host RAM, WSL2, and NVMe storage.

## 2. Goals and Non-Goals

### Goals

- Reproduce a strong candidate-rerank baseline with temporal validation.
- Process the 11 GB training JSONL without loading it into memory.
- Generate target-aware candidates from history, popularity, and configurable co-visitation matrices.
- Rank candidates independently for clicks, carts, and orders.
- Evaluate the exact competition Recall@20 and produce schema-valid submissions.
- Compare SASRec, Mamba, HSTU-inspired, and TIGER-style retrieval through one interface.
- Record accuracy, candidate coverage, runtime, and memory for every experiment.

### Non-Goals

- Reproducing the first-place 256 GB training setup exactly.
- Claiming that `HSTUStyleRetriever` is Meta's production HSTU implementation.
- Guaranteeing a leaderboard score without running a complete local validation and Kaggle submission.
- Using language models without item text, category, brand, or image metadata.

## 3. Metric

For target type $t$, the corpus-level metric is:

$$
R_t = \frac{\sum_s |P_{s,t}^{20} \cap G_{s,t}|}{\sum_s \min(20, |G_{s,t}|)}
$$

The final score is:

$$
0.10R_{clicks} + 0.30R_{carts} + 0.60R_{orders}
$$

Click ground truth contains only the first future click. Cart and order ground truth contain all unique future item IDs. Predictions are deduplicated before truncation to 20.

## 4. Constraints

| Resource | Budget | Design response |
|---|---:|---|
| GPU | RTX 5070, 12 GB | BF16, sampled objectives, compact embeddings, optional models |
| Host memory | 32 GB | Streaming JSONL, partitioned Parquet, bounded candidates |
| Storage | NVMe, over 250 GB free | Persist reusable matrices, features, indexes, and manifests |
| Runtime | Several hours per primary run | Five high-value matrices by default; research branches run separately |
| Platform | WSL2/Linux | CUDA-enabled PyTorch and XGBoost; portable CPU smoke path |

Python 3.12 is managed per project with `uv`. The host Python 3.14 installation remains untouched because compiled CUDA extensions often lag the newest interpreter.

## 5. Architecture

The editable architecture is in [`assets/architecture.drawio`](assets/architecture.drawio). Training and inference interactions are in [`assets/sequence.drawio`](assets/sequence.drawio).

The primary flow is:

1. Flatten JSONL sessions into compact event Parquet.
2. Persist earlier ranker-training and later local-validation temporal windows.
3. Build target-aware history, popularity, and co-visitation candidates.
4. Optionally add ANN or semantic-ID candidates.
5. Merge and deduplicate candidates while retaining source evidence.
6. Build leakage-safe session, item, and session-item features.
7. Train one LambdaMART model per target.
8. Evaluate Recall@20 or write a Kaggle submission.

CPU and NVMe own ETL, matrix reduction, feature materialization, and artifact persistence. The GPU owns neural training, embedding inference, and supported XGBoost training.

## 6. Package Boundaries

| Package | Responsibility |
|---|---|
| `data` | Typed events, streaming conversion, temporal split |
| `metrics` | Exact weighted Recall@20 |
| `covisitation` | Configurable item-pair aggregation and pruning |
| `candidates` | Candidate source adapters and deterministic fusion |
| `features` | Context-only session-item features |
| `ranking` | Query-group validation and XGBoost LambdaMART |
| `neural` | SASRec, Mamba adapter, HSTU-style model, ANN, Semantic IDs |
| `submission` | Three-row-per-session output contract |
| `artifacts` | Fingerprints, manifests, and stale-input detection primitives |
| `cli` | Stable user-facing command boundary |

## 7. Data Contracts

### Event

| Field | Type | Constraint |
|---|---|---|
| `session` | int64 | Non-negative anonymous session ID |
| `aid` | int32 | Product identifier |
| `ts` | int64 | Unix timestamp in milliseconds |
| `type` | string | `clicks`, `carts`, or `orders` |

Events are ordered by timestamp before pair generation or temporal splitting. Generated Parquet uses Zstandard compression and compact numeric types.

### Candidate

Every candidate has an `aid`, fused score, best source rank, sorted source list, and per-source score/rank evidence. Production feature tables additionally carry `(session, target)`, a deterministic fused rank, and the binary target label. Candidate fusion is deterministic and applies the total budget after deduplication.

### Submission

The CSV header is `session_type,labels`. Each test session has exactly three rows, labels are unique space-separated integer IDs, and each row contains at most 20 labels. Missing candidates are backfilled from target-specific popularity.

## 8. Temporal Validation and Leakage Rules

- An earlier cutoff creates ranker-training queries; its label window ends at the final
	validation cutoff.
- The final cutoff creates local-validation queries and labels.
- Popularity, matrices, embeddings, and features use context/training events only.
- Validation labels never enter pair counts, hard-negative pools, or item popularity.
- The next click is the first click after the cutoff.
- Repeated cart and order labels are deduplicated.
- Artifact manifests include source fingerprints and configuration hashes.

## 9. Candidate Generation

The default single-GPU profile uses five configurable matrix families:

1. Adjacent forward click transitions.
2. All-event pairs with exponential time decay.
3. Forward click/cart/order to cart/order pairs.
4. Cart/order to cart/order pairs.
5. Recent-window trend pairs.

The generic rule supports source and target event filters, direction, sequence distance, time window, half-life, and neighbor top-K. Session history and target-specific popularity provide high-precision revisits and cold-start fallback.

The production builder reads flattened Parquet in batches while carrying the final session
across each batch boundary. Weighted pairs are buffered into source-aid hash partitions. Each
partition is then aggregated independently by `(source_aid, target_aid)`, deterministically
sorted by score and target aid, and pruned to top-K. Temporary fragments are removed before the
completed directory is atomically promoted. The manifest records the rule hash, input file
metadata, session count, pair count, and output partition count.

Each matrix partition is converted to memory-mapped source IDs, offsets, targets, scores, and
ranks. Runtime lookup loads only the partition addressed by `source_aid % partitions` and uses
binary search within its source index. The five stores are published as one atomic suite.

Neural sources emit normalized session queries and item vectors. ANN results enter the same candidate merger, so candidate Recall@K can prove whether a new model adds coverage before reranker training.

## 10. Features and Ranking

Initial features include candidate score/rank/source agreement, all seven source-specific
presence/score/rank triples, session length and uniqueness, item frequency and recency in the
session, and per-action counts.

Three target-specific XGBoost models use `rank:ndcg`, one query group per session, histogram
trees, deterministic seeds, validation early stopping, and positive-preserving candidate
limits. Training keeps every positive plus the first 80 fused candidates; validation keeps the
first 120 by default while the metric denominator continues to use all original labels. Group
sizes must sum exactly to the row count. The bounded workstation profile deterministically
selects the first 50,000 eligible query sessions per target for training and evaluation; the
ranker manifest records these limits. GPU training is requested explicitly and must not silently
fall back.

## 11. Modern Retrieval Experiments

### SASRec

The stable neural control uses causal Transformer encoding, tied item embeddings, event and time embeddings, and target-type conditioning.

### Mamba

`MambaRetriever` uses the optional `mamba-ssm` CUDA extension. Installation is Linux-only and capability dependent. An unavailable extension produces an actionable error; it does not silently become SASRec.

### HSTU-Style

The local model adopts target-aware temporal transduction and gated attention at a small scale. It uses standard PyTorch attention rather than Meta's H100-oriented kernels and is named accordingly.

### TIGER-Style Semantic IDs

Collaborative item vectors are encoded with multi-level residual quantization. A compact sequence decoder can generate semantic-code beams that resolve to real items. Because OTTO lacks content metadata, collision, coverage, and incremental candidate recall are mandatory acceptance metrics.

## 12. Configuration and CLI

Configuration is validated with Pydantic. `configs/base.yaml` owns shared defaults; profiles override only resource- or experiment-specific values.

Stable commands include:

```bash
otto-recsys diagnostics
otto-recsys check-config configs/single_gpu.yaml
otto-recsys preprocess data/train.jsonl artifacts/events/train.parquet
otto-recsys build-covisitation artifacts/events/train.parquet artifacts/covisitation/time_decay --rule time_decay
otto-recsys prepare-validation artifacts/events/train.parquet artifacts/validation/views --config configs/single_gpu.yaml
otto-recsys build-covisitation-suite artifacts/validation/views/ranker_train/matrix_events.parquet artifacts/validation/matrices/ranker_train
otto-recsys build-matrix-store artifacts/validation/matrices/ranker_train artifacts/validation/stores/ranker_train
otto-recsys run-validation data/train.jsonl artifacts/single_gpu/full-validation --config configs/single_gpu.yaml
otto-recsys run artifacts/single_gpu/full-run --config configs/single_gpu.yaml
otto-recsys smoke artifacts/smoke
```

Long stages write immutable outputs plus a manifest. A stage may reuse output only when its
configuration hash and all upstream fingerprints match. The `run-validation` command owns local
temporal evaluation. The primary `run DESTINATION` command defaults to the repository train,
test, sample-submission, and single-GPU profile paths, imports compatible validation artifacts,
and continues through full-history retrieval, test inference, and an atomically validated
`DESTINATION/submission/submission.csv`. Rich reports step numbering, work units, elapsed time,
and ETA; `--no-progress` keeps non-interactive output stable.

The `validated` model strategy reuses the evaluated temporal rankers. The optional `refit`
strategy trains separately identified final models from complete query groups sampled across
both temporal windows. Refit models never consume test events and do not inherit validation
metrics that were measured on different model artifacts.

Candidate shards are committed in synchronized three-target session chunks. An atomic checkpoint
advances only after all clicks, carts, and orders shards are durable. On restart, invalid tail
files are removed, all targets roll back to their greatest common complete session, and processing
continues without duplicate or missing query groups. Matrix and store suites similarly retain
completed child manifests across interruptions.

## 13. Failure Handling

- Invalid input event types fail during conversion.
- Empty session context is excluded from validation queries.
- Invalid candidate budgets and ranking groups fail before training.
- Missing optional dependencies report installation guidance.
- CUDA out-of-memory is handled by reducing profile batch size or enabling accumulation, never by changing model semantics silently.
- Partial long-running output is written to a temporary path and promoted only after validation.
- Power loss preserves the last atomic candidate checkpoint and completed matrix/store children.
- Test inference writes atomic per-target prediction shards and resumes completed shards.
- Submission output is written to a temporary CSV and promoted only after sample-order, coverage,
	row-count, uniqueness, and label-count checks pass.

## 14. Experiment Methodology

Every experiment records configuration, git revision, input fingerprint, seed, candidate Recall@K by source, target Recall@20, weighted score, runtime, and peak host/GPU memory. The minimum ablation sequence is:

1. History + popularity.
2. Add default co-visitation matrices.
3. Add XGBoost reranking.
4. Add SASRec ANN candidates.
5. Replace or augment with Mamba.
6. Add HSTU-style candidates.
7. Add TIGER-style semantic candidates.

A research branch is retained in final fusion only when it improves repeated local validation or provides justified complementary recall at an acceptable resource cost.

## 15. Testing and Acceptance

- Unit tests cover metrics, split leakage, conversion, pair direction/decay, fusion, features, quantization, ANN, and submission schema.
- CPU integration executes the complete synthetic classical path.
- Optional Torch tests verify output shape and vector normalization.
- WSL2 acceptance requires one BF16 CUDA step and a bounded real-data slice.
- Draw.io files must parse as XML and remain editable in diagrams.net.
- Ruff, formatting, mypy, and pytest must pass before release.

## 16. Security and Data Handling

The dataset contains anonymous session and product IDs rather than account identity. Raw competition data and generated artifacts are ignored by Git. No command uploads data, models, or metrics. Kaggle terms and the dataset license remain the user's responsibility.

## 17. Known Limitations

- The measured 1,000-session weighted Recall@20 of 0.123803 is an engineering acceptance
	result, not a leaderboard estimate.
- Full-history matrix and test-candidate runtime remains CPU/NVMe-bound; GPU utilization is
	expected primarily during XGBoost training/inference and optional neural workloads.
- CUDA extension compatibility depends on Linux, installed PyTorch/CUDA versions, and Blackwell support.
- Mamba is unavailable in the verified PyTorch 2.13/cu130 environment because neither a
	compatible wheel nor a compatible local extension compiler/header combination is available.
- Semantic IDs are learned only from interactions and may add little beyond co-visitation on this dataset.