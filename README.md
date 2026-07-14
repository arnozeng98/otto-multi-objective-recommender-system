# OTTO Multi-Objective Recommender System

A reproducible, resource-aware solution for the Kaggle [OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system) competition. It combines strong co-visitation candidate generation and target-specific gradient-boosted ranking with optional modern sequential and generative retrieval experiments.

The project is designed for **WSL2/Linux, one NVIDIA RTX 5070 with 12 GB VRAM, 32 GB host RAM, and NVMe storage**. It does not require the 256 GB machines used by several top competition solutions.

> The original competition ended in 2023. This repository is intended for reproducible research, late submissions, and recommender-system engineering practice. It does not claim an unmeasured leaderboard score.

## Design

The practical default pipeline is:

1. Stream session JSONL into compact Parquet.
2. Create a leakage-safe global temporal validation split.
3. Generate candidates from session history, target popularity, and five configurable co-visitation families.
4. Optionally add SASRec, Mamba, HSTU-style, or semantic-ID candidates.
5. Deduplicate and retain candidate-source evidence.
6. Build context-only session, item, and session-item features.
7. Train independent XGBoost rankers for clicks, carts, and orders.
8. Evaluate weighted Recall@20 or generate a validated Kaggle submission.

The classical local-validation path is now executable end to end through one resumable command.
Each completed run is keyed by its source metadata, full validated configuration, and optional
session limit.

Editable diagrams:

- [System architecture and artifact flow](docs/assets/architecture.drawio)
- [Training and inference sequences](docs/assets/sequence.drawio)
- [Technical solution design and specification](docs/tsd.md)

## Metric

The competition uses corpus-level Recall@20:

$$
Score = 0.10R_{clicks} + 0.30R_{carts} + 0.60R_{orders}
$$

Orders therefore receive the largest optimization weight. The implementation deduplicates predictions, uses only the first future click, and follows the original denominator exactly.

## Repository Layout

```text
configs/                 Validated baseline, smoke, and research profiles
docs/                    Design specification and editable draw.io diagrams
src/otto_recsys/
	candidates/            History, popularity, co-visitation, and source fusion
	covisitation/          Configurable item-pair rules
	data/                  Streaming conversion, schemas, and temporal split
	features/              Leakage-safe candidate features
	metrics/               Exact weighted Recall@20
	neural/                SASRec, Mamba, HSTU-style, ANN, and Semantic IDs
	ranking/               XGBoost LambdaMART wrapper
	artifacts.py           Fingerprints and manifests
	cli.py                 Command-line interface
	smoke.py               Deterministic end-to-end synthetic pipeline
	submission.py          Kaggle output validation
tests/                   Unit and integration smoke tests
```

## Environment

### 1. WSL2 prerequisites

Install a current Ubuntu distribution under WSL2 and verify that the GPU is visible:

```bash
nvidia-smi
```

The Windows NVIDIA driver provides CUDA access to WSL2. Do not install a second Windows driver inside WSL.

### 2. Install uv and Python 3.12

The host may use Python 3.14; this project intentionally pins a separate Python 3.12 interpreter for CUDA extension compatibility.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --extra dev --extra rank
```

For neural experiments, install a current CUDA-enabled PyTorch build supported by the RTX 5070, then synchronize the neural group:

```bash
uv sync --extra dev --extra rank --extra neural
```

Mamba is optional and Linux/CUDA dependent:

```bash
uv sync --extra mamba
```

Run diagnostics before a GPU experiment:

```bash
uv run otto-recsys diagnostics
```

## Data

Place the Kaggle files in `data/`:

```text
data/train.jsonl
data/test.jsonl
data/sample_submission.csv
```

These large files are ignored by Git. Convert them without loading the full dataset into memory:

```bash
uv run otto-recsys preprocess \
	data/train.jsonl \
	artifacts/events/train.parquet

uv run otto-recsys preprocess \
	data/test.jsonl \
	artifacts/events/test.parquet
```

The converter validates event types, writes Zstandard-compressed Parquet, and reports a SHA-256 source fingerprint.

Use a deterministic prefix for a fast real-data check:

```bash
uv run otto-recsys preprocess \
	data/test.jsonl \
	artifacts/real-smoke/test-1000.parquet \
	--max-sessions 1000
```

Build a bounded-memory co-visitation matrix from the flattened training events:

```bash
uv run otto-recsys build-covisitation \
	artifacts/events/train.parquet \
	artifacts/covisitation/time_decay \
	--rule time_decay \
	--partitions 64
```

Available rules are `adjacent_clicks`, `time_decay`, `all_to_buy`, `buy_to_buy`, and
`recent_trend`. The builder streams sessions across Parquet batch boundaries, writes pair
fragments by source-item hash, reduces one partition at a time, and retains the configured
top-K neighbors per source item. Completed output is promoted atomically:

```text
artifacts/covisitation/time_decay/
	manifest.json
	matrix/
		partition-000.parquet
		...
```

Use `--pair-buffer-size` to trade memory for fragment count. Existing destinations are
protected unless `--overwrite` is supplied. Production runs build all five rules and convert
each partition to lazy memory-mapped lookup arrays:

```bash
uv run otto-recsys build-covisitation-suite \
	artifacts/validation/views/ranker_train/matrix_events.parquet \
	artifacts/validation/matrices/ranker_train

uv run otto-recsys build-matrix-store \
	artifacts/validation/matrices/ranker_train \
	artifacts/validation/stores/ranker_train
```

## Quick Start

Run the deterministic CPU pipeline before touching the real data:

```bash
uv run otto-recsys smoke artifacts/smoke
```

It exercises temporal splitting, co-visitation, candidate fusion, Recall@20, and submission generation. Expected outputs are:

```text
artifacts/smoke/metrics.json
artifacts/smoke/submission.csv
```

Validate a profile:

```bash
uv run otto-recsys check-config configs/single_gpu.yaml
uv run otto-recsys check-config configs/research/sasrec.yaml
```

Run the complete classical validation pipeline from JSONL through three evaluated rankers:

```bash
uv run otto-recsys run \
	data/train.jsonl \
	artifacts/single_gpu/validation \
	--config configs/single_gpu.yaml
```

Use `--max-sessions 1000` for a real-data acceptance slice. Repeating an identical completed
command reuses every artifact. A different source, configuration, or session limit is rejected
unless `--overwrite` is explicit. The terminal displays a Rich progress dashboard for all nine
steps with the current step number, unit counts, elapsed time, and ETA. Use `--no-progress` for
stable CI or redirected JSON output.

Interrupted runs resume automatically. Completed stage manifests are reused, matrix/store suites
reuse completed child rules, and candidate materialization checkpoints synchronized three-target
Parquet shards at complete session boundaries. A damaged tail shard is discarded and recovery
continues from the last session shared by clicks, carts, and orders. Do not pass `--overwrite`
when resuming, because it intentionally discards all prior artifacts.

From Windows, run the CUDA pipeline inside WSL while keeping the repository in the current
workspace:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu \
	--cd /mnt/c/Users/arnoz/Desktop/repos/otto-multi-objective-recommender-system \
	-- /home/arnozeng/.venvs/otto/bin/python -m otto_recsys.cli run \
	data/train.jsonl artifacts/single_gpu/full-validation \
	--config configs/single_gpu.yaml
```

The verified 1,000-session acceptance run used the CPU smoke ranker profile for 20 boosting
rounds and produced:

| Target | Recall@20 |
|---|---:|
| Clicks | 0.240481 |
| Carts | 0.086903 |
| Orders | 0.122807 |
| **Weighted** | **0.123803** |

This small-prefix score validates orchestration and metric semantics; it is not a leaderboard
estimate. The same run completed all nine persisted stages and a second execution reused all
nine without recomputation.

## Configuration Profiles

| Profile | Purpose |
|---|---|
| `configs/smoke.yaml` | Tiny CPU validation |
| `configs/single_gpu.yaml` | Primary bounded full-data experiment |
| `configs/research/sasrec.yaml` | Stable neural retrieval control |
| `configs/research/mamba.yaml` | Selective state-space retrieval |
| `configs/research/hstu_style.yaml` | Small HSTU-inspired ablation |
| `configs/research/tiger.yaml` | Residual semantic-ID experiment |

Modern models are candidate sources, not replacements for the full system. A source should enter final fusion only after improving candidate Recall@K or final Recall@20 in the same temporal split.

## Resource Strategy

- CPU and NVMe handle JSON parsing, Parquet, co-visitation reduction, and feature materialization.
- The GPU handles neural training and supported XGBoost histogram training.
- Neural item embeddings default to 48–64 dimensions.
- Full item softmax is avoided; training uses sampled candidates and hard negatives.
- Candidate fusion is capped at roughly 200–350 unique items per session and target.
- Ranker input keeps all positives plus the first 80 training candidates and first 120
	validation candidates per query by default.
- Research profiles run separately so a baseline iteration remains within several hours.

The in-memory reference builder remains useful for tests and bounded slices. Full-data runs use
the hash-partitioned builder above; its output is tested item-for-item against the reference
implementation.

## Testing and Quality

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Torch tests are skipped explicitly when the optional neural dependency is absent. Mamba must report an installation/capability error if its CUDA extension is unavailable; it never silently falls back to another architecture.

The verified WSL2 stack uses Python 3.12.13, PyTorch 2.13.0+cu130, XGBoost 3.3.0,
FAISS 1.14.3, and an RTX 5070 with BF16 support. SASRec, HSTU-style, and GPU XGBoost
forward/backward preflights pass. `mamba-ssm` remains an explicit optional skip because no
matching wheel exists and the available source toolchain has incompatible CUDA headers.

## Submission Contract

Every test session produces exactly three rows:

```text
12906577_clicks,135193 129431 119318 ...
12906577_carts,135193 129431 119318 ...
12906577_orders,135193 129431 119318 ...
```

Labels are unique integer IDs, space delimited, and truncated to 20. Target-specific popularity provides deterministic backfill for short candidate lists.

## Research Lineage

The candidate-rerank architecture follows lessons from the OTTO first-, second-, and third-place write-ups: rich item-item candidate generation, target-aware evidence, and tree-based reranking remain the strongest foundation for this interaction-only dataset.

Modern extensions are informed by:

- Mamba4Rec: efficient selective state-space models for sequential recommendation.
- Meta Generative Recommenders/HSTU: target-aware temporal sequence transduction and sampled objectives.
- TIGER: residual semantic IDs and generative retrieval.
- FAISS: efficient vector similarity search.

See the [TSD](docs/tsd.md) for explicit differences between faithful components and resource-bounded adaptations.

## License

Repository content is provided under the license in [LICENSE](LICENSE). Competition data is governed separately by Kaggle and OTTO terms and is not distributed by this project.