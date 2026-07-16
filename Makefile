PYTHON := python

.PHONY: install check test smoke diagnostics run

install:
	uv sync --extra dev --extra rank

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test:
	uv run pytest

smoke:
	uv run otto-recsys smoke artifacts/smoke

diagnostics:
	uv run otto-recsys diagnostics

run:
	uv run otto-recsys run artifacts/single_gpu/full-run --config configs/single_gpu.yaml