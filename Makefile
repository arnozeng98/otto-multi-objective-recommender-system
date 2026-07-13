PYTHON := python

.PHONY: install check test smoke diagnostics

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