PY ?= .venv/bin/python

.PHONY: help venv install format lint typecheck test test-unit test-integration check smoke demo clean

help:
	@echo "make install     Create .venv and install the project with dev extras"
	@echo "make format      Format with ruff"
	@echo "make lint        Lint with ruff"
	@echo "make typecheck   Type check with mypy"
	@echo "make test        Run the whole test suite (offline)"
	@echo "make check       format + lint + typecheck + test"
	@echo "make smoke       Run an offline end-to-end agent loop with the mock provider"
	@echo "make demo        Just the offline agent-loop demo"

venv:
	uv venv --python 3.11 .venv || python3 -m venv .venv

install: venv
	uv pip install --python $(PY) -e ".[dev,gemini]" || $(PY) -m pip install -e ".[dev,gemini]"

format:
	$(PY) -m ruff format src tests

lint:
	$(PY) -m ruff check --fix src tests

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest tests/unit

test-integration:
	$(PY) -m pytest tests/integration

check: format lint typecheck test

smoke:
	$(PY) -m agent --help
	$(PY) -m agent -p mock config show
	$(PY) -m agent -p mock doctor
	$(PY) scripts/demo_offline.py

demo:
	$(PY) scripts/demo_offline.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
