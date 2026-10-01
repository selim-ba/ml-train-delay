.PHONY: install install-all lint format test reproduce clean

install:        ## core + dev tools
	uv sync

install-all:    ## + deep learning and serving stacks
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

reproduce:      ## rebuild baselines and report (filled in as the pipeline grows)
	@echo "Not implemented yet — see docs/roadmap.md"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__