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
	@if uv run python -c 'import torch' 2>/dev/null; then SWISSDELAY_TORCH_TESTS=1 uv run pytest; else echo 'PyTorch not installed: graph-transformer tests skipped'; fi

reproduce:      ## rebuild baselines and report (filled in as the pipeline grows)
	@echo "Not implemented yet — see docs/roadmap.md"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

journeys:       ## rebuild journeys from data/interim
	uv run python -m swissdelay.data.journeys