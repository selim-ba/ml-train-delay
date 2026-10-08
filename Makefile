.PHONY: install install-all lint format test reproduce clean api dashboard docker

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
api:            ## serve the model locally: http://localhost:8000/docs
	uv run uvicorn swissdelay.serve.api:app --reload

dashboard:      ## dashboard on http://localhost:8501 (needs the API running)
	uv run streamlit run src/swissdelay/serve/dashboard.py

docker:         ## API + dashboard in Docker
	docker compose up --build
