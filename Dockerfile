# SwissDelay serving image: API (default) and dashboard.
# The model (models/champion) and the replay outputs are mounted at run time, not baked in.
FROM python:3.12-slim

# XGBoost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
# dependencies first (cached), then the package
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group serve --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --group serve

ENV PATH="/app/.venv/bin:$PATH" \
    SWISSDELAY_MODEL_DIR=/app/models/champion \
    SWISSDELAY_METRICS=/app/data/processed/replay/daily_metrics.parquet \
    SWISSDELAY_BENCHMARK=/app/reports/tabular_test.csv
EXPOSE 8000 8501
CMD ["uvicorn", "swissdelay.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
