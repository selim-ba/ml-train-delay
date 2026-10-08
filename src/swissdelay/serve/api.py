"""HTTP API for the deployed model and its production metrics.

Endpoints:

- ``GET /health``: model version, training period, interval calibration in use;
- ``POST /predict``: one or more prediction points (a train leaving a stop now, with the
  features of one horizon) → predicted change in delay (P50) with a calibrated P10–P90
  interval, and the implied delay at the target stop;
- ``GET /metrics/daily``: daily metrics of the production replay (filter by horizon and
  dates);
- ``GET /metrics/summary``: the same, aggregated per horizon and normal / disruption days.

Configuration (environment variables):

- ``SWISSDELAY_MODEL_DIR``: exported model (default ``models/champion``);
- ``SWISSDELAY_METRICS``: daily metrics parquet (default
  ``data/processed/replay/daily_metrics.parquet``).

Usage::

    uv sync --group serve
    uv run uvicorn swissdelay.serve.api:app --reload      # http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("SWISSDELAY_MODEL_DIR", "models/champion"))
METRICS_PATH = Path(
    os.environ.get("SWISSDELAY_METRICS", "data/processed/replay/daily_metrics.parquet")
)

FeatureValue = float | int | str | bool | None


class Point(BaseModel):
    """A train leaving a stop now, described by the model's features for one horizon."""

    horizon_min: int = Field(..., description="Prediction horizon: 15, 30 or 60 minutes")
    features: dict[str, FeatureValue] = Field(
        ..., description="Feature name → value (see GET /health for the list); missing → null"
    )


class PredictRequest(BaseModel):
    points: list[Point] = Field(..., min_length=1, max_length=10_000)


class Prediction(BaseModel):
    horizon_min: int
    delay_now_min: float = Field(..., description="Current departure delay (d0)")
    delta_p50: float = Field(..., description="Predicted change in delay by the target stop")
    delta_p10: float = Field(..., description="Lower end of the 80 % interval")
    delta_p90: float = Field(..., description="Upper end of the 80 % interval")
    delay_at_target_p50: float
    delay_at_target_p10: float
    delay_at_target_p90: float


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows (numpy types → Python, NaN → null)."""
    return json.loads(df.to_json(orient="records"))


def create_app(champion=None, metrics_path: Path = METRICS_PATH,
               model_dir: Path = MODEL_DIR) -> FastAPI:  # fmt: skip
    """Build the app. ``champion`` (anything with ``manifest``, ``columns`` and
    ``predict(df, horizon)``) is loaded lazily from ``model_dir`` when not given."""
    app = FastAPI(title="SwissDelay", version="1.0",
                  description="Change in delay of Swiss IC / IR / RE / EC trains at 15, 30 and "
                              "60 minutes, with calibrated 80 % intervals.")  # fmt: skip
    state = {"champion": champion}

    def model():
        if state["champion"] is None:
            from swissdelay.models import registry  # heavy import, only when serving

            if not (model_dir / "manifest.json").exists():
                raise HTTPException(503, f"No exported model in {model_dir}")
            state["champion"] = registry.load(model_dir)
        return state["champion"]

    def metrics() -> pd.DataFrame:
        if not metrics_path.exists():
            raise HTTPException(503, f"No daily metrics at {metrics_path}")
        m = pd.read_parquet(metrics_path)
        m["operating_day"] = pd.to_datetime(m["operating_day"]).dt.strftime("%Y-%m-%d")
        return m

    @app.get("/health")
    def health() -> dict:
        ch = model()
        man = ch.manifest
        cal = getattr(ch, "calibration", None)
        return {
            "status": "ok",
            "model": f"XGBoost {man['feature_set']}",
            "model_created": man.get("created"),
            "trained_on": man.get("trained_on"),
            "horizons_min": man["horizons"],
            "interval_calibration": (
                {"strategy": cal["strategy"], "window": cal["window"]} if cal
                else {"strategy": "static", "window": ["2026-04-01", "2026-04-30"]}
            ),
            "features": ch.columns,
        }  # fmt: skip

    @app.post("/predict", response_model=list[Prediction])
    def predict(req: PredictRequest) -> list[Prediction]:
        ch = model()
        horizons = set(ch.manifest["horizons"])
        bad = sorted({p.horizon_min for p in req.points} - horizons)
        if bad:
            raise HTTPException(422, f"Unknown horizon(s) {bad}; available: {sorted(horizons)}")
        rows = pd.DataFrame([{**p.features, "horizon_min": p.horizon_min} for p in req.points])
        missing = [c for c in ch.columns if c not in rows.columns]
        if missing:
            raise HTTPException(422, f"Missing features: {missing}")
        if rows["d0_min"].isna().any():
            raise HTTPException(422, "d0_min (current delay) is required for every point")
        out = pd.DataFrame(index=rows.index, columns=["p50", "lo", "hi"], dtype=float)
        for h, g in rows.groupby("horizon_min"):
            pred = ch.predict(g, int(h))
            out.loc[g.index, ["p50", "lo", "hi"]] = pred[["p50", "lo", "hi"]].to_numpy()
        d0 = rows["d0_min"].astype(float).to_numpy()
        p50, lo, hi = (out[c].astype(float).to_numpy() for c in ("p50", "lo", "hi"))
        horizon = rows["horizon_min"].astype(int).to_numpy()
        return [
            Prediction(
                horizon_min=int(horizon[i]), delay_now_min=float(d0[i]),
                delta_p50=float(p50[i]), delta_p10=float(lo[i]), delta_p90=float(hi[i]),
                delay_at_target_p50=float(d0[i] + p50[i]),
                delay_at_target_p10=float(d0[i] + lo[i]),
                delay_at_target_p90=float(d0[i] + hi[i]),
            )
            for i in range(len(rows))
        ]  # fmt: skip

    @app.get("/metrics/daily")
    def metrics_daily(
        horizon: int | None = Query(None, description="15, 30 or 60"),
        start: date | None = None,
        end: date | None = None,
    ) -> list[dict]:
        m = metrics()
        if horizon is not None:
            m = m[m["horizon_min"] == horizon]
        if start is not None:
            m = m[m["operating_day"] >= start.isoformat()]
        if end is not None:
            m = m[m["operating_day"] <= end.isoformat()]
        return _records(m)

    @app.get("/metrics/summary")
    def metrics_summary() -> list[dict]:
        from swissdelay.pipeline.replay import summary

        s = summary(metrics())
        s["skill_vs_historical"] = 1 - s["mae"] / s["mae_historical"]
        return _records(s)

    return app


app = create_app()
