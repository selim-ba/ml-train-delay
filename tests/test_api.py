"""Tests for the HTTP API with a fake model and a small metrics file (no data needed).
Skipped when the serving dependencies are not installed (``uv sync --group serve``)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from swissdelay.serve.api import create_app  # noqa: E402

COLUMNS = ["d0_min", "horizon_min", "category", "net_cur_n"]


class FakeChampion:
    """Δd = −0.1 × d0 at every horizon, interval ± 1 min."""

    manifest = {"feature_set": "full_network_plus", "created": "2026-10-05T19:42:00+00:00",
                "trained_on": ["2025-08-01", "2026-04-30"], "horizons": [15, 30, 60]}  # fmt: skip
    calibration = {"strategy": "rolling_tails", "window": ["2026-09-17", "2026-09-30"]}
    columns = COLUMNS

    def predict(self, df, horizon):
        p50 = -0.1 * df["d0_min"].astype(float).to_numpy()
        return pd.DataFrame({"p50": p50, "lo": p50 - 1, "hi": p50 + 1}, index=df.index)


@pytest.fixture()
def client(tmp_path):
    days = pd.date_range("2026-07-01", periods=5).strftime("%Y-%m-%d")
    metrics = pd.DataFrame([
        {"operating_day": d, "horizon_min": h, "n": 100, "is_disruption_day": i == 2,
         "mae": 0.7, "mae_historical": 0.9, "coverage_static": 0.78, "coverage_rolling": 0.8,
         "coverage_rolling_tails": 0.8, "above_static": 0.12, "above_rolling": 0.11,
         "above_rolling_tails": 0.1, "psi_d0_min": np.nan if i == 0 else 0.01, "alerts": ""}
        for i, d in enumerate(days) for h in (15, 30, 60)
    ])  # fmt: skip
    path = tmp_path / "daily_metrics.parquet"
    metrics.to_parquet(path, index=False)
    return TestClient(create_app(champion=FakeChampion(), metrics_path=path))


def point(h, d0=5.0, **extra):
    return {
        "horizon_min": h,
        "features": {"d0_min": d0, "category": "IC", "net_cur_n": 12, **extra},
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["horizons_min"] == [15, 30, 60]
    assert body["interval_calibration"]["strategy"] == "rolling_tails"
    assert body["features"] == COLUMNS


def test_predict(client):
    r = client.post("/predict", json={"points": [point(15), point(60, d0=10.0)]})
    assert r.status_code == 200
    p15, p60 = r.json()
    assert p15["horizon_min"] == 15 and p15["delta_p50"] == pytest.approx(-0.5)
    assert p15["delta_p10"] == pytest.approx(-1.5) and p15["delta_p90"] == pytest.approx(0.5)
    assert p15["delay_at_target_p50"] == pytest.approx(4.5)
    assert p60["delay_now_min"] == 10.0 and p60["delay_at_target_p50"] == pytest.approx(9.0)


def test_predict_errors(client):
    assert client.post("/predict", json={"points": [point(45)]}).status_code == 422
    r = client.post("/predict", json={"points": [{"horizon_min": 15, "features": {"d0_min": 1}}]})
    assert r.status_code == 422 and "category" in r.json()["detail"]
    r = client.post("/predict", json={"points": [point(15, d0=None)]})
    assert r.status_code == 422
    assert client.post("/predict", json={"points": []}).status_code == 422


def test_metrics(client):
    rows = client.get("/metrics/daily", params={"horizon": 30, "start": "2026-07-02"}).json()
    assert len(rows) == 4 and {r["horizon_min"] for r in rows} == {30}
    assert rows[0]["operating_day"] == "2026-07-02"
    first = client.get("/metrics/daily", params={"horizon": 15, "end": "2026-07-01"}).json()
    assert first[0]["psi_d0_min"] is None  # NaN → null
    summary = client.get("/metrics/summary").json()
    assert len(summary) == 6  # 3 horizons × normal / disruption
    assert summary[0]["skill_vs_historical"] == pytest.approx(1 - 0.7 / 0.9)
