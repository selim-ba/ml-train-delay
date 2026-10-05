"""Tests for the replay: per-tail offsets, rolling calibration (past days only), daily
metrics, drift and alerts. A fake champion replaces the XGBoost models."""

import numpy as np
import pandas as pd
import pytest

from swissdelay.models import quantile as qt
from swissdelay.models import registry
from swissdelay.pipeline import replay as rp

H = 15


class FakeChampion:
    """P50 = 0 and a fixed raw interval [−1, 1]; static offsets 0."""

    def __init__(self, reference: pd.Series):
        self.manifest = {
            "offsets": {str(H): dict.fromkeys(qt.D0_BUCKETS[1], 0.0)},
            "drift_reference": {str(H): {"d0_min": registry.drift_reference(reference)}},
        }

    def predict(self, df, horizon):
        n = len(df)
        return pd.DataFrame({"p50": np.zeros(n), "q10": -np.ones(n), "q90": np.ones(n)},
                            index=df.index)  # fmt: skip


def day(i: int, rng, n: int = 2000, sd: float = 1.5, d0_shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "trip_key": [f"d{i}t{j}" for j in range(n)], "stop_seq": 1, "horizon_min": H,
        "operating_day": f"2026-07-{i + 1:02d}", "d0_min": rng.exponential(1.0, n) + d0_shift,
        "delta_min": rng.normal(0, sd, n), "pred_historical": 0.0, "is_disruption_day": False,
    })  # fmt: skip


def test_tail_offsets():
    y = np.arange(1000, dtype=float)
    lo, hi = rp.tail_offsets(np.full(1000, 500.0), np.full(1000, 500.0), y)
    below, above = np.mean(y < 500 - lo), np.mean(y > 500 + hi)
    assert 0.08 <= below <= 0.10 and 0.08 <= above <= 0.10


def test_psi():
    rng = np.random.default_rng(0)
    ref = registry.drift_reference(pd.Series(rng.normal(size=50_000)))
    assert registry.psi(ref, pd.Series(rng.normal(size=20_000))) < 0.02
    assert registry.psi(ref, pd.Series(rng.normal(1.0, 1.0, size=20_000))) > 0.25


def test_rolling_calibration_uses_past_days_only():
    rng = np.random.default_rng(1)
    champ = FakeChampion(pd.Series(rng.exponential(1.0, 50_000)))
    cal = rp.RollingCalibrator(window=14, min_days=7)
    rows = []
    for i in range(10):
        d = day(i, rng)
        assert len(cal.days) == min(i, 14)  # the day being scored is never in the window
        m, frame = rp.score_horizon(champ, d, H, cal, champ.manifest["drift_reference"])
        rows.append(m)
        cal.add(d["operating_day"].iloc[0], frame)
    first, last = rows[0], rows[-1]
    # before 7 days of history the rolling strategies equal the static one
    assert first["coverage_rolling"] == first["coverage_static"]
    # static [−1, 1] covers ~50 % of N(0, 1.5); the rolling windows recover ~80 %
    assert last["coverage_static"] < 0.6
    assert 0.76 <= last["coverage_rolling"] <= 0.84
    assert 0.76 <= last["coverage_rolling_tails"] <= 0.84
    assert last["below_rolling_tails"] == pytest.approx(0.10, abs=0.02)
    assert last["above_rolling_tails"] == pytest.approx(0.10, abs=0.02)


def test_window_keeps_last_days():
    cal = rp.RollingCalibrator(window=3, min_days=1)
    f = pd.DataFrame({"horizon_min": [H], "d0_bucket": ["on time < 1"], "q10": [0.0],
                      "q90": [1.0], "y": [0.5]})  # fmt: skip
    for i in range(5):
        cal.add(f"2026-07-0{i + 1}", f)
    assert cal.last_days() == ["2026-07-03", "2026-07-04", "2026-07-05"]


def test_alerts():
    rng = np.random.default_rng(2)
    champ = FakeChampion(pd.Series(rng.exponential(1.0, 50_000)))
    cal = rp.RollingCalibrator()
    m, _ = rp.score_horizon(
        champ, day(0, rng, d0_shift=3.0), H, cal, champ.manifest["drift_reference"]
    )
    assert "drift:d0_min" in rp.alerts(m, pd.DataFrame())
    history = pd.DataFrame({"horizon_min": H, "mae": [0.5] * 10})
    assert "mae_jump" in rp.alerts({**m, "mae": 0.9, "psi_d0_min": 0.0}, history)
    assert rp.alerts({**m, "mae": 0.5, "psi_d0_min": 0.0, "coverage_rolling_tails": 0.8},
                     history) == ""  # fmt: skip
