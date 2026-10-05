"""Tests for the quantile models: metrics, sorting, and calibrated intervals on synthetic
data with a spread that grows with the current delay."""

import numpy as np
import pandas as pd
import pytest

from swissdelay.features.build import ALL_FEATURES, CATEGORICAL_FEATURES
from swissdelay.models import quantile as qt
from swissdelay.models import tabular as tb


def test_pinball_and_sorting():
    y = np.array([0.0, 1.0, 2.0])
    assert qt.pinball(y, np.ones(3), 0.9) == pytest.approx((0.1 + 0 + 0.9) / 3)
    assert qt.pinball(y, np.ones(3), 0.1) == pytest.approx((0.9 + 0 + 0.1) / 3)
    lo, mid, hi = qt.sort_quantiles(
        np.array([3.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])
    )
    assert lo.tolist() == [1, 0] and mid.tolist() == [2, 1] and hi.tolist() == [3, 2]


def test_interval_metrics():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = qt.interval_metrics(y, np.full(4, 0.5), np.full(4, 2.5), np.full(4, 0.5), np.full(4, 2.5))
    assert m["coverage"] == 0.5 and m["below_p10"] == 0.25 and m["above_p90"] == 0.25
    assert m["mean_width"] == 2.0 and m["crossing_share"] == 0.0


def test_conformal_offset():
    y = np.arange(100, dtype=float)
    # interval [10, 89] covers 80 of 100 points: no correction needed
    assert qt.conformal_offset(np.full(100, 10.0), np.full(100, 89.0), y) == pytest.approx(0, abs=1)
    # a too-narrow interval [40, 59] needs widening by about 30
    q = qt.conformal_offset(np.full(100, 40.0), np.full(100, 59.0), y)
    lo, hi = 40 - q, 59 + q
    assert np.mean((y >= lo) & (y <= hi)) >= 0.8 and q > 25


def synthetic(path, n_days=270, per_day=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-08-01", periods=n_days, freq="D").append(
        pd.date_range("2026-04-01", periods=61, freq="D")
    )
    n = len(days) * per_day
    df = pd.DataFrame({"operating_day": np.repeat(days.date, per_day)})
    for c in ALL_FEATURES:
        df[c] = rng.choice(["a", "b"], size=n) if c in CATEGORICAL_FEATURES else rng.normal(size=n)
    df["horizon_min"] = 15
    df["enters_from_abroad"], df["is_extra_trip"] = False, False
    df["d0_min"] = rng.exponential(2.0, size=n)
    # the spread of Δd grows with the current delay
    df["delta_min"] = -0.3 * df["d0_min"] + rng.normal(0, 0.2 + 0.3 * df["d0_min"], size=n)
    df["pred_historical"] = -0.8
    df["trip_key"], df["stop_seq"] = [f"t{i}" for i in range(n)], 1
    df["split"] = np.where(pd.to_datetime(df["operating_day"]) < "2026-05-01", "train", "valid")
    df["in_common_subset"], df["is_disruption_day"] = True, False
    df["line_name"], df["station_id"], df["target_seq"] = "IC1", 1, 2
    df.to_parquet(path, index=False)


def test_intervals_are_calibrated(tmp_path):
    path = tmp_path / "features_x.parquet"
    synthetic(path)
    params = tb.XGB_PARAMS | {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4,
                              "min_child_weight": 5, "early_stopping_rounds": 20}  # fmt: skip
    preds, info, offsets = qt.run("valid", feature_set="full", horizons=(15,), source=f"'{path}'",
                         params=params, save_models=False)  # fmt: skip
    assert set(info["quantile"]) == {0.1, 0.9} and (info["refit_trees"] > 0).all()
    assert set(offsets["horizon_min"]) == {15} and len(offsets) == 4
    # buckets with enough calibration points get their own offset; rare ones the pooled one
    own = offsets[~offsets["pooled"]]
    assert (own["n_calibration"] >= qt.MIN_CAL_POINTS).all() and own["offset_min"].abs().max() < 1
    assert offsets.loc[offsets["d0_bucket"] == "major > 10", "pooled"].item()
    for sfx in ("", "_cal"):
        lo, hi = qt.sort_quantiles(preds[f"pred_q10{sfx}"], preds[f"pred_q90{sfx}"])
        preds[f"lo{sfx}"], preds[f"hi{sfx}"] = lo, hi
    rep = qt.report(preds)
    assert set(rep["interval"]) == {"raw", "calibrated"}
    rep = rep[rep["interval"] == "raw"]
    overall = rep[rep["d0_bucket"] == "all"].iloc[0]
    assert 0.72 <= overall["coverage"] <= 0.88
    assert overall["crossing_share"] < 0.02
    # wider intervals for late trains
    width = rep.set_index("d0_bucket")["mean_width"]
    cal = qt.report(preds).query("interval == 'calibrated' and d0_bucket == 'all'").iloc[0]
    assert 0.75 <= cal["coverage"] <= 0.85
    assert width["moderate 3-10"] > 1.5 * width["on time < 1"]
