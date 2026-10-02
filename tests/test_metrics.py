import numpy as np
import pandas as pd
import pytest

from swissdelay.evaluation.days import flag_disruption_days
from swissdelay.evaluation.metrics import evaluate, mae_ci, mae_diff_ci


def _toy(n_days=30, per_day=50, seed=1):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    y = rng.normal(-0.8, 1.5, size=day.size)
    return day, y


def test_mae_ci_contains_point_estimate():
    day, y = _toy()
    err = np.abs(y)
    mae, lo, hi = mae_ci(day, err, n_boot=500)
    assert mae == pytest.approx(err.mean())
    assert lo < mae < hi


def test_diff_ci_identical_models_is_zero():
    day, y = _toy()
    d, lo, hi = mae_diff_ci(day, np.abs(y), np.abs(y), n_boot=200)
    assert d == 0 and lo == 0 and hi == 0


def test_diff_ci_detects_a_better_model():
    day, y = _toy()
    persistence = np.abs(y - 0.0)
    offset = np.abs(y - (-0.8))
    d, lo, hi = mae_diff_ci(day, offset, persistence, n_boot=500)
    assert d < 0 and hi < 0  # offset is significantly better


def test_evaluate_shape_and_reference():
    day, y = _toy(n_days=10, per_day=20)
    df = pd.DataFrame(
        {"operating_day": day, "horizon_min": 15, "delta_min": y, "pred_a": 0.0, "pred_b": -0.8}
    )
    res = evaluate(df, ["pred_a", "pred_b"], reference="pred_a", n_boot=100)
    assert list(res["model"]) == ["a", "b"]
    assert res.loc[0, "diff_vs_ref"] == 0
    assert (res["days"] == 10).all() and (res["n"] == 200).all()


def test_disruption_days_are_top_share():
    stats = pd.DataFrame(
        {
            "mean_delay_min": np.r_[np.ones(38), 5.0, 6.0],
            "cancelled_runs_share": np.r_[np.full(38, 0.01), 0.10, 0.20],
        }
    )
    out = flag_disruption_days(stats, share=0.05)  # ceil(0.05 * 40) = 2 days
    assert out["is_disruption_day"].sum() == 2
    assert out["is_disruption_day"].iloc[-2:].all()
