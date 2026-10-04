"""Tests for the tabular models on a small synthetic feature table."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from swissdelay.features.build import ALL_FEATURES, CATEGORICAL_FEATURES, META_COLUMNS
from swissdelay.features.network import ALL_NETWORK_FEATURES, NETWORK_FEATURES
from swissdelay.models import tabular as tb

SMALL_XGB = tb.XGB_PARAMS | {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4,
                             "min_child_weight": 5, "early_stopping_rounds": 20}  # fmt: skip


def test_feature_sets_are_nested_and_clean():
    sets = {name: set(tb.feature_columns(name)) for name in tb.FEATURE_SETS}
    assert (
        sets["current"] < sets["history_1"] < sets["history_3"] < sets["history_5"] < sets["full"]
    )
    assert sets["full_network"] == sets["full"] | set(NETWORK_FEATURES)
    assert sets["full_network_plus"] == sets["full"] | set(ALL_NETWORK_FEATURES)
    assert "pred_historical" not in sets["full_no_prior"]
    forbidden = {"delta_min", "target_seq", "split", "in_common_subset", "is_disruption_day"}
    for cols in sets.values():
        assert cols <= set(ALL_FEATURES) | set(ALL_NETWORK_FEATURES)
        assert not cols & forbidden
        assert not cols & (set(META_COLUMNS) - {"station_id", "line_name"})


def synthetic(path, n_days=270, per_day=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-08-01", periods=n_days, freq="D").append(
        pd.date_range("2026-04-01", periods=61, freq="D")  # tuning month + validation
    )
    n = len(days) * per_day
    df = pd.DataFrame({"operating_day": np.repeat(days.date, per_day)})
    for c in ALL_FEATURES:
        if c in CATEGORICAL_FEATURES:
            df[c] = rng.choice(["a", "b", "c"], size=n)
        else:
            df[c] = rng.normal(size=n)
    df["horizon_min"] = 15
    df["enters_from_abroad"] = rng.random(n) < 0.1
    df["is_extra_trip"] = False
    df["d0_min"] = rng.exponential(2.0, size=n)
    df["slack_to_target_min"] = rng.uniform(0, 4, size=n)
    # Late trains recover up to the available slack (non-linear), plus noise
    df["delta_min"] = -np.minimum(0.5 * df["d0_min"], df["slack_to_target_min"]) + rng.normal(
        0, 0.3, size=n
    )
    df["pred_historical"] = -0.8
    df["trip_key"] = [f"t{i}" for i in range(n)]
    df["stop_seq"] = 1
    df["split"] = np.where(pd.to_datetime(df["operating_day"]) < "2026-05-01", "train", "valid")
    df["in_common_subset"] = True
    df["is_disruption_day"] = False
    df["line_name"], df["station_id"], df["target_seq"] = "IC1", 1, 2
    df.to_parquet(path, index=False)
    return df


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    path = tmp_path_factory.mktemp("tab") / "features_x.parquet"
    synthetic(path)
    preds, info = tb.run(["ridge", "xgb"], ["current", "full"], horizons=(15,),
                         source=f"'{path}'", xgb_params=SMALL_XGB)  # fmt: skip
    return preds, info


def mae(preds, col):
    return float(np.mean(np.abs(preds["delta_min"] - preds[col])))


def test_predictions_cover_validation_only(results):
    preds, info = results
    assert (preds["split"] == "valid").all()
    expected = {"pred_ridge_full", "pred_xgb_full", "pred_xgb_norefit_full",
                "pred_ridge_current", "pred_xgb_current", "pred_xgb_norefit_current"}  # fmt: skip
    assert expected <= set(preds.columns)
    assert len(info) == 4 and (info["tune_mae"] > 0).all()


def test_refit_uses_more_trees(results):
    _, info = results
    xgb_rows = info[info["model"] == "xgb"]
    assert (xgb_rows["refit_trees"] >= xgb_rows["best_iteration"] + 1).all()


def test_models_learn_the_signal(results):
    preds, _ = results
    baseline = mae(preds, "pred_historical")
    assert mae(preds, "pred_ridge_full") < 0.85 * baseline
    assert mae(preds, "pred_xgb_full") < 0.7 * baseline
    # the non-linear interaction (recovery capped by slack) is better captured by trees
    assert mae(preds, "pred_xgb_full") < mae(preds, "pred_ridge_full")
    # without slack the trees lose information
    assert mae(preds, "pred_xgb_full") < mae(preds, "pred_xgb_current")


def test_refit_is_not_worse_on_this_data(results):
    preds, _ = results
    # same signal, one more month of data: the refit should not degrade much
    assert mae(preds, "pred_xgb_full") <= 1.05 * mae(preds, "pred_xgb_norefit_full")


def test_network_features_are_joined(tmp_path, monkeypatch):
    feats = tmp_path / "features_x.parquet"
    df = synthetic(feats, n_days=3, per_day=4)
    net = df[["trip_key", "stop_seq", "horizon_min"]].iloc[:-1].copy()  # last point missing
    for i, c in enumerate(NETWORK_FEATURES):
        net[c] = float(i)
    net.to_parquet(tmp_path / "network_x.parquet", index=False)
    monkeypatch.setattr(tb, "NETWORK_SOURCE", f"'{tmp_path / 'network_x.parquet'}'")
    cols = ["trip_key", *tb.feature_columns("full_network")]
    got = tb.load(duckdb.connect(), "true", cols, source=f"'{feats}'").set_index("trip_key")
    assert len(got) == len(df)
    assert got.loc[df["trip_key"].iloc[0], NETWORK_FEATURES[2]] == 2.0
    assert got.loc[df["trip_key"].iloc[-1], NETWORK_FEATURES].isna().all()
