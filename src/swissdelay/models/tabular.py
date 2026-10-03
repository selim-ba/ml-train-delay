"""Ridge and XGBoost on the tabular features, one model per horizon, with ablations.

Data use (no peeking at validation):

- **fit:** training months except the last (Aug 2025 – Mar 2026);
- **tune:** the last training month (April 2026), for Ridge's penalty and XGBoost's number
  of trees (early stopping);
- **validation (May 2026):** comparison of models and ablations only;
- **test (June 2026):** only with ``--test``, once the choices are final.

Models:

- **Ridge:** linear, squared loss with L2 penalty. Numeric features are imputed (training
  median, plus a missing-value indicator) and standardised; categorical ones one-hot encoded.
- **XGBoost:** gradient-boosted trees with the absolute-error objective (predicts the
  conditional median, matching the MAE metric). Missing values and categories are handled
  natively. Two steps: early stopping on the tuning month fixes the number of trees, then a
  fresh model is refitted on fitting + tuning months with 10 % more trees (like Ridge, which
  is refitted on both after choosing its penalty). ``--no-refit`` skips step 2; otherwise the
  step-1 predictions are kept as ``xgb_norefit`` for the ablation.

Feature sets (ablations): ``current`` (current delay + timetable + context + historical prior),
then ``history_1`` / ``history_3`` / ``history_5`` (1 / 3 / 5 previous stops), ``full``
(+ timetable slack) and ``full_no_prior``. ``history_5`` vs ``full`` is the slack ablation.

Outputs:

- ``data/processed/tabular_predictions_<split>.parquet``: keys, target, breakdown columns,
  ``pred_historical`` and one ``pred_<model>_<set>`` column per fitted model;
- ``reports/tabular_<split>.csv``: MAE with day-bootstrap CIs vs the historical baseline.

Usage::

    uv run python -m swissdelay.models.tabular                          # ridge + xgb, full set
    uv run python -m swissdelay.models.tabular --sets all --sample 0.3  # ablations, faster
    uv run python -m swissdelay.models.tabular --test                   # final run on test
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from swissdelay import config
from swissdelay.evaluation.metrics import evaluate
from swissdelay.features.build import CATEGORICAL_FEATURES, FEATURE_GROUPS

log = logging.getLogger("swissdelay.tabular")

TUNE_START = "2026-04-01"  # last training month, used for tuning only
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
RIDGE_MAX_ROWS = 1_000_000  # a linear model needs no more; keeps memory low
REFIT_TREE_FACTOR = 1.1  # more trees when refitting on fit + tune (more data)
XGB_PARAMS = dict(
    objective="reg:absoluteerror",
    tree_method="hist",
    n_estimators=6000,  # upper bound: early stopping on the tuning month decides
    learning_rate=0.1,  # 0.05 hit the 6000-tree cap at 15 / 30 min on the full data
    max_depth=8,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    max_cat_to_onehot=1,
    early_stopping_rounds=50,
    eval_metric="mae",
    enable_categorical=True,
    n_jobs=-1,
)

_BASE = ["current", "timetable", "context", "prior"]
FEATURE_SETS: dict[str, list[str]] = {
    "current": _BASE,
    "history_1": [*_BASE, "history_1"],
    "history_3": [*_BASE, "history_1", "history_3"],
    "history_5": [*_BASE, "history_1", "history_3", "history_5"],
    "full": [*_BASE, "history_1", "history_3", "history_5", "slack"],
    "full_no_prior": ["current", "timetable", "context", "history_1", "history_3", "history_5",
                      "slack"],
}  # fmt: skip
EVAL_COLUMNS = [
    "operating_day", "trip_key", "stop_seq", "horizon_min", "split", "delta_min",
    "in_common_subset", "is_disruption_day", "category", "d0_min", "pred_historical",
]  # fmt: skip
PREDICTIONS_DIR = config.PROCESSED
REPORTS_DIR = config.REPORTS
MODELS_DIR = config.ROOT / "models"


def feature_columns(set_name: str) -> list[str]:
    cols: list[str] = []
    for group in FEATURE_SETS[set_name]:
        cols += [c for c in FEATURE_GROUPS[group] if c not in cols]
    return cols


def numeric_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in CATEGORICAL_FEATURES]


# --------------------------------------------------------------------------- data


def load(
    con: duckdb.DuckDBPyConnection,
    where: str,
    columns: list[str],
    source: str | None = None,
    sample: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Load feature rows matching ``where`` (optionally a random share of journeys)."""
    source = source or f"'{config.PROCESSED}/features_*.parquet'"
    cols = ", ".join(dict.fromkeys(columns))
    sampling = ""
    if sample < 1.0:  # sample whole journeys, so a journey's points stay together
        sampling = f"AND hash(trip_key || '{seed}') % 1000 < {int(sample * 1000)}"
    df = con.sql(f"SELECT {cols} FROM read_parquet({source}) WHERE {where} {sampling}").df()
    for c in df.columns:  # boolean *features* as 0 / 1; bookkeeping flags stay boolean
        if df[c].dtype == bool and c not in EVAL_COLUMNS:
            df[c] = df[c].astype("int8")
    return df


def to_xgb_matrix(df: pd.DataFrame, cols: list[str], categories: dict[str, list]) -> pd.DataFrame:
    """Feature matrix for XGBoost: float32 numbers, categories with the training levels."""
    x = df[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = pd.Categorical(x[c], categories=categories[c])
        else:
            x[c] = x[c].astype("float32")
    return x


# --------------------------------------------------------------------------- models


def ridge_pipeline(cols: list[str], alpha: float) -> Pipeline:
    num = numeric_columns(cols)
    cat = [c for c in cols if c in CATEGORICAL_FEATURES] + (["hour"] if "hour" in cols else [])
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                num,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), cat),
        ]  # fmt: skip
    )
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def fit_ridge(fit: pd.DataFrame, tune: pd.DataFrame, cols: list[str]) -> tuple[Pipeline, dict]:
    """Choose the penalty on the tuning month, then refit on fit + tune.

    Uses at most ``RIDGE_MAX_ROWS`` random rows of each set (plenty for a linear model).
    """
    if len(fit) > RIDGE_MAX_ROWS:
        fit = fit.sample(RIDGE_MAX_ROWS, random_state=0)
    if len(tune) > RIDGE_MAX_ROWS:
        tune = tune.sample(RIDGE_MAX_ROWS, random_state=0)
    scores = {}
    for alpha in RIDGE_ALPHAS:
        model = ridge_pipeline(cols, alpha).fit(fit[cols], fit["delta_min"])
        scores[alpha] = float(np.mean(np.abs(tune["delta_min"] - model.predict(tune[cols]))))
    best = min(scores, key=scores.get)
    both = pd.concat([fit, tune], ignore_index=True)
    model = ridge_pipeline(cols, best).fit(both[cols], both["delta_min"])
    return model, {"alpha": best, "tune_mae": scores[best]}


def fit_xgb(
    fit: pd.DataFrame, tune: pd.DataFrame, cols: list[str], params: dict | None = None
) -> tuple[xgb.XGBRegressor, dict]:
    """Fit on ``fit``, early-stop on ``tune``."""
    categories = {
        c: sorted(fit[c].dropna().astype(str).unique()) for c in cols if c in CATEGORICAL_FEATURES
    }
    for c in categories:  # categories as strings everywhere
        fit[c], tune[c] = fit[c].astype(str), tune[c].astype(str)
    model = xgb.XGBRegressor(**(params or XGB_PARAMS))
    model.fit(
        to_xgb_matrix(fit, cols, categories),
        fit["delta_min"],
        eval_set=[(to_xgb_matrix(tune, cols, categories), tune["delta_min"])],
        verbose=False,
    )
    model.categories_ = categories  # kept for prediction
    return model, {"best_iteration": int(model.best_iteration), "tune_mae": float(model.best_score)}


def refit_xgb(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    cols: list[str],
    n_trees: int,
    params: dict | None = None,
) -> xgb.XGBRegressor:
    """Step 2: a fresh model on fit + tune with a fixed number of trees (no early stopping).

    ``n_trees`` comes from step 1 (early stopping on the tuning month), scaled by
    ``REFIT_TREE_FACTOR`` because the model now sees more data.
    """
    both = pd.concat([fit, tune], ignore_index=True)
    categories = {
        c: sorted(both[c].dropna().astype(str).unique()) for c in cols if c in CATEGORICAL_FEATURES
    }
    for c in categories:
        both[c] = both[c].astype(str)
    p = dict(params or XGB_PARAMS)
    p.pop("early_stopping_rounds", None)
    p["n_estimators"] = n_trees
    model = xgb.XGBRegressor(**p)
    model.fit(to_xgb_matrix(both, cols, categories), both["delta_min"], verbose=False)
    model.categories_ = categories
    return model


def predict(model, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if isinstance(model, xgb.XGBRegressor):
        x = df[cols].copy()
        for c in model.categories_:
            x[c] = x[c].astype(str)
        return model.predict(to_xgb_matrix(x, cols, model.categories_))
    return model.predict(df[cols])


# --------------------------------------------------------------------------- driver


def run(
    models: list[str],
    sets: list[str],
    horizons: tuple[int, ...] = tuple(config.HORIZONS_MIN),
    eval_split: str = "valid",
    sample: float = 1.0,
    source: str | None = None,
    xgb_params: dict | None = None,
    out_dir: Path | None = None,
    refit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every (model, feature set) per horizon; return predictions and fit information.

    XGBoost is trained in two steps when ``refit`` is true: (1) fit on the fitting months,
    early-stopping on the tuning month, which fixes the number of trees; (2) a fresh model
    on fitting + tuning months with that number of trees. Predictions of both steps are
    kept: ``pred_xgb_<set>`` (final model) and ``pred_xgb_norefit_<set>`` (step 1), so the
    refit can be evaluated as an ablation.
    """
    con = duckdb.connect()
    all_cols = list(dict.fromkeys(c for s in sets for c in feature_columns(s)))
    load_cols = list(dict.fromkeys(EVAL_COLUMNS + all_cols))
    preds = []
    fit_info = []
    for h in horizons:
        t0 = time.time()
        fit = load(con, f"split = 'train' AND operating_day < DATE '{TUNE_START}' "
                        f"AND horizon_min = {h}", load_cols, source, sample)  # fmt: skip
        tune = load(con, f"split = 'train' AND operating_day >= DATE '{TUNE_START}' "
                         f"AND horizon_min = {h}", load_cols, source, sample)  # fmt: skip
        ev = load(con, f"split = '{eval_split}' AND horizon_min = {h}", load_cols, source)
        log.info("h=%d: fit %d rows, tune %d, %s %d", h, len(fit), len(tune), eval_split, len(ev))
        out = ev[EVAL_COLUMNS].copy()
        for set_name in sets:
            cols = feature_columns(set_name)
            for m in models:
                t1 = time.time()
                if m == "ridge":
                    model, info = fit_ridge(fit, tune, cols)
                else:
                    model, info = fit_xgb(fit.copy(), tune.copy(), cols, xgb_params)
                    if refit:
                        out[f"pred_xgb_norefit_{set_name}"] = predict(model, ev, cols)
                        n_trees = int(np.ceil((info["best_iteration"] + 1) * REFIT_TREE_FACTOR))
                        model = refit_xgb(fit, tune, cols, n_trees, xgb_params)
                        info["refit_trees"] = n_trees
                out[f"pred_{m}_{set_name}"] = predict(model, ev, cols)
                fit_info.append({"horizon_min": h, "model": m, "set": set_name, **info,
                                 "seconds": round(time.time() - t1)})  # fmt: skip
                log.info("h=%d %s/%s: %s (%.0fs)", h, m, set_name, info, time.time() - t1)
                if out_dir is not None:
                    _save(model, out_dir / f"{m}_{set_name}_h{h}")
        preds.append(out)
        log.info("h=%d done in %.0fs", h, time.time() - t0)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(fit_info)


def _save(model, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, xgb.XGBRegressor):
        model.save_model(stem.with_suffix(".json"))
    else:
        import joblib

        joblib.dump(model, stem.with_suffix(".joblib"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=["ridge", "xgb"], choices=["ridge", "xgb"])
    parser.add_argument("--sets", nargs="+", default=["full"],
                        help=f"feature sets, or 'all': {', '.join(FEATURE_SETS)}")  # fmt: skip
    parser.add_argument("--sample", type=float, default=1.0, help="share of training journeys")
    parser.add_argument("--test", action="store_true", help="evaluate on the test month")
    parser.add_argument("--learning-rate", type=float, default=XGB_PARAMS["learning_rate"],
                        help="XGBoost learning rate")  # fmt: skip
    parser.add_argument(
        "--no-refit", action="store_true", help="skip XGBoost's refit on fitting + tuning months"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sets = list(FEATURE_SETS) if args.sets == ["all"] else args.sets
    split = "test" if args.test else "valid"
    xgb_params = XGB_PARAMS | {"learning_rate": args.learning_rate}
    preds, info = run(
        args.models,
        sets,
        eval_split=split,
        sample=args.sample,
        xgb_params=xgb_params,
        out_dir=MODELS_DIR,
        refit=not args.no_refit,
    )

    model_cols = [c for c in preds.columns if c.startswith(("pred_ridge", "pred_xgb"))]
    pred_cols = ["pred_historical", *model_cols]
    path = PREDICTIONS_DIR / f"tabular_predictions_{split}.parquet"
    preds.to_parquet(path, index=False)
    log.info("Wrote %s", path)

    common = preds[preds["in_common_subset"].astype(bool)]
    res = evaluate(common, pred_cols, reference="pred_historical")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS_DIR / f"tabular_{split}.csv", index=False)
    info.to_csv(REPORTS_DIR / f"tabular_fit_info_{split}.csv", index=False)
    print(f"\n{split} — common subset (MAE in min, 95 % day-bootstrap CI; diff vs historical)")
    cols = ["horizon_min", "model", "mae", "mae_lo", "mae_hi", "p90_abs_err", "diff_vs_ref",
            "diff_lo", "diff_hi"]  # fmt: skip
    print(res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
