"""Prediction intervals: quantile XGBoost for P10 and P90 of Δd.

The deployed point forecast is the champion (XGBoost ``full_network_plus``, absolute-error
objective, i.e. the conditional median P50). This module adds two models per horizon with
the **pinball (quantile) loss** at 0.1 and 0.9, with the same features, settings and
two-step training as the champion:

1. fit on Aug 2025 – Mar 2026, early stopping on April 2026 (pinball loss);
2. refit on Aug – Apr with 10 % more trees.

The interval ``[P10, P90]`` should contain about **80 %** of actual Δd values. Quantiles
fitted separately can cross; at prediction time ``(P10, P50, P90)`` are sorted per point,
and the share of crossings before sorting is reported.

**Conformal calibration** (conformalized quantile regression, by current-delay bucket):
the step-1 models (trained on Aug – Mar) predict April, which they never saw. For each
bucket, the conformity score ``max(P10 − y, y − P90)`` is computed on April, and its
``⌈(n + 1) · 0.8⌉ / n`` quantile ``Q`` widens (``Q > 0``) or narrows (``Q < 0``) the final
intervals: ``[P10 − Q, P90 + Q]``. Both raw and calibrated intervals are reported.

Outputs:

- ``data/processed/quantile_predictions_<split>.parquet``: keys, evaluation columns,
  ``pred_q10``, ``pred_q90`` (raw) and, when the champion's predictions of the same split
  are available, ``pred_p50`` and the sorted ``lo`` / ``hi``;
- ``reports/quantile_<split>.csv``: coverage, tail shares, width and pinball losses per
  horizon (and by current delay), on the common subset;
- ``models/xgb_q10_<set>_h*.json``, ``xgb_q90_<set>_h*.json``.

Usage::

    uv run python -m swissdelay.models.quantile                 # validation (May)
    uv run python -m swissdelay.models.quantile --sample 0.1    # quick check
    uv run python -m swissdelay.models.quantile --test          # June, final run only
"""

from __future__ import annotations

import argparse
import logging
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from swissdelay import config
from swissdelay.models import tabular as tb

log = logging.getLogger("swissdelay.quantile")

QUANTILES = (0.1, 0.9)
FEATURE_SET = "full_network_plus"
KEYS = ["trip_key", "stop_seq", "horizon_min"]
MIN_CAL_POINTS = 200  # smaller calibration buckets use the offset of the whole month
D0_BUCKETS = ([-99, 1, 3, 10, 999], ["on time < 1", "minor 1-3", "moderate 3-10", "major > 10"])


def quantile_params(alpha: float, base: dict | None = None) -> dict:
    """Champion settings with the pinball loss at ``alpha``."""
    p = dict(base or tb.XGB_PARAMS)
    p.update(objective="reg:quantileerror", quantile_alpha=alpha, eval_metric="quantile")
    return p


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """Mean pinball loss of quantile predictions ``q`` at level ``alpha``."""
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def sort_quantiles(*cols: np.ndarray) -> list[np.ndarray]:
    """Sort quantile predictions per point (removes crossings)."""
    s = np.sort(np.column_stack(cols), axis=1)
    return [s[:, i] for i in range(s.shape[1])]


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, q10: np.ndarray,
                     q90: np.ndarray) -> dict:  # fmt: skip
    """Coverage of ``[lo, hi]``, tail shares, width, pinball losses and crossing rate."""
    return dict(
        n=len(y),
        coverage=float(np.mean((y >= lo) & (y <= hi))),
        below_p10=float(np.mean(y < lo)),
        above_p90=float(np.mean(y > hi)),
        mean_width=float(np.mean(hi - lo)),
        median_width=float(np.median(hi - lo)),
        pinball_q10=pinball(y, q10, 0.1),
        pinball_q90=pinball(y, q90, 0.9),
        crossing_share=float(np.mean(q10 > q90)),
    )


def d0_bucket(d0: pd.Series) -> pd.Series:
    return pd.cut(d0, D0_BUCKETS[0], right=False, labels=D0_BUCKETS[1]).astype(str)


def conformal_offset(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, coverage: float = 0.8) -> float:
    """Split-conformal correction ``Q`` so that ``[lo − Q, hi + Q]`` covers ``coverage``."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def calibrate(tune: pd.DataFrame, q10: np.ndarray, q90: np.ndarray,
              min_points: int = MIN_CAL_POINTS) -> pd.DataFrame:  # fmt: skip
    """Offset per current-delay bucket, from out-of-sample predictions on the tuning month.
    A bucket with fewer than ``min_points`` points gets the offset of the whole month."""
    b = d0_bucket(tune["d0_min"]).to_numpy()
    y = tune["delta_min"].to_numpy()
    pooled = conformal_offset(q10, q90, y)
    rows = []
    for k in D0_BUCKETS[1]:
        m = b == k
        n = int(m.sum())
        own = n >= min_points
        offset = conformal_offset(q10[m], q90[m], y[m]) if own else pooled
        rows.append({"d0_bucket": k, "n_calibration": n, "offset_min": offset, "pooled": not own})
    return pd.DataFrame(rows)


def report(preds: pd.DataFrame) -> pd.DataFrame:
    """Interval metrics per horizon, overall and by current delay, on the common subset, for
    raw and (if present) calibrated intervals."""
    df = preds[preds["in_common_subset"].astype(bool)].copy()
    df["d0_bucket"] = d0_bucket(df["d0_min"])
    variants = [("raw", "lo", "hi")]
    if "lo_cal" in df:
        variants.append(("calibrated", "lo_cal", "hi_cal"))
    rows = []
    for interval, lo, hi in variants:
        for h, gh in df.groupby("horizon_min"):
            for bucket in ["all", *D0_BUCKETS[1]]:
                g = gh if bucket == "all" else gh[gh["d0_bucket"] == bucket]
                if g.empty:
                    continue
                m = interval_metrics(g["delta_min"].to_numpy(), g[lo].to_numpy(),
                                     g[hi].to_numpy(), g["pred_q10"].to_numpy(),
                                     g["pred_q90"].to_numpy())  # fmt: skip
                rows.append({"interval": interval, "horizon_min": h, "d0_bucket": bucket, **m})
    return pd.DataFrame(rows)


def run(
    eval_split: str = "valid",
    sample: float = 1.0,
    feature_set: str = FEATURE_SET,
    horizons: tuple[int, ...] = tuple(config.HORIZONS_MIN),
    source: str | None = None,
    params: dict | None = None,
    save_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-step quantile models (P10, P90) per horizon; predictions on ``eval_split``, with
    conformal offsets per current-delay bucket. Returns predictions, fit information and
    offsets."""
    con = duckdb.connect()
    cols = tb.feature_columns(feature_set)
    load_cols = list(dict.fromkeys(tb.EVAL_COLUMNS + cols))
    preds, info, offsets = [], [], []
    for h in horizons:
        fit = tb.load(con, f"split = 'train' AND operating_day < DATE '{tb.TUNE_START}' "
                           f"AND horizon_min = {h}", load_cols, source, sample)  # fmt: skip
        tune = tb.load(con, f"split = 'train' AND operating_day >= DATE '{tb.TUNE_START}' "
                            f"AND horizon_min = {h}", load_cols, source, sample)  # fmt: skip
        ev = tb.load(con, f"split = '{eval_split}' AND horizon_min = {h}", load_cols, source)
        log.info("h=%d: fit %d, tune %d, %s %d rows", h, len(fit), len(tune), eval_split, len(ev))
        out = ev[tb.EVAL_COLUMNS].copy()
        tune_pred = {}
        for alpha in QUANTILES:
            t0 = time.time()
            name = f"q{round(alpha * 100)}"
            p = quantile_params(alpha, params)
            model, fit_info = tb.fit_xgb(fit.copy(), tune.copy(), cols, p)
            tune_pred[name] = tb.predict(model, tune, cols)  # step 1: April is out of sample
            n_trees = int(np.ceil((fit_info["best_iteration"] + 1) * tb.REFIT_TREE_FACTOR))
            model = tb.refit_xgb(fit, tune, cols, n_trees, p)
            out[f"pred_{name}"] = tb.predict(model, ev, cols)
            info.append({"horizon_min": h, "quantile": alpha,
                         "best_iteration": fit_info["best_iteration"],
                         "tune_pinball": fit_info["tune_mae"], "refit_trees": n_trees,
                         "seconds": round(time.time() - t0)})  # fmt: skip
            log.info("h=%d %s: %s (%.0fs)", h, name, info[-1], time.time() - t0)
            if save_models:
                tb._save(model, tb.MODELS_DIR / f"xgb_{name}_{feature_set}_h{h}")
        cal = calibrate(tune, tune_pred["q10"], tune_pred["q90"]).assign(horizon_min=h)
        offsets.append(cal)
        q = dict(zip(cal["d0_bucket"], cal["offset_min"], strict=True))
        log.info("h=%d conformal offsets (min): %s", h, {k: round(v, 3) for k, v in q.items()})
        shift = d0_bucket(out["d0_min"]).map(q).fillna(0.0).to_numpy()
        out["pred_q10_cal"], out["pred_q90_cal"] = out["pred_q10"] - shift, out["pred_q90"] + shift
        preds.append(out)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(info), pd.concat(offsets)


def add_median(preds: pd.DataFrame, split: str, feature_set: str = FEATURE_SET) -> pd.DataFrame:
    """Join the champion's P50 (same split) and sort (P10, P50, P90) per point. Without it,
    only (P10, P90) are sorted."""
    path = tb.PREDICTIONS_DIR / f"tabular_predictions_{split}.parquet"
    col = f"pred_xgb_{feature_set}"
    out = preds.copy()
    if path.exists() and col in pq.read_schema(path).names:
        p50 = pd.read_parquet(path, columns=[*KEYS, col]).rename(columns={col: "pred_p50"})
        out = out.merge(p50, on=KEYS, how="left")
    has_p50 = "pred_p50" in out and out["pred_p50"].notna().all()
    if not has_p50:
        log.warning("champion predictions for %s not found: sorting P10 / P90 only", split)
    for suffix in ("", "_cal"):
        lo, hi = out[f"pred_q10{suffix}"], out[f"pred_q90{suffix}"]
        if has_p50:
            out[f"lo{suffix}"], _, out[f"hi{suffix}"] = sort_quantiles(lo, out["pred_p50"], hi)
        else:
            out[f"lo{suffix}"], out[f"hi{suffix}"] = sort_quantiles(lo, hi)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=float, default=1.0, help="share of training journeys")
    parser.add_argument("--test", action="store_true", help="evaluate on the test month")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    split = "test" if args.test else "valid"
    preds, info, offsets = run(split, args.sample)
    preds = add_median(preds, split)
    path = tb.PREDICTIONS_DIR / f"quantile_predictions_{split}.parquet"
    preds.to_parquet(path, index=False)
    rep = report(preds)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    rep.to_csv(config.REPORTS / f"quantile_{split}.csv", index=False)
    info.to_csv(config.REPORTS / f"quantile_fit_info_{split}.csv", index=False)
    offsets.to_csv(config.REPORTS / f"quantile_offsets_{split}.csv", index=False)
    log.info("Wrote %s", path)
    print(f"\n{split} — common subset: P10–P90 interval (target coverage 0.80, tails 0.10 each)")
    print(rep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
