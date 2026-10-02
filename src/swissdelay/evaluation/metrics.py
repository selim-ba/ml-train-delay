"""Error metrics with day-clustered bootstrap confidence intervals.

Trains running on the same day share weather, incidents and works, so their errors are
correlated. Confidence intervals therefore resample **whole days**, not single points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 1000
CI_LEVEL = 0.95


def _day_sums(day: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-day sum of ``values`` and count, over the unique days."""
    _, idx = np.unique(day, return_inverse=True)
    sums = np.bincount(idx, weights=values)
    counts = np.bincount(idx).astype(float)
    return sums, counts


def _bootstrap_ratio(
    sums: np.ndarray, counts: np.ndarray, n_boot: int, seed: int, level: float
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = len(sums)
    draws = rng.integers(0, d, size=(n_boot, d))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return float(lo), float(hi)


def mae_ci(
    day: np.ndarray,
    abs_err: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    level: float = CI_LEVEL,
) -> tuple[float, float, float]:
    """MAE and its day-clustered bootstrap confidence interval: (mae, lo, hi)."""
    abs_err = np.asarray(abs_err, dtype=float)
    sums, counts = _day_sums(np.asarray(day), abs_err)
    lo, hi = _bootstrap_ratio(sums, counts, n_boot, seed, level)
    return float(abs_err.mean()), lo, hi


def mae_diff_ci(
    day: np.ndarray,
    abs_err_a: np.ndarray,
    abs_err_b: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    level: float = CI_LEVEL,
) -> tuple[float, float, float]:
    """MAE(a) − MAE(b) on the same points, with a paired day-clustered CI.

    Negative means model a is better. The difference is significant when the interval
    excludes 0.
    """
    diff = np.asarray(abs_err_a, dtype=float) - np.asarray(abs_err_b, dtype=float)
    sums, counts = _day_sums(np.asarray(day), diff)
    lo, hi = _bootstrap_ratio(sums, counts, n_boot, seed, level)
    return float(diff.mean()), lo, hi


def evaluate(
    df: pd.DataFrame,
    pred_cols: list[str],
    target: str = "delta_min",
    group_cols: tuple[str, ...] = ("horizon_min",),
    reference: str | None = None,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per group and model: n, MAE with CI, P90 absolute error.

    If ``reference`` is given, also the paired MAE difference vs that model with its CI
    (negative = better than the reference).
    """
    rows = []
    groups = df.groupby(list(group_cols), observed=True) if group_cols else [((), df)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        day = g["operating_day"].to_numpy()
        y = g[target].to_numpy(dtype=float)
        ref_err = np.abs(y - g[reference].to_numpy(dtype=float)) if reference else None
        for col in pred_cols:
            err = np.abs(y - g[col].to_numpy(dtype=float))
            mae, lo, hi = mae_ci(day, err, n_boot, seed)
            row = dict(zip(group_cols, key, strict=True))
            row |= {
                "model": col.removeprefix("pred_"),
                "n": len(g),
                "days": len(np.unique(day)),
                "mae": mae,
                "mae_lo": lo,
                "mae_hi": hi,
                "p90_abs_err": float(np.quantile(err, 0.9)),
            }
            if reference:
                d, dlo, dhi = mae_diff_ci(day, err, ref_err, n_boot, seed)
                row |= {"diff_vs_ref": d, "diff_lo": dlo, "diff_hi": dhi}
            rows.append(row)
    return pd.DataFrame(rows)
