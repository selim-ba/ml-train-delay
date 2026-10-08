"""Simulated production: replay Jul – Sep 2026 day by day with the frozen champion.

For each operating day, in order, as a nightly job would:

1. score the day's prediction points with the exported champion (``models/champion/``):
   P50 and three versions of the P10–P90 interval:

   - ``static``: conformal offsets calibrated on April 2026 (as in the test month);
   - ``rolling``: offsets recomputed per current-delay bucket on the last ``WINDOW_DAYS``
     replayed days (symmetric, like ``static``);
   - ``rolling_tails``: same window, one offset per tail, so each tail targets 10 %;

2. compare with what happened (realised Δd): MAE, P90 absolute error, skill vs the
   historical median, coverage and tails of each interval, interval width;
3. compare input distributions with training (PSI of current delay, hour, station traffic)
   and raise alerts (drift, low coverage, MAE above its recent level);
4. only then add the day to the rolling calibration window: a day never calibrates itself.

**This is a replay, not live inference.** Features of Jul – Sep were built with the same
code and leakage rules as for training (events before ``t − 2 min``), month by month; the
replay reads them one day at a time. The live pipeline would build the same features from
each new daily file.

Outputs: ``data/processed/replay/daily_metrics.parquet`` (also ``reports/daily_metrics.csv``),
``data/processed/replay/predictions_YYYY-MM.parquet``, and the final calibration state
``models/champion/calibration.json`` (the ``rolling_tails`` offsets of the last 14 days, used
by the API).

Usage::

    uv run python -m swissdelay.models.registry            # once: export the champion
    uv run python -m swissdelay.pipeline.replay            # 2026-07-01 … 2026-09-30
    uv run python -m swissdelay.pipeline.replay --start 2026-07-01 --end 2026-07-07
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from swissdelay import config
from swissdelay.models import quantile as qt
from swissdelay.models import registry
from swissdelay.models import tabular as tb

log = logging.getLogger("swissdelay.replay")

REPLAY_DIR = config.PROCESSED / "replay"
START, END = "2026-07-01", config.PERIOD_END
WINDOW_DAYS = 14
MIN_WINDOW_DAYS = 7  # before that, the rolling strategies fall back to the static offsets
TAIL = 0.10
STRATEGIES = ("static", "rolling", "rolling_tails")
ALERT_PSI = 0.25
ALERT_COVERAGE = 0.70
ALERT_MAE_RATIO = 1.25  # MAE above 1.25 × median of the previous WINDOW_DAYS days


def tail_offsets(q10: np.ndarray, q90: np.ndarray, y: np.ndarray, tail: float = TAIL):
    """One conformal offset per tail: ``[q10 − lo, q90 + hi]`` leaves ``tail`` below and
    ``tail`` above (finite-sample corrected quantiles)."""
    n = len(y)
    level = min(1.0, np.ceil((n + 1) * (1 - tail)) / n)
    lo = float(np.quantile(q10 - y, level, method="higher"))
    hi = float(np.quantile(y - q90, level, method="higher"))
    return lo, hi


class RollingCalibrator:
    """Residual history of the last ``window`` replayed days, per horizon."""

    def __init__(self, window: int = WINDOW_DAYS, min_days: int = MIN_WINDOW_DAYS,
                 min_points: int = qt.MIN_CAL_POINTS):  # fmt: skip
        self.days: deque[tuple[str, pd.DataFrame]] = deque(maxlen=window)
        self.min_days, self.min_points = min_days, min_points

    def add(self, day: str, frame: pd.DataFrame) -> None:
        """``frame``: horizon_min, d0_bucket, q10, q90, y of one day (after it is scored)."""
        self.days.append((day, frame[["horizon_min", "d0_bucket", "q10", "q90", "y"]]))

    @property
    def ready(self) -> bool:
        return len(self.days) >= self.min_days

    def offsets(self, horizon: int) -> tuple[dict, dict] | None:
        """``(symmetric, per_tail)`` offsets by current-delay bucket, or ``None`` if the
        window is too short. Small buckets use the offsets of the whole window."""
        if not self.ready:
            return None
        hist = pd.concat([f for _, f in self.days])
        hist = hist[hist["horizon_min"] == horizon]
        args = hist["q10"].to_numpy(), hist["q90"].to_numpy(), hist["y"].to_numpy()
        pooled_sym, pooled_tails = qt.conformal_offset(*args), tail_offsets(*args)
        sym, tails = {}, {}
        for b in qt.D0_BUCKETS[1]:
            g = hist[hist["d0_bucket"] == b]
            if len(g) >= self.min_points:
                a = g["q10"].to_numpy(), g["q90"].to_numpy(), g["y"].to_numpy()
                sym[b], tails[b] = qt.conformal_offset(*a), tail_offsets(*a)
            else:
                sym[b], tails[b] = pooled_sym, pooled_tails
        return sym, tails

    def last_days(self) -> list[str]:
        return [d for d, _ in self.days]


Bounds = dict[str, tuple[np.ndarray, np.ndarray]]


def intervals(
    pred: pd.DataFrame, buckets: pd.Series, static: dict, rolling: tuple[dict, dict] | None
) -> Bounds:
    """Lower and upper bounds of each strategy, sorted with P50 (no crossing)."""

    def bounds(lo_shift, hi_shift):
        lo, _, hi = qt.sort_quantiles(pred["q10"].to_numpy() - lo_shift, pred["p50"].to_numpy(),
                                      pred["q90"].to_numpy() + hi_shift)  # fmt: skip
        return lo, hi

    s = buckets.map(static).fillna(0.0).to_numpy()
    out = {"static": bounds(s, s)}
    if rolling is None:  # not enough history yet
        out["rolling"] = out["rolling_tails"] = out["static"]
    else:
        sym, tails = rolling
        r = buckets.map(sym).fillna(0.0).to_numpy()
        lo_t = buckets.map({k: v[0] for k, v in tails.items()}).fillna(0.0).to_numpy()
        hi_t = buckets.map({k: v[1] for k, v in tails.items()}).fillna(0.0).to_numpy()
        out["rolling"], out["rolling_tails"] = bounds(r, r), bounds(lo_t, hi_t)
    return out


def score_horizon(champion, rows: pd.DataFrame, horizon: int, cal: RollingCalibrator,
                  references: dict) -> tuple[dict, pd.DataFrame]:  # fmt: skip
    """Metrics row and prediction frame for one day and horizon."""
    pred = champion.predict(rows, horizon)
    y = rows["delta_min"].to_numpy()
    buckets = qt.d0_bucket(rows["d0_min"])
    bounds = intervals(pred, buckets, champion.manifest["offsets"][str(horizon)],
                       cal.offsets(horizon))  # fmt: skip
    err = np.abs(y - pred["p50"].to_numpy())
    m = {
        "horizon_min": horizon, "n": len(y),
        "is_disruption_day": bool(rows["is_disruption_day"].astype(bool).any()),
        "mae": float(err.mean()), "p90_abs_err": float(np.quantile(err, 0.9)),
        "mae_historical": float(np.mean(np.abs(y - rows["pred_historical"].to_numpy()))),
        "rolling_window_days": len(cal.days),
    }  # fmt: skip
    m["skill_vs_historical"] = 1 - m["mae"] / m["mae_historical"]
    for name, (lo, hi) in bounds.items():
        m[f"coverage_{name}"] = float(np.mean((y >= lo) & (y <= hi)))
        m[f"below_{name}"] = float(np.mean(y < lo))
        m[f"above_{name}"] = float(np.mean(y > hi))
        m[f"width_{name}"] = float(np.median(hi - lo))
    for f, ref in references[str(horizon)].items():
        m[f"psi_{f}"] = registry.psi(ref, rows[f])
    frame = rows[[*qt.KEYS, "operating_day"]].assign(
        y=y, p50=pred["p50"].to_numpy(), q10=pred["q10"].to_numpy(), q90=pred["q90"].to_numpy(),
        d0_bucket=buckets.to_numpy(),
        **{f"{k}_{n}": v for n, (lo, hi) in bounds.items() for k, v in (("lo", lo), ("hi", hi))},
    )  # fmt: skip
    return m, frame


def alerts(row: dict, history: pd.DataFrame) -> str:
    """Comma-separated alerts for one day and horizon (empty if none)."""
    out = [f"drift:{k[4:]}" for k, v in row.items() if k.startswith("psi_") and v > ALERT_PSI]
    if row["coverage_rolling_tails"] < ALERT_COVERAGE:
        out.append("low_coverage")
    if history.empty:
        return ",".join(out)
    past = history[history["horizon_min"] == row["horizon_min"]].tail(WINDOW_DAYS)
    if len(past) >= MIN_WINDOW_DAYS and row["mae"] > ALERT_MAE_RATIO * past["mae"].median():
        out.append("mae_jump")
    return ",".join(out)


def _day_line(metrics: list[dict], day: str) -> str:
    parts = []
    for m in (m for m in metrics if m["operating_day"] == day):
        cov = m["coverage_rolling_tails"]
        parts.append(f"h{m['horizon_min']} MAE {m['mae']:.3f} cov {cov:.2f}")
        if m["alerts"]:
            parts[-1] += f" [{m['alerts']}]"
    return "  ".join(parts)


def day_rows(con: duckdb.DuckDBPyConnection, day: str, columns: list[str]) -> pd.DataFrame:
    """Labelled prediction points of one operating day, all horizons."""
    cols = list(dict.fromkeys([*tb.EVAL_COLUMNS, *columns, *registry.DRIFT_FEATURES]))
    return tb.load(con, f"operating_day = DATE '{day}' AND delta_min IS NOT NULL", cols)


def replay(start: str = START, end: str = END, champion=None,
           calibration_dir: Path | None = registry.CHAMPION_DIR) -> pd.DataFrame:  # fmt: skip
    champion = champion or registry.load()
    con = duckdb.connect()
    cal = RollingCalibrator()
    metrics: list[dict] = []
    frames: dict[str, list[pd.DataFrame]] = {}
    for day in pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"):
        t0 = time.time()
        rows = day_rows(con, day, champion.columns)
        if rows.empty:
            log.warning("%s: no labelled points", day)
            continue
        day_frames = []
        for h, g in rows.groupby("horizon_min"):
            m, frame = score_horizon(champion, g, int(h), cal, champion.manifest["drift_reference"])
            m = {"operating_day": day, **m}
            m["alerts"] = alerts(m, pd.DataFrame(metrics))
            metrics.append(m)
            day_frames.append(frame)
        day_frame = pd.concat(day_frames)
        cal.add(day, day_frame)  # scored: from now on the day may calibrate later days
        frames.setdefault(day[:7], []).append(day_frame)
        log.info("%s: %s (%.1fs)", day, _day_line(metrics, day), time.time() - t0)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    for month, fs in frames.items():
        pd.concat(fs).to_parquet(REPLAY_DIR / f"predictions_{month}.parquet", index=False)
    if calibration_dir is not None and cal.ready:
        offsets = {h: cal.offsets(h)[1] for h in champion.manifest["horizons"]}
        path = registry.save_calibration(offsets, cal.last_days(), calibration_dir)
        log.info("Saved rolling_tails offsets of %s … %s to %s", cal.last_days()[0],
                 cal.last_days()[-1], path)  # fmt: skip
    out = pd.DataFrame(metrics)
    out.to_parquet(REPLAY_DIR / "daily_metrics.parquet", index=False)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "daily_metrics.csv", index=False)
    return out


def summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Point-weighted summary per horizon (and disruption days) of the replay."""
    rows = []
    for (h, dis), g in metrics.groupby(["horizon_min", "is_disruption_day"]):
        w = g["n"] / g["n"].sum()
        r = {"horizon_min": h, "disruption_day": dis, "days": len(g), "points": int(g["n"].sum()),
             "mae": float((w * g["mae"]).sum()),
             "mae_historical": float((w * g["mae_historical"]).sum())}  # fmt: skip
        for s in STRATEGIES:
            r[f"coverage_{s}"] = float((w * g[f"coverage_{s}"]).sum())
            r[f"above_{s}"] = float((w * g[f"above_{s}"]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = replay(args.start, args.end)
    print(f"\nReplay {args.start} … {args.end}: {metrics['operating_day'].nunique()} days")
    print(summary(metrics).round(3).to_string(index=False))
    n_alerts = (metrics["alerts"] != "").sum()
    print(f"\n{n_alerts} day-horizon alerts; see reports/daily_metrics.csv")


if __name__ == "__main__":
    main()
