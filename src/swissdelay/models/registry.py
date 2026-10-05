"""The deployed model as a self-contained artifact: export once, load anywhere.

``models/champion/`` holds everything needed to predict without the training code's
in-memory state:

- ``xgb_p50_h{H}.json``, ``xgb_q10_h{H}.json``, ``xgb_q90_h{H}.json``: the champion
  (XGBoost ``full_network_plus``, refit on Aug 2025 – Apr 2026) and its two quantile models;
- ``manifest.json``: feature set and columns, category levels of the categorical features
  per horizon (the codes XGBoost was trained with), conformal offsets of the intervals
  (calibrated on April 2026), the training period, and reference distributions for drift
  monitoring.

Usage::

    uv run python -m swissdelay.models.registry          # export models/champion/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from swissdelay import config
from swissdelay.features.build import CATEGORICAL_FEATURES
from swissdelay.models import quantile as qt
from swissdelay.models import tabular as tb

log = logging.getLogger("swissdelay.registry")

CHAMPION_DIR = tb.MODELS_DIR / "champion"
FEATURE_SET = "full_network_plus"
DRIFT_FEATURES = ("d0_min", "hour", "net_cur_n")
DRIFT_BINS = 10


def drift_reference(values: pd.Series, bins: int = DRIFT_BINS) -> dict:
    """Bin edges (training deciles, open-ended) and the training share in each bin."""
    v = values.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
    return {"edges": edges.tolist(), "shares": (counts / counts.sum()).tolist(),
            "missing_share": float(values.isna().mean())}  # fmt: skip


def psi(reference: dict, values: pd.Series, eps: float = 1e-4) -> float:
    """Population stability index of ``values`` against a :func:`drift_reference`.
    Rule of thumb: < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 large shift."""
    v = values.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return float("nan")
    edges = np.asarray(reference["edges"])
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
    actual = np.clip(counts / counts.sum(), eps, None)
    expected = np.clip(np.asarray(reference["shares"]), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def export(
    out_dir: Path = CHAMPION_DIR,
    feature_set: str = FEATURE_SET,
    models_dir: Path = tb.MODELS_DIR,
    # the saved models come from the final (test) runs; their offsets were computed in the
    # same runs, on April 2026 (never on June)
    offsets_path: Path = config.REPORTS / "quantile_offsets_test.csv",
    source: str | None = None,
    horizons: tuple[int, ...] = tuple(config.HORIZONS_MIN),
) -> dict:
    """Copy the champion and quantile models and write the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = tb.feature_columns(feature_set)
    cats = [c for c in cols if c in CATEGORICAL_FEATURES]
    con = duckdb.connect()
    # the refit models were trained on all training months: categories come from there
    train = tb.load(con, "split = 'train'", ["horizon_min", *cats, *DRIFT_FEATURES], source)
    # same rule as tabular.refit_xgb: sorted levels seen in the training rows of each horizon
    categories = {
        str(h): {c: sorted(g[c].dropna().astype(str).unique()) for c in cats}
        for h, g in train.groupby("horizon_min")
    }
    offsets = pd.read_csv(offsets_path)
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "feature_set": feature_set,
        "feature_columns": cols,
        "categories": categories,
        "trained_on": [config.PERIOD_START, config.TRAIN_END],
        "horizons": list(horizons),
        "offsets": {
            str(h): dict(zip(g["d0_bucket"], g["offset_min"].astype(float), strict=True))
            for h, g in offsets.groupby("horizon_min")
        },
        "drift_reference": {
            str(h): {f: drift_reference(g[f]) for f in DRIFT_FEATURES}
            for h, g in train.groupby("horizon_min")
        },
    }
    for h in horizons:
        sources = {"p50": f"xgb_{feature_set}_h{h}", "q10": f"xgb_q10_{feature_set}_h{h}",
                   "q90": f"xgb_q90_{feature_set}_h{h}"}  # fmt: skip
        for name, src in sources.items():
            shutil.copyfile(models_dir / f"{src}.json", out_dir / f"xgb_{name}_h{h}.json")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("Exported %s (%d horizons, %d features)", out_dir, len(horizons), len(cols))
    return manifest


@dataclass
class Champion:
    """Loaded champion: point forecast, quantiles and calibrated intervals."""

    manifest: dict
    models: dict[tuple[int, str], xgb.Booster]

    @property
    def columns(self) -> list[str]:
        return self.manifest["feature_columns"]

    def matrix(self, df: pd.DataFrame, horizon: int) -> xgb.DMatrix:
        levels = self.manifest["categories"][str(horizon)]
        x = df[self.columns].copy()
        for c in self.columns:
            if c in levels:
                x[c] = pd.Categorical(x[c].astype(str), categories=levels[c])
            else:
                x[c] = x[c].astype("float32")
        return xgb.DMatrix(x, enable_categorical=True)

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """P50, raw P10 / P90 and calibrated interval (offsets of the manifest) for rows of
        one horizon."""
        m = self.matrix(df, horizon)
        p50, q10, q90 = (self.models[(horizon, k)].predict(m) for k in ("p50", "q10", "q90"))
        shift = qt.d0_bucket(df["d0_min"]).map(self.manifest["offsets"][str(horizon)]).fillna(0)
        # P50 is always the champion's own forecast; the interval is widened to contain it
        # where the separately trained quantile models cross it
        lo = np.minimum(q10 - shift.to_numpy(), p50)
        hi = np.maximum(q90 + shift.to_numpy(), p50)
        return pd.DataFrame({"p50": p50, "q10": q10, "q90": q90, "lo": lo, "hi": hi},
                            index=df.index)  # fmt: skip


def load(path: Path = CHAMPION_DIR) -> Champion:
    manifest = json.loads((path / "manifest.json").read_text())
    models = {}
    for h in manifest["horizons"]:
        for k in ("p50", "q10", "q90"):
            booster = xgb.Booster()
            booster.load_model(path / f"xgb_{k}_h{h}.json")
            models[(h, k)] = booster
    return Champion(manifest, models)


def verify(champion: Champion, split: str = "test", n: int = 20_000) -> dict[int, float]:
    """Largest difference between the loaded champion and the predictions saved by the run
    that trained these model files (the final ``--test`` run): should be ~1e-6. Checks
    columns, categories and models end to end."""
    saved = pd.read_parquet(tb.PREDICTIONS_DIR / f"tabular_predictions_{split}.parquet",
                            columns=[*qt.KEYS, f"pred_xgb_{FEATURE_SET}"])  # fmt: skip
    con = duckdb.connect()
    out = {}
    for h in champion.manifest["horizons"]:
        rows = tb.load(con, f"split = '{split}' AND horizon_min = {h}",
                       [*qt.KEYS, *champion.columns]).head(n)  # fmt: skip
        got = champion.predict(rows, h)["p50"].to_numpy()
        ref = rows[qt.KEYS].merge(saved, on=qt.KEYS, how="left")[f"pred_xgb_{FEATURE_SET}"]
        diff = np.abs(got - ref.to_numpy())
        out[h] = float(np.nanmax(diff))
        log.info("h=%d: %d rows checked, %.2f %% differ by more than 1e-4, max %.2g", h,
                 len(diff), 100 * np.mean(diff > 1e-4), out[h])  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = export()
    diffs = verify(load())
    log.info("max |loaded − saved| P50 by horizon: %s", diffs)
    if max(diffs.values()) > 1e-3:
        raise SystemExit("Loaded champion does not reproduce the saved predictions")
    short = {k: v for k, v in manifest.items() if k not in ("drift_reference", "categories")}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
