"""Baselines for Δd (change in delay), fitted on the training months only.

- **persistence:** Δd = 0, the delay stays the same.
- **offset:** Δd = median of the training labels for the horizon (≈ −0.8 min). It captures
  the structural gap between a departure delay now and an arrival delay at the target
  (trains tend to arrive slightly early); any model must beat it to show real skill.
- **historical:** median training Δd for the same line, station, hour and day type
  (weekday / Saturday / Sunday), with fallbacks to coarser groups when a combination has
  fewer than ``MIN_COUNT`` training labels:
  (line, station, hour, day type) → (line, station, hour) → (line, station) → (station)
  → horizon only (= offset). The median is used because the metric is MAE.

Output: ``data/processed/baseline_predictions.parquet``, one row per labelled point and
horizon, with ``pred_persistence``, ``pred_offset``, ``pred_historical`` and
``hist_level`` (which fallback level answered).

Usage::

    uv run python -m swissdelay.models.baselines
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from swissdelay import config
from swissdelay.evaluation.days import DAILY_STATS_PATH, build_daily_stats
from swissdelay.evaluation.metrics import evaluate
from swissdelay.evaluation.splits import split_sql

MIN_COUNT = 30
LEVELS: list[tuple[str, ...]] = [
    ("line_name", "station_id", "hour", "day_type"),
    ("line_name", "station_id", "hour"),
    ("line_name", "station_id"),
    ("station_id",),
    (),
]
OFFSET_LEVEL = len(LEVELS) - 1
PRED_COLS = ["pred_persistence", "pred_offset", "pred_historical"]
PREDICTIONS_PATH = config.PROCESSED / "baseline_predictions.parquet"

FEATURES_SQL = (
    "hour(dep_planned) AS hour, "
    "CASE dayofweek(operating_day) WHEN 0 THEN 'sunday' WHEN 6 THEN 'saturday' "
    "ELSE 'weekday' END AS day_type"
)


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def create_labelled_view(con: duckdb.DuckDBPyConnection, labels_source: str) -> None:
    """View ``lab``: labelled points with split and the grouping features."""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW lab AS
        SELECT *, {FEATURES_SQL}, {split_sql()} AS split
        FROM read_parquet({labels_source})
        WHERE label_status = 'ok'
        """
    )


def fit(con: duckdb.DuckDBPyConnection, train_filter: str = "split = 'train'") -> None:
    """Create one table of training medians per fallback level: ``hist_0`` … ``hist_4``."""
    for i, keys in enumerate(LEVELS):
        cols = ", ".join((*keys, "horizon_min"))
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE hist_{i} AS
            SELECT {cols}, count(*) AS n, median(delta_min) AS value
            FROM lab WHERE {train_filter}
            GROUP BY {cols}
            HAVING count(*) >= {MIN_COUNT}
            """
        )


def predict_sql(where: str = "TRUE", daily_stats: Path | None = DAILY_STATS_PATH) -> str:
    """Predictions of the three baselines for the rows of ``lab`` matching ``where``."""
    joins, values = [], []
    for i, keys in enumerate(LEVELS):
        cond = " AND ".join(
            [f"h{i}.{k} = l.{k}" for k in keys] + [f"h{i}.horizon_min = l.horizon_min"]
        )
        joins.append(f"LEFT JOIN hist_{i} h{i} ON {cond}")
        values.append(f"h{i}.value")
    level = (
        "CASE "
        + " ".join(f"WHEN h{i}.value IS NOT NULL THEN {i}" for i in range(len(LEVELS)))
        + " END"
    )
    daily_join = (
        f"LEFT JOIN read_parquet('{_sql_path(daily_stats)}') d USING (operating_day)"
        if daily_stats is not None
        else ""
    )
    disruption = "coalesce(d.is_disruption_day, false)" if daily_stats is not None else "false"
    return f"""
    SELECT l.operating_day, l.split, l.trip_key, l.stop_seq, l.category, l.operator_abbr,
           l.line_name, l.station_id, l.hour, l.day_type, l.horizon_min, l.sched_gap_min,
           l.d0_min, l.delta_min, l.in_common_subset,
           {disruption}                         AS is_disruption_day,
           0.0                                  AS pred_persistence,
           h{OFFSET_LEVEL}.value                AS pred_offset,
           coalesce({", ".join(values)})        AS pred_historical,
           {level}                              AS hist_level
    FROM lab l
    {" ".join(joins)}
    {daily_join}
    WHERE {where}
    """


def main() -> None:
    con = duckdb.connect()
    journeys = sorted(config.PROCESSED.glob("journeys_*.parquet"))
    if not journeys or not list(config.PROCESSED.glob("labels_*.parquet")):
        raise SystemExit("Run swissdelay.data.journeys and swissdelay.features.labels first")

    stats = build_daily_stats(con, journeys)
    print(f"Disruption days: {int(stats['is_disruption_day'].sum())} of {len(stats)}")

    create_labelled_view(con, f"'{_sql_path(config.PROCESSED)}/labels_*.parquet'")
    fit(con)
    for i, keys in enumerate(LEVELS):
        n = con.execute(f"SELECT count(*) FROM hist_{i}").fetchone()[0]
        print(f"level {i} {keys or ('horizon',)}: {n:,} groups")

    out = _sql_path(PREDICTIONS_PATH)
    con.execute(f"COPY ({predict_sql()}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    print(f"Wrote {PREDICTIONS_PATH}")

    # Headline: common subset, validation and test months
    for split in ("valid", "test"):
        df: pd.DataFrame = con.sql(
            f"SELECT * FROM read_parquet('{_sql_path(PREDICTIONS_PATH)}') "
            f"WHERE split = '{split}' AND in_common_subset"
        ).df()
        res = evaluate(df, PRED_COLS, reference="pred_offset")
        print(f"\n{split} — common subset (MAE in min, 95 % day-bootstrap CI)")
        print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
