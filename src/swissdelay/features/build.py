"""Model features for each labelled prediction point and horizon.

Every feature uses only information available at the prediction time ``t`` (the measured
departure at the current stop ``C``):

- the train's own stops up to ``C`` (measured delays of earlier stops, arrival and dwell at
  ``C``): they happen before ``t``;
- the **timetable** of the whole journey (planned times are known in advance);
- statistics **learned on the training months only** (typical running and dwell times,
  the historical-median prior).

Actual times of stops after ``C`` are never used (tested in ``tests/test_features.py``).

Slack: for each pair of consecutive stations, the 20th percentile of the measured running
time over the training months is a "fast but realistic" running time; likewise for dwell
times per station and category. The **reserve** of a segment (or stop) is its planned time
minus that reference. The slack to the target is the sum of the reserves on the segments
and intermediate stops between ``C`` and the target stop.

Outputs:

- ``data/processed/segment_stats.parquet`` and ``dwell_stats.parquet`` (training months);
- ``data/processed/features_YYYY-MM.parquet``: one row per labelled point and horizon.

Usage::

    uv run python -m swissdelay.features.build            # all months
    uv run python -m swissdelay.features.build 2026-05    # one month (reference stats reused)
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import duckdb

from swissdelay import config
from swissdelay.evaluation.splits import split_sql

log = logging.getLogger("swissdelay.features")

REFERENCE_QUANTILE = 0.2
MIN_COUNT = 20
SEGMENT_STATS_PATH = config.PROCESSED / "segment_stats.parquet"
DWELL_STATS_PATH = config.PROCESSED / "dwell_stats.parquet"
BASELINE_PREDICTIONS_PATH = config.PROCESSED / "baseline_predictions.parquet"

KEY_COLUMNS = ["operating_day", "trip_key", "stop_seq", "horizon_min"]
META_COLUMNS = ["split", "delta_min", "in_common_subset", "is_disruption_day", "line_name",
                "station_id", "target_seq"]  # fmt: skip

# Feature groups, used for the ablations
FEATURE_GROUPS: dict[str, list[str]] = {
    "current": ["d0_min", "arr_delay_now_min", "dwell_excess_now_min"],
    "history_1": ["delay_lag1_min", "delta_last1_min"],
    "history_3": ["delay_lag3_min", "delta_last3_min", "trend3_per10min"],
    "history_5": [
        "delay_lag5_min", "delta_last5_min", "trend5_per10min", "max_prev_delay_min",
        "n_observed_prev",
    ],
    "slack": ["slack_to_target_min", "run_reserve_to_target_min", "dwell_reserve_to_target_min"],
    "timetable": [
        "horizon_min", "sched_gap_min", "n_stops_to_target", "stop_seq", "stops_remaining",
        "frac_journey", "minutes_since_origin", "planned_dwell_now_min",
    ],
    "context": [
        "hour", "weekday", "day_type", "category", "operator_abbr", "enters_from_abroad",
        "is_extra_trip",
    ],
    "prior": ["pred_historical"],
}  # fmt: skip
CATEGORICAL_FEATURES = ["day_type", "category", "operator_abbr"]
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def _list(paths: list[Path]) -> str:
    return "[" + ", ".join(f"'{_sql_path(p)}'" for p in paths) + "]"


# --------------------------------------------------------------------------- reference stats


def segment_stats_sql(source: str, train_end: str) -> str:
    """Reference running time per pair of consecutive stations (training months only)."""
    return f"""
    WITH s AS (
      SELECT station_id AS from_station_id,
             lead(station_id) OVER w AS to_station_id,
             date_diff('second', dep_planned, lead(arr_planned) OVER w) / 60.0 AS planned_run_min,
             lead(arr_delay_min) OVER w - dep_delay_min AS run_gain_min
      FROM read_parquet({source})
      WHERE operating_day <= DATE '{train_end}'
      WINDOW w AS (PARTITION BY trip_key ORDER BY stop_seq)
    )
    SELECT from_station_id, to_station_id, count(*) AS n,
           median(planned_run_min) AS planned_run_median_min,
           quantile_cont(planned_run_min + run_gain_min, {REFERENCE_QUANTILE}) AS run_ref_min
    FROM s
    WHERE to_station_id IS NOT NULL AND run_gain_min IS NOT NULL AND planned_run_min IS NOT NULL
    GROUP BY ALL
    HAVING count(*) >= {MIN_COUNT}
    """


def dwell_stats_sql(source: str, train_end: str) -> str:
    """Reference dwell time per station and category (training months only)."""
    return f"""
    WITH s AS (
      SELECT station_id, category,
             date_diff('second', arr_planned, dep_planned) / 60.0 AS planned_dwell_min,
             dep_delay_min - arr_delay_min AS dwell_excess_min
      FROM read_parquet({source})
      WHERE operating_day <= DATE '{train_end}'
        AND arr_planned IS NOT NULL AND dep_planned IS NOT NULL
        AND NOT coalesce(is_pass_through, false)
    )
    SELECT station_id, category, count(*) AS n,
           quantile_cont(planned_dwell_min + dwell_excess_min, {REFERENCE_QUANTILE})
             AS dwell_ref_min
    FROM s
    WHERE dwell_excess_min IS NOT NULL
    GROUP BY ALL
    HAVING count(*) >= {MIN_COUNT}
    """


def build_reference_stats(
    con: duckdb.DuckDBPyConnection,
    journeys: list[Path],
    train_end: str = config.TRAIN_END,
    segment_path: Path = SEGMENT_STATS_PATH,
    dwell_path: Path = DWELL_STATS_PATH,
) -> None:
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    src = _list(journeys)
    for sql, path in ((segment_stats_sql(src, train_end), segment_path),
                      (dwell_stats_sql(src, train_end), dwell_path)):  # fmt: skip
        con.execute(f"COPY ({sql}) TO '{_sql_path(path)}' (FORMAT parquet, COMPRESSION zstd)")


# --------------------------------------------------------------------------- features


def features_sql(
    journeys: str,
    labels: str,
    segment_stats: Path = SEGMENT_STATS_PATH,
    dwell_stats: Path = DWELL_STATS_PATH,
    baseline: Path | None = BASELINE_PREDICTIONS_PATH,
) -> str:
    """Features for the labelled points of one month."""
    prior_join = (
        f"LEFT JOIN read_parquet('{_sql_path(baseline)}') b "
        "ON b.trip_key = l.trip_key AND b.stop_seq = l.stop_seq AND b.horizon_min = l.horizon_min"
    )
    prior_cols = "b.pred_historical, coalesce(b.is_disruption_day, false) AS is_disruption_day"
    if baseline is None:
        prior_join = ""
        prior_cols = "CAST(NULL AS DOUBLE) AS pred_historical, false AS is_disruption_day"
    return f"""
    WITH j AS (
      SELECT * FROM read_parquet({journeys})
    ),
    stops AS (
      SELECT j.*,
             lead(station_id) OVER w AS next_station_id,
             date_diff('second', dep_planned, lead(arr_planned) OVER w) / 60.0 AS planned_run_min,
             date_diff('second', arr_planned, dep_planned) / 60.0 AS planned_dwell_min
      FROM j
      WINDOW w AS (PARTITION BY trip_key ORDER BY stop_seq)
    ),
    res AS (
      SELECT s.*,
             s.planned_run_min - ss.run_ref_min     AS run_reserve_min,
             s.planned_dwell_min - ds.dwell_ref_min AS dwell_reserve_min
      FROM stops s
      LEFT JOIN read_parquet('{_sql_path(segment_stats)}') ss
        ON ss.from_station_id = s.station_id AND ss.to_station_id = s.next_station_id
      LEFT JOIN read_parquet('{_sql_path(dwell_stats)}') ds
        ON ds.station_id = s.station_id AND ds.category = s.category
    ),
    -- Per stop: cumulative reserves (timetable only) and history (earlier stops only)
    cum AS (
      SELECT *,
             coalesce(sum(coalesce(run_reserve_min, 0)) OVER wb, 0)   AS cum_run_before,
             coalesce(sum(coalesce(dwell_reserve_min, 0)) OVER wb, 0) AS cum_dwell_before,
             lag(dep_delay_min, 1) OVER w AS delay_lag1_min,
             lag(dep_delay_min, 3) OVER w AS delay_lag3_min,
             lag(dep_delay_min, 5) OVER w AS delay_lag5_min,
             date_diff('second', lag(dep_planned, 3) OVER w, dep_planned) / 60.0 AS gap_lag3_min,
             date_diff('second', lag(dep_planned, 5) OVER w, dep_planned) / 60.0 AS gap_lag5_min,
             max(dep_delay_min) OVER wb   AS max_prev_delay_min,
             count(dep_delay_min) OVER wb AS n_observed_prev,
             date_diff('second', min(dep_planned) OVER (PARTITION BY trip_key), dep_planned)
               / 60.0 AS minutes_since_origin
      FROM res
      WINDOW w AS (PARTITION BY trip_key ORDER BY stop_seq),
             wb AS (PARTITION BY trip_key ORDER BY stop_seq
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ),
    l AS (
      SELECT * FROM read_parquet({labels}) WHERE label_status = 'ok'
    )
    SELECT
      -- keys and metadata
      l.operating_day, l.trip_key, l.stop_seq, l.horizon_min,
      {split_sql("l.operating_day")} AS split,
      l.delta_min, l.in_common_subset, {prior_cols},
      l.line_name, l.station_id, l.target_seq,
      -- current
      l.d0_min,
      c.arr_delay_min                            AS arr_delay_now_min,
      c.dep_delay_min - c.arr_delay_min          AS dwell_excess_now_min,
      -- history
      c.delay_lag1_min, l.d0_min - c.delay_lag1_min AS delta_last1_min,
      c.delay_lag3_min, l.d0_min - c.delay_lag3_min AS delta_last3_min,
      10 * (l.d0_min - c.delay_lag3_min) / nullif(c.gap_lag3_min, 0) AS trend3_per10min,
      c.delay_lag5_min, l.d0_min - c.delay_lag5_min AS delta_last5_min,
      10 * (l.d0_min - c.delay_lag5_min) / nullif(c.gap_lag5_min, 0) AS trend5_per10min,
      c.max_prev_delay_min, c.n_observed_prev,
      -- slack between now and the target (timetable + training references)
      (t.cum_run_before - c.cum_run_before)
        + (t.cum_dwell_before - c.cum_dwell_before - coalesce(c.dwell_reserve_min, 0))
                                                  AS slack_to_target_min,
      t.cum_run_before - c.cum_run_before         AS run_reserve_to_target_min,
      t.cum_dwell_before - c.cum_dwell_before - coalesce(c.dwell_reserve_min, 0)
                                                  AS dwell_reserve_to_target_min,
      -- timetable
      l.sched_gap_min,
      l.target_seq - l.stop_seq                   AS n_stops_to_target,
      c.n_stops - l.stop_seq                      AS stops_remaining,
      l.stop_seq / c.n_stops                      AS frac_journey,
      c.minutes_since_origin,
      c.planned_dwell_min                         AS planned_dwell_now_min,
      -- context
      hour(l.dep_planned)                         AS hour,
      isodow(l.operating_day)                     AS weekday,
      CASE dayofweek(l.operating_day) WHEN 0 THEN 'sunday' WHEN 6 THEN 'saturday'
           ELSE 'weekday' END                     AS day_type,
      l.category, l.operator_abbr,
      c.enters_from_abroad, coalesce(c.is_extra_trip, false) AS is_extra_trip
    FROM l
    JOIN cum c ON c.trip_key = l.trip_key AND c.stop_seq = l.stop_seq
    JOIN cum t ON t.trip_key = l.trip_key AND t.stop_seq = l.target_seq
    {prior_join}
    ORDER BY l.operating_day, l.trip_key, l.stop_seq, l.horizon_min
    """


def build_month(
    con: duckdb.DuckDBPyConnection,
    journeys: Path,
    labels: Path,
    out_path: Path,
    segment_stats: Path = SEGMENT_STATS_PATH,
    dwell_stats: Path = DWELL_STATS_PATH,
    baseline: Path | None = BASELINE_PREDICTIONS_PATH,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    sql = features_sql(
        f"'{_sql_path(journeys)}'", f"'{_sql_path(labels)}'", segment_stats, dwell_stats, baseline
    )
    con.execute(f"COPY ({sql}) TO '{_sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(out_path)
    return con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(out_path)}')").fetchone()[0]


def _month_of(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", path.name)
    if not m:
        raise ValueError(f"no YYYY-MM in {path.name}")
    return m.group(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("months", nargs="*", help="YYYY-MM months to build (default: all)")
    parser.add_argument("--rebuild-stats", action="store_true", help="recompute reference stats")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    journeys = sorted(config.PROCESSED.glob("journeys_*.parquet"))
    if not journeys or not BASELINE_PREDICTIONS_PATH.exists():
        raise SystemExit("Run journeys, labels and models.baselines first")
    con = duckdb.connect()
    if args.rebuild_stats or not SEGMENT_STATS_PATH.exists() or not DWELL_STATS_PATH.exists():
        log.info("Reference running / dwell times from training months (<= %s)", config.TRAIN_END)
        build_reference_stats(con, journeys)
    for name, path in (("segments", SEGMENT_STATS_PATH), ("dwell groups", DWELL_STATS_PATH)):
        n = con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(path)}')").fetchone()[0]
        log.info("%d %s with a reference time", n, name)

    wanted = set(args.months)
    for jpath in journeys:
        month = _month_of(jpath)
        if wanted and month not in wanted:
            continue
        lpath = config.PROCESSED / f"labels_{month}.parquet"
        n = build_month(con, jpath, lpath, config.PROCESSED / f"features_{month}.parquet")
        log.info("%s: %d rows", month, n)


if __name__ == "__main__":
    main()
