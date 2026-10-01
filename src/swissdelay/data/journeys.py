"""Rebuild in-scope train journeys from the ingested actual data.

Applies the decisions of ``notebooks/01-data-quality.ipynb`` (section 9):

1. Scope: ``category`` in IC / IR / RE / EC.
2. Swiss-numbered stops only; cross-border runs keep their Swiss section and are
   flagged ``enters_from_abroad``; foreign-only runs disappear.
3. Every row is kept, but only ``REAL`` times count as observed.
4. Operators FART, DB and DB Regio are not eligible as prediction points / targets.
5. A station is eligible if >= 80 % of its in-scope departures are ``REAL`` over the
   training months (computed on training data only, so no leakage).
6. A ``REAL`` time giving a delay outside [-5 min, +6 h] is treated as missing.
7. Pass-through rows are not eligible (kept as network observations).
8. Negative measured dwell / running times are kept as they are.
9. Stops are ordered by planned time; all times are full timestamps.
10. Anomalous journeys are flagged ``is_valid_journey = false`` and are not eligible.
    A stop whose planned departure is before its planned arrival (engineering-works
    timetables) is flagged ``is_plan_inconsistent`` and only that stop is not eligible.
    Journeys with a single Swiss stop are valid but have nothing to predict.
11. Cancelled stops are not eligible.

Outputs (one file per month, a journey never spans two files because files are split
by ``operating_day``):

- ``data/processed/station_quality.parquet``: REAL share per station, training months.
- ``data/processed/journeys_YYYY-MM.parquet``: one row per Swiss stop of an in-scope run.

Usage::

    uv run python -m swissdelay.data.journeys              # all ingested months
    uv run python -m swissdelay.data.journeys 2025-08      # one month (station quality reused)
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import duckdb

from swissdelay import config

log = logging.getLogger("swissdelay.journeys")

SCOPE_CATEGORIES = ("IC", "IR", "RE", "EC")
EXCLUDED_OPERATORS = ("FART", "DB", "DB Regio")
MIN_STATION_REAL_SHARE = 0.8
DELAY_MIN_MIN, DELAY_MAX_MIN = -5.0, 360.0
SWISS_PREFIX = 85  # station_id // 100000

STATION_QUALITY_PATH = config.PROCESSED / "station_quality.parquet"

# Column order of the output files
OUTPUT_COLUMNS = [
    "operating_day", "trip_key", "trip_id", "operator_abbr", "category", "line_name",
    "train_number", "stop_seq", "n_stops", "station_id", "station_name",
    "arr_planned", "arr_actual", "arr_status", "arr_delay_min",
    "dep_planned", "dep_actual", "dep_status", "dep_delay_min",
    "is_cancelled", "is_extra_trip", "is_pass_through",
    "enters_from_abroad", "is_valid_journey", "is_target_operator", "is_valid_station",
    "is_plan_inconsistent",
    "is_eligible_stop", "is_prediction_point",
]  # fmt: skip


def _sql_list(values: tuple[str, ...]) -> str:
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + ")"


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


SCOPE_SQL = f"category IN {_sql_list(SCOPE_CATEGORIES)}"
SWISS_SQL = f"coalesce(station_id // 100000 = {SWISS_PREFIX}, false)"


def _valid_delay_sql(status: str, planned: str, actual: str) -> str:
    """Delay in minutes if the time is REAL and within the validity range, else NULL."""
    d = f"date_diff('second', {planned}, {actual}) / 60.0"
    return (
        f"CASE WHEN {status} = 'REAL' AND {d} BETWEEN {DELAY_MIN_MIN} AND {DELAY_MAX_MIN} "
        f"THEN {d} END"
    )


def station_quality_sql(source: str, train_end: str) -> str:
    """REAL share of in-scope departures per Swiss-numbered station, training months only."""
    return f"""
    SELECT station_id,
           mode(station_name)                AS station_name,
           count(*)                          AS dep_events,
           avg((dep_status = 'REAL')::INT)   AS share_real,
           avg((dep_status = 'REAL')::INT) >= {MIN_STATION_REAL_SHARE} AS is_valid_station
    FROM read_parquet({source})
    WHERE {SCOPE_SQL} AND {SWISS_SQL} AND dep_planned IS NOT NULL
      AND operating_day <= DATE '{train_end}'
    GROUP BY station_id
    ORDER BY dep_events DESC
    """


def journeys_sql(source: str, stations: str) -> str:
    """One row per Swiss stop of an in-scope run, with eligibility flags."""
    order = "coalesce(arr_planned, dep_planned), coalesce(dep_planned, arr_planned), station_id"
    return f"""
    WITH base AS (
      SELECT *,
             operating_day::VARCHAR || '|' || trip_id AS trip_key,
             {SWISS_SQL}                              AS is_swiss
      FROM read_parquet({source})
      WHERE {SCOPE_SQL}
    ),
    ordered AS (
      SELECT *,
             row_number() OVER w             AS full_seq,
             count(*) OVER (PARTITION BY trip_key) AS full_n,
             lead(arr_planned) OVER w        AS next_arr_planned,
             lead(station_id)  OVER w        AS next_station_id
      FROM base
      WINDOW w AS (PARTITION BY trip_key ORDER BY {order})
    ),
    trip AS (
      SELECT trip_key,
             count(*)                                               AS n_all,
             count(DISTINCT station_id)                             AS n_stations,
             bool_or(full_seq = 1 AND arr_planned IS NOT NULL)      AS first_has_arr,
             bool_or(full_seq = full_n AND dep_planned IS NOT NULL) AS last_has_dep,
             coalesce(bool_or(dep_planned > next_arr_planned), false) AS order_violation,
             coalesce(bool_or(station_id = next_station_id), false)   AS consecutive_duplicate,
             count(*) FILTER (WHERE is_swiss)                       AS n_swiss,
             min(coalesce(arr_planned, dep_planned)) FILTER (WHERE NOT is_swiss) AS first_foreign_t,
             min(coalesce(arr_planned, dep_planned)) FILTER (WHERE is_swiss)     AS first_swiss_t,
             mode(operator_abbr)                                    AS trip_operator
      FROM ordered
      GROUP BY trip_key
    ),
    swiss AS (
      SELECT o.*,
             coalesce(t.first_foreign_t < t.first_swiss_t, false) AS enters_from_abroad,
             NOT (t.n_stations < t.n_all OR t.first_has_arr OR t.last_has_dep
                  OR t.order_violation OR t.consecutive_duplicate) AS is_valid_journey,
             coalesce(o.arr_planned > o.dep_planned, false)          AS is_plan_inconsistent,
             t.n_swiss                                              AS n_swiss,
             coalesce(t.trip_operator NOT IN {_sql_list(EXCLUDED_OPERATORS)}, true)
                                                                    AS is_target_operator,
             coalesce(s.is_valid_station, false)                    AS is_valid_station
      FROM ordered o
      JOIN trip t USING (trip_key)
      LEFT JOIN read_parquet('{_sql_path(stations)}') s USING (station_id)
      WHERE o.is_swiss
    ),
    final AS (
      SELECT *,
             row_number() OVER (PARTITION BY trip_key ORDER BY full_seq) AS stop_seq,
             count(*) OVER (PARTITION BY trip_key)                       AS n_stops,
             {_valid_delay_sql("arr_status", "arr_planned", "arr_actual")} AS arr_delay_min,
             {_valid_delay_sql("dep_status", "dep_planned", "dep_actual")} AS dep_delay_min,
             is_valid_journey AND is_target_operator AND is_valid_station
               AND n_swiss >= 2 AND NOT is_plan_inconsistent
               AND NOT coalesce(is_pass_through, false)
               AND NOT coalesce(is_cancelled, false)                     AS is_eligible_stop
      FROM swiss
    )
    SELECT {", ".join(OUTPUT_COLUMNS[:-1])},
           is_eligible_stop AND dep_delay_min IS NOT NULL AS is_prediction_point
    FROM final
    ORDER BY operating_day, trip_key, stop_seq
    """


def build_station_quality(
    con: duckdb.DuckDBPyConnection,
    sources: list[Path],
    out_path: Path = STATION_QUALITY_PATH,
    train_end: str = config.TRAIN_END,
) -> None:
    source = "[" + ", ".join(f"'{_sql_path(p)}'" for p in sources) + "]"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({station_quality_sql(source, train_end)}) "
        f"TO '{_sql_path(out_path)}' (FORMAT parquet, COMPRESSION zstd)"
    )


def build_month(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    out_path: Path,
    stations_path: Path = STATION_QUALITY_PATH,
) -> dict:
    """Build one month of journeys and return summary counts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    sql = journeys_sql(f"'{_sql_path(source)}'", stations_path)
    con.execute(f"COPY ({sql}) TO '{_sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(out_path)
    row = con.execute(
        f"""
        SELECT count(*), count(DISTINCT trip_key),
               count(*) FILTER (WHERE is_eligible_stop),
               count(*) FILTER (WHERE is_prediction_point),
               count(DISTINCT trip_key) FILTER (WHERE NOT is_valid_journey)
        FROM read_parquet('{_sql_path(out_path)}')
        """
    ).fetchone()
    keys = ["rows", "journeys", "eligible_stops", "prediction_points", "invalid_journeys"]
    return dict(zip(keys, row, strict=True))


def _month_of(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", path.name)
    if not m:
        raise ValueError(f"no YYYY-MM in {path.name}")
    return m.group(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("months", nargs="*", help="YYYY-MM months to build (default: all)")
    parser.add_argument(
        "--rebuild-stations", action="store_true", help="recompute station_quality.parquet"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sources = sorted(config.INTERIM.glob("trains_*.parquet"))
    if not sources:
        raise SystemExit("No data/interim/trains_*.parquet: run swissdelay.data.ingest first")
    con = duckdb.connect()

    if args.rebuild_stations or not STATION_QUALITY_PATH.exists():
        log.info("Station quality from training months (<= %s)", config.TRAIN_END)
        build_station_quality(con, sources)
    n_valid, n_all = con.execute(
        f"SELECT count(*) FILTER (WHERE is_valid_station), count(*) "
        f"FROM read_parquet('{_sql_path(STATION_QUALITY_PATH)}')"
    ).fetchone()
    log.info(
        "%d / %d stations valid (>= %.0f %% REAL)", n_valid, n_all, 100 * MIN_STATION_REAL_SHARE
    )

    wanted = set(args.months)
    for src in sources:
        month = _month_of(src)
        if wanted and month not in wanted:
            continue
        stats = build_month(con, src, config.PROCESSED / f"journeys_{month}.parquet")
        log.info(
            "%s: %d journeys, %d stops, %d eligible, %d prediction points, %d invalid journeys",
            month,
            stats["journeys"],
            stats["rows"],
            stats["eligible_stops"],
            stats["prediction_points"],
            stats["invalid_journeys"],
        )


if __name__ == "__main__":
    main()
