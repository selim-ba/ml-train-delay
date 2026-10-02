"""Horizon labels: the change in delay 15, 30 and 60 minutes ahead.

For every prediction point of ``data/processed/journeys_YYYY-MM.parquet``
(a measured departure of an eligible stop, delay ``d0`` at time ``t = dep_actual``)
and every horizon ``h``:

- **Target stop:** the first downstream stop of the same journey whose *planned* arrival
  is at least ``h`` after the *planned* departure. It is chosen from the timetable only,
  so it never depends on how the train actually ran. Stops that can never be targets
  are skipped, all known in advance: pass-through rows, stations that are not valid
  (< 80 % REAL over the training months) and stops with inconsistent planned times.
  Cancelled stops are *not* skipped: a cancellation is only known when it happens.
- **Target:** ``delta_min = target_delay_min - d0``, where ``target_delay_min`` is the
  valid REAL arrival delay at the target stop. Persistence is ``delta_min = 0``.
- **No label**, counted by reason in ``label_status``:
  ``terminates`` (no target stop at least ``h`` ahead), ``cancelled`` (target stop
  cancelled), ``not_measured`` (target arrival not REAL or invalid).
- ``in_common_subset``: the point is labelled at all three horizons.

Output: ``data/processed/labels_YYYY-MM.parquet``, one row per prediction point and
horizon (long format).

Usage::

    uv run python -m swissdelay.features.labels            # all months
    uv run python -m swissdelay.features.labels 2025-08    # one month
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import duckdb

from swissdelay import config

log = logging.getLogger("swissdelay.labels")

HORIZONS = tuple(config.HORIZONS_MIN)  # (15, 30, 60)
LABEL_STATUSES = ("ok", "terminates", "cancelled", "not_measured")

OUTPUT_COLUMNS = [
    "operating_day", "trip_key", "stop_seq", "station_id", "category", "line_name",
    "operator_abbr", "train_number", "dep_planned", "t_pred", "d0_min", "horizon_min",
    "target_seq", "target_station_id", "target_arr_planned", "sched_gap_min",
    "target_delay_min", "delta_min", "label_status", "in_common_subset",
]  # fmt: skip


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def labels_sql(source: str, horizons: tuple[int, ...] = HORIZONS) -> str:
    """One row per prediction point and horizon, with the target stop and its label."""
    h_list = ", ".join(str(int(h)) for h in horizons)
    return f"""
    WITH j AS (
      SELECT * FROM read_parquet({source})
    ),
    pp AS (
      SELECT operating_day, trip_key, stop_seq, station_id, category, line_name,
             operator_abbr, train_number, dep_planned,
             dep_actual    AS t_pred,
             dep_delay_min AS d0_min
      FROM j
      WHERE is_prediction_point
    ),
    -- Candidate target stops: fixed in advance, never from actual times
    cand AS (
      SELECT trip_key, stop_seq AS target_seq, station_id AS target_station_id,
             arr_planned AS target_arr_planned, arr_delay_min AS target_delay_min,
             coalesce(is_cancelled, false) AS target_cancelled
      FROM j
      WHERE arr_planned IS NOT NULL
        AND NOT coalesce(is_pass_through, false)
        AND is_valid_station
        AND NOT is_plan_inconsistent
    ),
    hz AS (
      SELECT unnest([{h_list}]) AS horizon_min
    ),
    first_target AS (
      SELECT p.trip_key, p.stop_seq, hz.horizon_min, min(c.target_seq) AS target_seq
      FROM pp p
      CROSS JOIN hz
      JOIN cand c
        ON c.trip_key = p.trip_key
       AND c.target_seq > p.stop_seq
       AND c.target_arr_planned >= p.dep_planned + to_minutes(CAST(hz.horizon_min AS BIGINT))
      GROUP BY ALL
    ),
    labelled AS (
      SELECT p.*, hz.horizon_min,
             c.target_seq, c.target_station_id, c.target_arr_planned,
             date_diff('second', p.dep_planned, c.target_arr_planned) / 60.0 AS sched_gap_min,
             c.target_delay_min,
             CASE WHEN c.target_seq IS NULL     THEN 'terminates'
                  WHEN c.target_cancelled       THEN 'cancelled'
                  WHEN c.target_delay_min IS NULL THEN 'not_measured'
                  ELSE 'ok' END AS label_status
      FROM pp p
      CROSS JOIN hz
      LEFT JOIN first_target f
        ON f.trip_key = p.trip_key AND f.stop_seq = p.stop_seq
       AND f.horizon_min = hz.horizon_min
      LEFT JOIN cand c
        ON c.trip_key = f.trip_key AND c.target_seq = f.target_seq
    )
    SELECT operating_day, trip_key, stop_seq, station_id, category, line_name,
           operator_abbr, train_number, dep_planned, t_pred, d0_min, horizon_min,
           target_seq, target_station_id, target_arr_planned, sched_gap_min,
           target_delay_min,
           CASE WHEN label_status = 'ok' THEN target_delay_min - d0_min END AS delta_min,
           label_status,
           bool_and(label_status = 'ok') OVER (PARTITION BY trip_key, stop_seq)
             AS in_common_subset
    FROM labelled
    ORDER BY operating_day, trip_key, stop_seq, horizon_min
    """


def build_month(con: duckdb.DuckDBPyConnection, source: Path, out_path: Path) -> list[tuple]:
    """Build the labels of one month; return (horizon, status, n) counts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    sql = labels_sql(f"'{_sql_path(source)}'")
    con.execute(f"COPY ({sql}) TO '{_sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(out_path)
    return con.execute(
        f"""
        SELECT horizon_min, label_status, count(*) AS n
        FROM read_parquet('{_sql_path(out_path)}')
        GROUP BY ALL ORDER BY horizon_min, label_status
        """
    ).fetchall()


def _month_of(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", path.name)
    if not m:
        raise ValueError(f"no YYYY-MM in {path.name}")
    return m.group(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("months", nargs="*", help="YYYY-MM months to build (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sources = sorted(config.PROCESSED.glob("journeys_*.parquet"))
    if not sources:
        raise SystemExit("No data/processed/journeys_*.parquet: run swissdelay.data.journeys first")
    con = duckdb.connect()
    wanted = set(args.months)
    for src in sources:
        month = _month_of(src)
        if wanted and month not in wanted:
            continue
        counts = build_month(con, src, config.PROCESSED / f"labels_{month}.parquet")
        per_h: dict[int, dict[str, int]] = {}
        for h, status, n in counts:
            per_h.setdefault(h, {})[status] = n
        parts = []
        for h, c in sorted(per_h.items()):
            total = sum(c.values())
            parts.append(f"{h} min: {c.get('ok', 0) / total:.1%} labelled")
        log.info("%s: %s", month, ", ".join(parts))


if __name__ == "__main__":
    main()
