"""Network-state features: what is happening around a train when it leaves.

For each labelled prediction point (a train leaving station ``C`` at time ``t``, target
stop ``T``), summaries of **all trains** (any category: S-Bahn, regional and international
trains share the tracks).

Group ``network`` (30-minute window):

- at the current station ``C``, the next station and the target station: number of
  measured arrivals / departures, their mean delay and the share more than 3 min late;
- on the next segment ``C → next``: number of trains that just ran it and their mean
  delay gained on it;
- over the whole Swiss network: mean delay and share late.

Group ``network_plus``:

- **trend:** the same summaries over 10 minutes (current, next, segment, network) and
  60 minutes (current, network);
- **corridor:** every station and segment of the train's own route from ``C`` (excluded)
  to ``T``: events and delay there, delay gained by trains on those segments, and
  ``net_cor_gain_sum``, the sum over segments of their mean gain (what other trains just
  gained or lost on the way to ``T``);
- **train ahead:** the last train that left ``C`` towards the same next station: headway,
  its departure delay and, if it has already arrived, the delay it gained.

**No leakage.** An event counts only if it was measured at least ``LAG_MIN`` = 2 minutes
before ``t``. Events are bucketed by minute; only whole minutes strictly before the minute
of ``t − 2 min`` are used, so every counted event happened before ``t − 2 min``. Windows end
there. The route to ``T`` comes from the timetable. Tested in ``tests/test_network.py``.

Computation: events are counted per station (or segment) and minute. For each month, the
minutes the month needs are turned into running totals; a window sum is then
``total(end) − total(end − window)``, two ``ASOF`` lookups.

Outputs:

- ``data/processed/network_station_minutes.parquet``, ``network_segment_minutes.parquet``,
  ``network_all_minutes.parquet``: event counts, delay sums and late counts per minute;
  ``network_segment_departures.parquet``: one row per measured departure on a segment;
- ``data/processed/network_YYYY-MM.parquet``: keys + network features per labelled point
  and horizon (joined to the other features by ``swissdelay.models.tabular``).

Usage::

    uv run python -m swissdelay.features.network            # all months
    uv run python -m swissdelay.features.network 2026-05    # one month (minute tables reused)
    uv run python -m swissdelay.features.network --rebuild-minutes
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import duckdb

from swissdelay import config

log = logging.getLogger("swissdelay.network")

LAG_MIN = config.REPORTING_LAG_MIN  # 2
WINDOW_MIN = 30
SHORT_WINDOW_MIN = 10
LONG_WINDOW_MIN = 60
AHEAD_MAX_MIN = 60  # train ahead: ignored if it left more than this before the cut-off
LATE_MIN = 3.0
DELAY_MIN_MIN, DELAY_MAX_MIN = -5.0, 360.0

STATION_MINUTES = config.PROCESSED / "network_station_minutes.parquet"
SEGMENT_MINUTES = config.PROCESSED / "network_segment_minutes.parquet"
ALL_MINUTES = config.PROCESSED / "network_all_minutes.parquet"
SEGMENT_DEPARTURES = config.PROCESSED / "network_segment_departures.parquet"
MINUTE_TABLES = (STATION_MINUTES, SEGMENT_MINUTES, ALL_MINUTES, SEGMENT_DEPARTURES)

NETWORK_FEATURES = [
    "net_cur_n", "net_cur_mean_delay", "net_cur_share_late",
    "net_next_n", "net_next_mean_delay", "net_next_share_late",
    "net_tgt_n", "net_tgt_mean_delay", "net_tgt_share_late",
    "net_seg_n", "net_seg_mean_gain",
    "net_all_mean_delay", "net_all_share_late",
]  # fmt: skip
NETWORK_PLUS_FEATURES = [
    # trend
    "net_cur_n_10", "net_cur_mean_delay_10", "net_next_n_10", "net_next_mean_delay_10",
    "net_seg_n_10", "net_seg_mean_gain_10", "net_all_mean_delay_10", "net_all_share_late_10",
    "net_cur_mean_delay_60", "net_all_mean_delay_60",
    # corridor to the target
    "net_cor_n", "net_cor_mean_delay", "net_cor_share_late",
    "net_cor_seg_n", "net_cor_seg_mean_gain", "net_cor_gain_sum", "net_cor_seg_coverage",
    # train ahead on the next segment
    "net_ahead_headway_min", "net_ahead_dep_delay", "net_ahead_seg_gain",
]  # fmt: skip
ALL_NETWORK_FEATURES = NETWORK_FEATURES + NETWORK_PLUS_FEATURES
KEYS = ["trip_key", "stop_seq", "horizon_min"]


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def _list(paths: list[Path]) -> str:
    return "[" + ", ".join(f"'{_sql_path(p)}'" for p in paths) + "]"


def _minute(ts: str) -> str:
    """Integer minute index of a timestamp."""
    return f"CAST(floor(epoch({ts}) / 60) AS BIGINT)"


def _second(ts: str) -> str:
    """Integer second index of a timestamp."""
    return f"CAST(floor(epoch({ts})) AS BIGINT)"


def _valid(delay: str) -> str:
    return f"{delay} BETWEEN {DELAY_MIN_MIN} AND {DELAY_MAX_MIN}"


# --------------------------------------------------------------------------- minute tables


def events_sql(source: str) -> str:
    """Measured (REAL, valid) arrivals and departures of all trains at Swiss stations."""
    return f"""
    WITH r AS (
      SELECT operating_day::VARCHAR || '|' || trip_id AS trip_key, station_id,
             arr_planned, dep_planned, arr_actual, dep_actual,
             CASE WHEN arr_status = 'REAL'
                  THEN date_diff('second', arr_planned, arr_actual) / 60.0 END AS arr_d,
             CASE WHEN dep_status = 'REAL'
                  THEN date_diff('second', dep_planned, dep_actual) / 60.0 END AS dep_d
      FROM read_parquet({source})
      WHERE coalesce(station_id // 100000 = 85, false)
    )
    SELECT trip_key, station_id, arr_planned, dep_planned, arr_actual, dep_actual,
           CASE WHEN {_valid("arr_d")} THEN arr_d END AS arr_d,
           CASE WHEN {_valid("dep_d")} THEN dep_d END AS dep_d
    FROM r
    """


def build_minute_tables(
    con: duckdb.DuckDBPyConnection,
    interim: list[Path],
    station_path: Path = STATION_MINUTES,
    segment_path: Path = SEGMENT_MINUTES,
    all_path: Path = ALL_MINUTES,
    departures_path: Path = SEGMENT_DEPARTURES,
) -> None:
    """Event count ``n``, delay sum ``s`` and late count ``l`` per station / segment /
    whole network and minute, plus one row per measured departure on a segment."""
    station_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"CREATE OR REPLACE TEMP TABLE ev AS {events_sql(_list(interim))}")
    # one row per measured event (arrival or departure) at a station
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pt AS
        SELECT station_id, {_minute("arr_actual")} AS minute, arr_d AS d
        FROM ev WHERE arr_d IS NOT NULL
        UNION ALL
        SELECT station_id, {_minute("dep_actual")} AS minute, dep_d AS d
        FROM ev WHERE dep_d IS NOT NULL
        """
    )
    # measured departures with the next stop (its arrival, if measured)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE dep AS
        WITH o AS (
          SELECT station_id AS from_station_id, dep_actual, dep_d,
                 lead(station_id) OVER w AS to_station_id,
                 lead(arr_d) OVER w      AS next_arr_d,
                 lead(arr_actual) OVER w AS next_arr_actual
          FROM ev
          WINDOW w AS (PARTITION BY trip_key ORDER BY coalesce(arr_planned, dep_planned),
                                                      coalesce(dep_planned, arr_planned))
        )
        SELECT from_station_id, to_station_id,
               {_second("dep_actual")} AS dep_s, dep_d,
               CASE WHEN next_arr_d IS NOT NULL THEN {_second("next_arr_actual")} END AS arr_s,
               next_arr_d AS arr_d,
               CASE WHEN next_arr_d IS NOT NULL THEN {_minute("next_arr_actual")} END
                 AS arr_minute
        FROM o
        WHERE dep_d IS NOT NULL AND to_station_id IS NOT NULL
        """
    )
    late = f"count(*) FILTER (WHERE d > {LATE_MIN})"
    jobs = [
        (station_path, f"SELECT station_id, minute, count(*) AS n, sum(d) AS s, {late} AS l "
                       "FROM pt GROUP BY ALL"),
        (segment_path, "SELECT from_station_id, to_station_id, arr_minute AS minute, "
                       "count(*) AS n, sum(arr_d - dep_d) AS s "
                       "FROM dep WHERE arr_minute IS NOT NULL GROUP BY ALL"),
        (all_path, f"SELECT minute, count(*) AS n, sum(d) AS s, {late} AS l FROM pt GROUP BY ALL"),
        (departures_path, "SELECT from_station_id, to_station_id, dep_s, dep_d, arr_s, arr_d "
                          "FROM dep"),
    ]  # fmt: skip
    for path, sql in jobs:
        con.execute(
            f"COPY ({sql} ORDER BY ALL) TO '{_sql_path(path)}' (FORMAT parquet, COMPRESSION zstd)"
        )


# --------------------------------------------------------------------------- features


def _window_lookup(
    name: str,
    table: str,
    on: list[tuple[str, str]],
    cols: list[str],
    window: int = WINDOW_MIN,
    base: str = "pts",
    keys: list[str] = KEYS,
) -> str:
    """CTE ``name``: window sums (end − start) of the running totals in ``table`` for every
    row of ``base`` (which has ``m_end``)."""

    def asof(alias: str, minute: str) -> str:
        cond = "".join(f"p.{a} = {alias}.{b} AND " for a, b in on)
        return f"ASOF LEFT JOIN {table} {alias} ON {cond}{minute} >= {alias}.minute"

    pk = ", ".join(f"p.{k}" for k in keys)
    ek = ", ".join(f"e.{k}" for k in keys)
    sums = ", ".join(f"coalesce(e.{c}, 0) - coalesce(s.{c}, 0) AS {c}" for c in cols)
    return f"""
    {name}_e AS (
      SELECT {pk}, {", ".join(f"e.{c}" for c in cols)}
      FROM {base} p {asof("e", "p.m_end")}
    ),
    {name}_s AS (
      SELECT {pk}, {", ".join(f"s.{c}" for c in cols)}
      FROM {base} p {asof("s", f"p.m_end - {window}")}
    ),
    {name} AS (
      SELECT {ek}, {sums}
      FROM {name}_e e JOIN {name}_s s USING ({", ".join(keys)})
    )"""


def route_sql(journeys: Path) -> str:
    """Route of each point's train from the stop after ``C`` to the target (timetable):
    one row per stop ``k`` with its station and the previous station."""
    return f"""
    SELECT p.trip_key, p.stop_seq, p.horizon_min, j.stop_seq AS k,
           j.station_id, j.prev_station_id
    FROM pts p
    JOIN (SELECT trip_key, stop_seq, station_id,
                 lag(station_id) OVER (PARTITION BY trip_key ORDER BY stop_seq)
                   AS prev_station_id
          FROM read_parquet('{_sql_path(journeys)}')) j
      ON j.trip_key = p.trip_key AND j.stop_seq > p.stop_seq AND j.stop_seq <= p.target_seq
    """


def prepare(
    con: duckdb.DuckDBPyConnection,
    points: str,
    journeys: Path | None = None,
    station_minutes: Path = STATION_MINUTES,
    segment_minutes: Path = SEGMENT_MINUTES,
    all_minutes: Path = ALL_MINUTES,
    departures: Path = SEGMENT_DEPARTURES,
) -> None:
    """Temp tables for one batch of points.

    ``points`` is a query with columns trip_key, stop_seq, horizon_min, t_pred, target_seq,
    cur_station_id, next_station_id, target_station_id. Creates ``pts`` (points + ``m_end``,
    the last whole minute strictly before the minute of ``t − LAG_MIN``, and ``cut_s``, the
    first second after it), ``route`` (from ``journeys``; empty without), running totals
    ``cum_station`` / ``cum_segment`` / ``cum_all`` and departures ``deps`` over the minutes
    these points need only. Window sums are differences of running totals inside that
    range, so they are exact.
    """
    m_end = _minute(f"t_pred - INTERVAL {LAG_MIN} MINUTE") + " - 1"
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE pts AS
            SELECT *, {m_end} AS m_end, ({m_end} + 1) * 60 AS cut_s FROM ({points})"""
    )
    longest = max(WINDOW_MIN, SHORT_WINDOW_MIN, LONG_WINDOW_MIN, AHEAD_MAX_MIN)
    lo, hi = con.execute(f"SELECT min(m_end) - {longest} - 1, max(m_end) FROM pts").fetchone()
    if lo is None:  # no points
        lo, hi = 0, -1
    if journeys is None:
        route = (
            "SELECT trip_key, stop_seq, horizon_min, 0 AS k, NULL::INTEGER AS station_id, "
            "NULL::INTEGER AS prev_station_id FROM pts WHERE false"
        )
    else:
        route = route_sql(journeys)
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE route AS
            SELECT r.*, p.m_end FROM ({route}) r
            JOIN pts p USING (trip_key, stop_seq, horizon_min)"""
    )
    cum = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    tables = [
        ("cum_station", station_minutes, "station_id", "station_id,", True),
        ("cum_segment", segment_minutes, "from_station_id, to_station_id",
         "from_station_id, to_station_id,", False),
        ("cum_all", all_minutes, None, "", True),
    ]  # fmt: skip
    for name, path, part, keys, has_late in tables:
        over = f"PARTITION BY {part} ORDER BY minute {cum}" if part else f"ORDER BY minute {cum}"
        late = ", sum(l) OVER w AS cl" if has_late else ""
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {name} AS
            SELECT {keys} minute, sum(n) OVER w AS cn, sum(s) OVER w AS cs{late}
            FROM read_parquet('{_sql_path(path)}')
            WHERE minute BETWEEN {lo} AND {hi}
            WINDOW w AS ({over})
            """
        )
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE deps AS
            SELECT * FROM read_parquet('{_sql_path(departures)}')
            WHERE dep_s BETWEEN {lo * 60} AND {(hi + 1) * 60}"""
    )


def network_sql() -> str:
    """Network features for the points in ``pts`` (run :func:`prepare` first)."""
    st = ["cn", "cs", "cl"]
    sg = ["cn", "cs"]
    on_cur = [("cur_station_id", "station_id")]
    on_next = [("next_station_id", "station_id")]
    on_seg = [("cur_station_id", "from_station_id"), ("next_station_id", "to_station_id")]
    rk = [*KEYS, "k"]
    short, long_ = SHORT_WINDOW_MIN, LONG_WINDOW_MIN
    lookups = ",".join(
        [
            _window_lookup("cur", "cum_station", on_cur, st),
            _window_lookup("nxt", "cum_station", on_next, st),
            _window_lookup("tgt", "cum_station", [("target_station_id", "station_id")], st),
            _window_lookup("seg", "cum_segment", on_seg, sg),
            _window_lookup("allnet", "cum_all", [], st),
            _window_lookup("cur10", "cum_station", on_cur, st, short),
            _window_lookup("nxt10", "cum_station", on_next, st, short),
            _window_lookup("seg10", "cum_segment", on_seg, sg, short),
            _window_lookup("all10", "cum_all", [], st, short),
            _window_lookup("cur60", "cum_station", on_cur, st, long_),
            _window_lookup("all60", "cum_all", [], st, long_),
            _window_lookup("cst", "cum_station", [("station_id", "station_id")], st,
                           base="route", keys=rk),
            _window_lookup("csg", "cum_segment", [("prev_station_id", "from_station_id"),
                                                  ("station_id", "to_station_id")], sg,
                           base="route", keys=rk),
        ]
    )  # fmt: skip

    def stats(alias: str, prefix: str, suffix: str = "", late: bool = True) -> str:
        out = (
            f"{alias}.cn AS net_{prefix}_n{suffix}, "
            f"{alias}.cs / nullif({alias}.cn, 0) AS net_{prefix}_mean_delay{suffix}"
        )
        if late:
            out += f", {alias}.cl / nullif({alias}.cn, 0) AS net_{prefix}_share_late{suffix}"
        return out

    def mean(alias: str, col: str, what: str = "cs") -> str:
        return f"{alias}.{what} / nullif({alias}.cn, 0) AS {col}"

    recent = f"d.dep_s > p.cut_s - {AHEAD_MAX_MIN * 60}"
    return f"""
    WITH {lookups},
    cor AS (
      SELECT trip_key, stop_seq, horizon_min, sum(cn) AS cn, sum(cs) AS cs, sum(cl) AS cl
      FROM cst GROUP BY ALL
    ),
    corseg AS (
      SELECT trip_key, stop_seq, horizon_min, sum(cn) AS cn, sum(cs) AS cs,
             sum(cs / nullif(cn, 0)) AS gain_sum, avg((cn > 0)::DOUBLE) AS coverage
      FROM csg GROUP BY ALL
    ),
    ahead AS (
      SELECT p.trip_key, p.stop_seq, p.horizon_min,
             CASE WHEN {recent} THEN (epoch(p.t_pred) - d.dep_s) / 60.0 END AS headway,
             CASE WHEN {recent} THEN d.dep_d END AS dep_d,
             CASE WHEN {recent} AND d.arr_s < p.cut_s THEN d.arr_d - d.dep_d END AS gain
      FROM pts p ASOF LEFT JOIN deps d
        ON p.cur_station_id = d.from_station_id AND p.next_station_id = d.to_station_id
       AND p.cut_s > d.dep_s
    )
    SELECT p.trip_key, p.stop_seq, p.horizon_min,
           {stats("cur", "cur")},
           {stats("nxt", "next")},
           {stats("tgt", "tgt")},
           seg.cn AS net_seg_n, {mean("seg", "net_seg_mean_gain")},
           {mean("allnet", "net_all_mean_delay")},
           {mean("allnet", "net_all_share_late", "cl")},
           {stats("cur10", "cur", "_10", late=False)},
           {stats("nxt10", "next", "_10", late=False)},
           seg10.cn AS net_seg_n_10, {mean("seg10", "net_seg_mean_gain_10")},
           {mean("all10", "net_all_mean_delay_10")},
           {mean("all10", "net_all_share_late_10", "cl")},
           {mean("cur60", "net_cur_mean_delay_60")},
           {mean("all60", "net_all_mean_delay_60")},
           coalesce(cor.cn, 0) AS net_cor_n, {mean("cor", "net_cor_mean_delay")},
           {mean("cor", "net_cor_share_late", "cl")},
           coalesce(corseg.cn, 0) AS net_cor_seg_n, {mean("corseg", "net_cor_seg_mean_gain")},
           corseg.gain_sum AS net_cor_gain_sum, corseg.coverage AS net_cor_seg_coverage,
           ahead.headway AS net_ahead_headway_min, ahead.dep_d AS net_ahead_dep_delay,
           ahead.gain AS net_ahead_seg_gain
    FROM pts p
    JOIN cur    USING (trip_key, stop_seq, horizon_min)
    JOIN nxt    USING (trip_key, stop_seq, horizon_min)
    JOIN tgt    USING (trip_key, stop_seq, horizon_min)
    JOIN seg    USING (trip_key, stop_seq, horizon_min)
    JOIN allnet USING (trip_key, stop_seq, horizon_min)
    JOIN cur10  USING (trip_key, stop_seq, horizon_min)
    JOIN nxt10  USING (trip_key, stop_seq, horizon_min)
    JOIN seg10  USING (trip_key, stop_seq, horizon_min)
    JOIN all10  USING (trip_key, stop_seq, horizon_min)
    JOIN cur60  USING (trip_key, stop_seq, horizon_min)
    JOIN all60  USING (trip_key, stop_seq, horizon_min)
    JOIN ahead  USING (trip_key, stop_seq, horizon_min)
    LEFT JOIN cor    USING (trip_key, stop_seq, horizon_min)
    LEFT JOIN corseg USING (trip_key, stop_seq, horizon_min)
    ORDER BY p.trip_key, p.stop_seq, p.horizon_min
    """


def points_sql(journeys: Path, labels: Path) -> str:
    """Labelled points with prediction time, current, next and target stations."""
    return f"""
    SELECT l.trip_key, l.stop_seq, l.horizon_min, l.t_pred, l.target_seq,
           l.station_id AS cur_station_id, n.next_station_id, l.target_station_id
    FROM read_parquet('{_sql_path(labels)}') l
    JOIN (SELECT trip_key, stop_seq,
                 lead(station_id) OVER (PARTITION BY trip_key ORDER BY stop_seq) AS next_station_id
          FROM read_parquet('{_sql_path(journeys)}')) n
      USING (trip_key, stop_seq)
    WHERE l.label_status = 'ok'
    """


def build_month(con: duckdb.DuckDBPyConnection, journeys: Path, labels: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    prepare(con, points_sql(journeys, labels), journeys)
    con.execute(f"COPY ({network_sql()}) TO '{_sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(out)
    return con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(out)}')").fetchone()[0]


def _month_of(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", path.name)
    if not m:
        raise ValueError(f"no YYYY-MM in {path.name}")
    return m.group(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("months", nargs="*", help="YYYY-MM months to build (default: all)")
    parser.add_argument("--rebuild-minutes", action="store_true", help="recompute minute tables")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    interim = sorted(config.INTERIM.glob("trains_*.parquet"))
    if not interim:
        raise SystemExit("No data/interim/trains_*.parquet")
    con = duckdb.connect()
    if args.rebuild_minutes or not all(p.exists() for p in MINUTE_TABLES):
        log.info("Building minute tables from %d months of all trains", len(interim))
        build_minute_tables(con, interim)
    for path in MINUTE_TABLES:
        n = con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(path)}')").fetchone()[0]
        log.info("%s: %d rows", path.name, n)

    wanted = set(args.months)
    for jpath in sorted(config.PROCESSED.glob("journeys_*.parquet")):
        month = _month_of(jpath)
        if wanted and month not in wanted:
            continue
        lpath = config.PROCESSED / f"labels_{month}.parquet"
        n = build_month(con, jpath, lpath, config.PROCESSED / f"network_{month}.parquet")
        log.info("%s: %d rows", month, n)


if __name__ == "__main__":
    main()
