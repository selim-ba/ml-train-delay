"""Per-point subgraphs for the graph transformer, one horizon at a time.

**Static station graph** (timetable of the training months only): a directed edge
``a → b`` when at least ``MIN_EDGE_RUNS`` planned runs go from ``a`` to the next stop ``b``,
with its median planned running time. Each station gets an index (``1 … V``; 0 = unknown
or padding) and its planned stops per day (traffic).

**Subgraph of a prediction point** (train leaving stop ``C`` at ``t``, target stop ``T``):

- route nodes: every stop of the train from ``C`` to ``T`` (timetable), roles
  ``current`` / ``route`` / ``target``;
- 1-hop neighbours: stations linked to a route node in the static graph, most-served
  first, up to ``N_MAX`` nodes in total;
- edges: static edges between these nodes, route edges first, up to ``E_MAX``.

**Features** (same leakage rule as ``swissdelay.features.network``: only events measured
before the minute of ``t − 2 min``):

- node: events, mean delay and share > 3 min late over 10 / 30 / 60 min; role; position
  along the route; planned minutes from ``t`` to that stop; log planned traffic;
- edge: log planned runs, planned running time, on-route flag; trains that ran it and
  their mean delay gain over 10 / 30 min.

Outputs in ``data/processed/graph/h{H}/YYYY-MM/`` (numpy arrays padded per point):
``points.parquet`` (keys, ``delta_min``, row index ``pidx``), ``node_x`` (P, N_MAX, F_node,
float16), ``node_station`` (P, N_MAX, int32), ``node_n`` (P), ``edge_x`` (P, E_MAX, F_edge,
float16), ``edge_index`` (P, E_MAX, 2: source and target node position, −1 = padding),
``edge_n`` (P).

Usage::

    uv run python -m swissdelay.features.graph                  # horizon 15, all months
    uv run python -m swissdelay.features.graph --horizon 15 2026-05
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from swissdelay import config
from swissdelay.features import network as nw

log = logging.getLogger("swissdelay.graph")

N_MAX = 32
E_MAX = 96
MIN_EDGE_RUNS = 50
NODE_WINDOWS = (10, 30, 60)
EDGE_WINDOWS = (10, 30)
ROLES = ("current", "route", "target", "neighbour")

NODE_FEATURES = [
    *(f"{s}_{w}" for w in NODE_WINDOWS for s in ("log_n", "mean_delay", "share_late")),
    *(f"is_{r}" for r in ROLES),
    "route_offset", "planned_min", "log_traffic",
]  # fmt: skip
EDGE_FEATURES = [
    "log_runs", "run_planned_min", "on_route",
    *(f"{s}_{w}" for w in EDGE_WINDOWS for s in ("log_n", "mean_gain")),
]  # fmt: skip
NODE_KEYS = [*nw.KEYS, "pos"]
EDGE_KEYS = [*nw.KEYS, "eid"]

GRAPH_DIR = config.PROCESSED / "graph"
STATIONS_PATH = GRAPH_DIR / "stations.parquet"
EDGES_PATH = GRAPH_DIR / "edges.parquet"

_p = nw._sql_path


# --------------------------------------------------------------------------- static graph


def build_static_graph(
    con: duckdb.DuckDBPyConnection,
    interim: list[Path],
    stations_path: Path = STATIONS_PATH,
    edges_path: Path = EDGES_PATH,
    end: str = config.TRAIN_END,
    min_runs: int = MIN_EDGE_RUNS,
) -> tuple[int, int]:
    """Station index and directed edges from the planned runs up to ``end``."""
    stations_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE plan AS
        SELECT operating_day::VARCHAR || '|' || trip_id AS trip_key, operating_day, station_id,
               arr_planned, dep_planned
        FROM read_parquet({nw._list(interim)})
        WHERE coalesce(station_id // 100000 = 85, false) AND operating_day <= DATE '{end}'
        """
    )
    edges = f"""
        WITH o AS (
          SELECT station_id AS from_station_id, dep_planned,
                 lead(station_id) OVER w  AS to_station_id,
                 lead(arr_planned) OVER w AS next_arr_planned
          FROM plan
          WINDOW w AS (PARTITION BY trip_key ORDER BY coalesce(arr_planned, dep_planned),
                                                      coalesce(dep_planned, arr_planned))
        )
        SELECT from_station_id, to_station_id, count(*) AS n_runs,
               median(date_diff('second', dep_planned, next_arr_planned) / 60.0)
                 AS run_planned_min
        FROM o
        WHERE to_station_id IS NOT NULL AND from_station_id <> to_station_id
          AND dep_planned IS NOT NULL AND next_arr_planned IS NOT NULL
        GROUP BY ALL
        HAVING count(*) >= {min_runs}
        ORDER BY ALL
    """
    stations = """
        SELECT station_id, row_number() OVER (ORDER BY station_id) AS idx,
               count(*) / count(DISTINCT operating_day) AS daily_stops
        FROM plan GROUP BY station_id ORDER BY station_id
    """
    con.execute(f"COPY ({edges}) TO '{_p(edges_path)}' (FORMAT parquet)")
    con.execute(f"COPY ({stations}) TO '{_p(stations_path)}' (FORMAT parquet)")
    n_st = con.execute(f"SELECT count(*) FROM read_parquet('{_p(stations_path)}')").fetchone()[0]
    n_ed = con.execute(f"SELECT count(*) FROM read_parquet('{_p(edges_path)}')").fetchone()[0]
    return n_st, n_ed


# --------------------------------------------------------------------------- subgraphs


def points_sql(journeys: Path, labels: Path, horizon: int) -> str:
    """Labelled points of one horizon, with their target ``delta_min``."""
    return f"""
    SELECT p.*, l.delta_min
    FROM ({nw.points_sql(journeys, labels)}) p
    JOIN read_parquet('{_p(labels)}') l USING (trip_key, stop_seq, horizon_min)
    WHERE p.horizon_min = {horizon}
    """


def _select_nodes(con: duckdb.DuckDBPyConnection, journeys: Path, stations: Path) -> int:
    """Temp table ``gnodes``: route nodes then neighbours, ``pos`` 0 … N_MAX − 1.
    Returns the number of nodes dropped by the ``N_MAX`` cap."""
    keys = ", ".join(nw.KEYS)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE route_nodes AS
        SELECT p.trip_key, p.stop_seq, p.horizon_min, p.target_seq, j.station_id,
               min(j.stop_seq) AS k,
               min(epoch(coalesce(j.arr_planned, j.dep_planned)) - epoch(p.t_pred)) / 60.0
                 AS planned_min
        FROM pts p
        JOIN read_parquet('{_p(journeys)}') j
          ON j.trip_key = p.trip_key AND j.stop_seq BETWEEN p.stop_seq AND p.target_seq
        GROUP BY ALL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE all_nodes AS
        WITH nb AS (
          SELECT DISTINCT r.trip_key, r.stop_seq, r.horizon_min, a.b AS station_id
          FROM route_nodes r JOIN adj a ON a.a = r.station_id
        ),
        nb_only AS (
          SELECT * FROM nb ANTI JOIN route_nodes USING ({keys}, station_id)
        ),
        u AS (
          SELECT trip_key, stop_seq, horizon_min, station_id, k, planned_min,
                 CASE WHEN k = stop_seq THEN 'current'
                      WHEN k = target_seq THEN 'target' ELSE 'route' END AS role,
                 0 AS grp
          FROM route_nodes
          UNION ALL
          SELECT trip_key, stop_seq, horizon_min, station_id, NULL::BIGINT, NULL::DOUBLE,
                 'neighbour', 1
          FROM nb_only
        )
        SELECT u.*, coalesce(s.idx, 0) AS station_idx, coalesce(s.daily_stops, 0) AS daily_stops,
               row_number() OVER (PARTITION BY {keys}
                                  ORDER BY grp, k, s.daily_stops DESC NULLS LAST, station_id)
                 - 1 AS pos
        FROM u LEFT JOIN read_parquet('{_p(stations)}') s USING (station_id)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gnodes AS
        SELECT n.*, p.m_end FROM all_nodes n JOIN pts p USING ({keys}) WHERE n.pos < {N_MAX}
        """
    )
    return con.execute(f"SELECT count(*) FROM all_nodes WHERE pos >= {N_MAX}").fetchone()[0]


def _select_edges(con: duckdb.DuckDBPyConnection, edges: Path) -> int:
    """Temp table ``gedges``: static edges between the nodes of each subgraph, route edges
    first, ``eid`` 0 … E_MAX − 1. Returns the number of edges dropped by the cap."""
    keys = ", ".join(nw.KEYS)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE all_edges AS
        WITH e AS (
          SELECT s.trip_key, s.stop_seq, s.horizon_min, s.pos AS src, d.pos AS dst,
                 s.station_id AS from_station_id, d.station_id AS to_station_id,
                 x.n_runs, x.run_planned_min,
                 coalesce(s.k IS NOT NULL AND d.k = s.k + 1, false) AS on_route, s.m_end
          FROM gnodes s
          JOIN read_parquet('{_p(edges)}') x ON x.from_station_id = s.station_id
          JOIN gnodes d
            ON d.trip_key = s.trip_key AND d.stop_seq = s.stop_seq
           AND d.horizon_min = s.horizon_min AND d.station_id = x.to_station_id
        )
        SELECT *, row_number() OVER (PARTITION BY {keys}
                                     ORDER BY on_route DESC, n_runs DESC, src, dst) - 1 AS eid
        FROM e
        """
    )
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE gedges AS SELECT * FROM all_edges WHERE eid < {E_MAX}"
    )
    return con.execute(f"SELECT count(*) FROM all_edges WHERE eid >= {E_MAX}").fetchone()[0]


def _node_sql() -> str:
    lookups = ",".join(
        nw._window_lookup(f"n{w}", "cum_station", [("station_id", "station_id")],
                          ["cn", "cs", "cl"], w, base="gnodes", keys=NODE_KEYS)
        for w in NODE_WINDOWS
    )  # fmt: skip
    cols = []
    for w in NODE_WINDOWS:
        cols += [
            f"ln(1 + n{w}.cn) AS log_n_{w}",
            f"coalesce(n{w}.cs / nullif(n{w}.cn, 0), 0) AS mean_delay_{w}",
            f"coalesce(n{w}.cl / nullif(n{w}.cn, 0), 0) AS share_late_{w}",
        ]
    cols += [f"(g.role = '{r}')::DOUBLE AS is_{r}" for r in ROLES]
    cols += [
        "coalesce(g.k - g.stop_seq, 0)::DOUBLE AS route_offset",
        "coalesce(g.planned_min, 0) AS planned_min",
        "ln(1 + g.daily_stops) AS log_traffic",
    ]
    joins = "".join(f" JOIN n{w} USING ({', '.join(NODE_KEYS)})" for w in NODE_WINDOWS)
    return f"""
    WITH {lookups}
    SELECT gp.pidx, g.pos, g.station_idx, {", ".join(cols)}
    FROM gnodes g JOIN gpoints gp USING ({", ".join(nw.KEYS)}){joins}
    """


def _edge_sql() -> str:
    on = [("from_station_id", "from_station_id"), ("to_station_id", "to_station_id")]
    lookups = ",".join(
        nw._window_lookup(f"e{w}", "cum_segment", on, ["cn", "cs"], w, base="gedges",
                          keys=EDGE_KEYS)
        for w in EDGE_WINDOWS
    )  # fmt: skip
    cols = ["ln(1 + g.n_runs) AS log_runs", "g.run_planned_min", "g.on_route::DOUBLE AS on_route"]
    for w in EDGE_WINDOWS:
        cols += [
            f"ln(1 + e{w}.cn) AS log_n_{w}",
            f"coalesce(e{w}.cs / nullif(e{w}.cn, 0), 0) AS mean_gain_{w}",
        ]
    joins = "".join(f" JOIN e{w} USING ({', '.join(EDGE_KEYS)})" for w in EDGE_WINDOWS)
    return f"""
    WITH {lookups}
    SELECT gp.pidx, g.eid, g.src, g.dst, {", ".join(cols)}
    FROM gedges g JOIN gpoints gp USING ({", ".join(nw.KEYS)}){joins}
    """


def _pack(df: pd.DataFrame, slot: str, size: int, feats: list[str], n_points: int):
    p, q = df["pidx"].to_numpy(), df[slot].to_numpy()
    x = np.zeros((n_points, size, len(feats)), np.float16)
    x[p, q] = df[feats].to_numpy(np.float32)
    n = np.bincount(p, minlength=n_points).astype(np.int16)
    return x, p, q, n


def extract(
    con: duckdb.DuckDBPyConnection,
    points: str,
    journeys: Path,
    stations: Path = STATIONS_PATH,
    edges: Path = EDGES_PATH,
    minutes: tuple[Path, ...] = nw.MINUTE_TABLES,
) -> dict:
    """Padded node and edge arrays for the points of query ``points`` (columns of
    :func:`points_sql`)."""
    nw.prepare(con, points, journeys, *minutes)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gpoints AS
        SELECT trip_key, stop_seq, horizon_min, delta_min,
               row_number() OVER (ORDER BY trip_key, stop_seq, horizon_min) - 1 AS pidx
        FROM pts
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE adj AS
        SELECT from_station_id AS a, to_station_id AS b FROM read_parquet('{_p(edges)}')
        UNION
        SELECT to_station_id AS a, from_station_id AS b FROM read_parquet('{_p(edges)}')
        """
    )
    dropped_nodes = _select_nodes(con, journeys, stations)
    dropped_edges = _select_edges(con, edges)

    pts = con.sql("SELECT * FROM gpoints ORDER BY pidx").df()
    n_points = len(pts)
    nodes = con.sql(_node_sql()).df()
    node_x, p, q, node_n = _pack(nodes, "pos", N_MAX, NODE_FEATURES, n_points)
    node_station = np.zeros((n_points, N_MAX), np.int32)
    node_station[p, q] = nodes["station_idx"].to_numpy()

    ed = con.sql(_edge_sql()).df()
    edge_x, p, q, edge_n = _pack(ed, "eid", E_MAX, EDGE_FEATURES, n_points)
    edge_index = np.full((n_points, E_MAX, 2), -1, np.int8)
    edge_index[p, q, 0] = ed["src"].to_numpy()
    edge_index[p, q, 1] = ed["dst"].to_numpy()
    return dict(
        points=pts, node_x=node_x, node_station=node_station, node_n=node_n,
        edge_x=edge_x, edge_index=edge_index, edge_n=edge_n,
        dropped_nodes=dropped_nodes, dropped_edges=dropped_edges,
    )  # fmt: skip


ARRAYS = ("node_x", "node_station", "node_n", "edge_x", "edge_index", "edge_n")


def save(graphs: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs["points"].to_parquet(out_dir / "points.parquet", index=False)
    for name in ARRAYS:
        np.save(out_dir / f"{name}.npy", graphs[name])


def load(out_dir: Path, mmap: bool = True) -> dict:
    mode = "r" if mmap else None
    out = {name: np.load(out_dir / f"{name}.npy", mmap_mode=mode) for name in ARRAYS}
    out["points"] = pd.read_parquet(out_dir / "points.parquet")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("months", nargs="*", help="YYYY-MM months to build (default: all)")
    parser.add_argument("--horizon", type=int, default=15, choices=config.HORIZONS_MIN)
    parser.add_argument("--rebuild-static", action="store_true", help="recompute station graph")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not all(p.exists() for p in nw.MINUTE_TABLES):
        raise SystemExit("Missing minute tables: run swissdelay.features.network first")
    con = duckdb.connect()
    if args.rebuild_static or not (STATIONS_PATH.exists() and EDGES_PATH.exists()):
        interim = sorted(config.INTERIM.glob("trains_*.parquet"))
        n_st, n_ed = build_static_graph(con, interim)
        log.info("Static graph (planned runs up to %s): %d stations, %d edges",
                 config.TRAIN_END, n_st, n_ed)  # fmt: skip

    wanted = set(args.months)
    for jpath in sorted(config.PROCESSED.glob("journeys_*.parquet")):
        month = nw._month_of(jpath)
        if wanted and month not in wanted:
            continue
        lpath = config.PROCESSED / f"labels_{month}.parquet"
        g = extract(con, points_sql(jpath, lpath, args.horizon), jpath)
        save(g, GRAPH_DIR / f"h{args.horizon}" / month)
        n = len(g["points"])
        log.info(
            "%s: %d points, nodes mean %.1f / max %d (%d dropped), edges mean %.1f / max %d "
            "(%d dropped)", month, n, g["node_n"].mean(), g["node_n"].max(), g["dropped_nodes"],
            g["edge_n"].mean(), g["edge_n"].max(), g["dropped_edges"],
        )  # fmt: skip


if __name__ == "__main__":
    main()
