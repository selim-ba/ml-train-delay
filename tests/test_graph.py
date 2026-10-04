"""Tests for per-point subgraphs: node and edge selection, consistency of the window
features with ``swissdelay.features.network``, caps, and no use of events measured after
``t − 2 min``."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from swissdelay.features import graph as gr
from swissdelay.features import network as nw

DAY = pd.Timestamp("2026-03-10")
A, B, C, D, E, F = 8500001, 8500002, 8500003, 8500004, 8500005, 8500006


def t(hhmm: str) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return DAY + pd.Timedelta(hours=h, minutes=m)


def stop(trip, station, arr=None, dep=None, delay_s=0):
    return dict(
        operating_day=DAY.date(), trip_id=str(trip), station_id=station,
        arr_planned=arr, dep_planned=dep,
        arr_actual=None if arr is None else arr + pd.Timedelta(seconds=delay_s),
        dep_actual=None if dep is None else dep + pd.Timedelta(seconds=delay_s),
        arr_status=None if arr is None else "REAL", dep_status=None if dep is None else "REAL",
    )  # fmt: skip


def line_abcd(trip, start: str, delay_s: int) -> list[dict]:
    """A (dep +0) → B (arr +8, dep +9) → C (arr +20, dep +21) → D (arr +30)."""
    s = t(start)

    def m(x: int) -> pd.Timestamp:
        return s + pd.Timedelta(minutes=x)

    return [
        stop(trip, A, dep=m(0), delay_s=delay_s),
        stop(trip, B, m(8), m(9), delay_s),
        stop(trip, C, m(20), m(21), delay_s),
        stop(trip, D, arr=m(30), delay_s=delay_s),
    ]


ROWS = [
    *line_abcd(1, "11:00", 0),
    *line_abcd(2, "11:20", 60),
    *line_abcd(3, "11:40", 120),
    stop(4, E, dep=t("11:30"), delay_s=300), stop(4, B, arr=t("11:38"), delay_s=300),
    stop(5, F, dep=t("11:35"), delay_s=0), stop(5, C, arr=t("11:45"), delay_s=0),
    stop(6, F, dep=t("11:45"), delay_s=240), stop(6, C, arr=t("11:55"), delay_s=240),
]  # fmt: skip
T_PRED = t("12:00") + pd.Timedelta(seconds=30)  # cut-off: events before 11:58:00

TYPES = {
    "operating_day": "DATE", "trip_id": "VARCHAR", "station_id": "INTEGER",
    "arr_planned": "TIMESTAMP", "dep_planned": "TIMESTAMP", "arr_actual": "TIMESTAMP",
    "dep_actual": "TIMESTAMP", "arr_status": "VARCHAR", "dep_status": "VARCHAR",
}  # fmt: skip


POINT_TYPES = {
    "trip_key": "VARCHAR", "stop_seq": "INTEGER", "horizon_min": "INTEGER",
    "t_pred": "TIMESTAMP", "target_seq": "INTEGER", "cur_station_id": "INTEGER",
    "next_station_id": "INTEGER", "target_station_id": "INTEGER", "delta_min": "DOUBLE",
}  # fmt: skip
JOURNEY_TYPES = {"trip_key": "VARCHAR", "stop_seq": "INTEGER", "station_id": "INTEGER",
                 "arr_planned": "TIMESTAMP", "dep_planned": "TIMESTAMP"}  # fmt: skip


def write(con, df, path, types):
    con.register("df_tmp", df)
    cols = ", ".join(f"CAST({c} AS {types[c]}) AS {c}" for c in types)
    con.execute(f"COPY (SELECT {cols} FROM df_tmp) TO '{path}' (FORMAT parquet)")
    con.unregister("df_tmp")


def run(tmp_path, rows=ROWS):
    """Static graph + subgraph of one point P (route A, B, C, D; leaving A; target C)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    interim = tmp_path / "trains_2026-03.parquet"
    write(con, pd.DataFrame(rows), interim, TYPES)
    minutes = tuple(tmp_path / f"{k}.parquet" for k in ("st", "sg", "all", "dep"))
    nw.build_minute_tables(con, [interim], *minutes)
    stations, edges = tmp_path / "stations.parquet", tmp_path / "edges.parquet"
    gr.build_static_graph(con, [interim], stations, edges, min_runs=1)
    journeys = tmp_path / "journeys.parquet"
    write(con, pd.DataFrame({
        "trip_key": "P", "stop_seq": [1, 2, 3, 4], "station_id": [A, B, C, D],
        "arr_planned": [None, t("12:08"), t("12:20"), t("12:30")],
        "dep_planned": [t("12:00"), t("12:09"), t("12:21"), None],
    }), journeys, JOURNEY_TYPES)  # fmt: skip
    points = tmp_path / "points.parquet"
    point = dict(trip_key="P", stop_seq=1, horizon_min=15, t_pred=T_PRED, target_seq=3,
                 cur_station_id=A, next_station_id=B, target_station_id=C,
                 delta_min=-0.5)  # fmt: skip
    write(con, pd.DataFrame([point]), points, POINT_TYPES)
    g = gr.extract(con, f"SELECT * FROM read_parquet('{points}')", journeys, stations, edges,
                   minutes)  # fmt: skip
    net = con.sql(nw.network_sql()).df().iloc[0]  # same temp tables as the extraction
    return g, net


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return run(tmp_path_factory.mktemp("graph"))


def node(g, pos, name):
    return float(g["node_x"][0, pos, gr.NODE_FEATURES.index(name)])


def edge(g, eid, name):
    return float(g["edge_x"][0, eid, gr.EDGE_FEATURES.index(name)])


def test_static_graph(tmp_path):
    con = duckdb.connect()
    interim = tmp_path / "trains.parquet"
    write(con, pd.DataFrame(ROWS), interim, TYPES)
    st, ed = tmp_path / "s.parquet", tmp_path / "e.parquet"
    assert gr.build_static_graph(con, [interim], st, ed, min_runs=1) == (6, 5)
    e = pd.read_parquet(ed).set_index(["from_station_id", "to_station_id"])
    assert e.loc[(A, B), "n_runs"] == 3 and e.loc[(A, B), "run_planned_min"] == 8
    assert e.loc[(F, C), "n_runs"] == 2 and e.loc[(E, B), "run_planned_min"] == 8
    s = pd.read_parquet(st).set_index("station_id")
    assert s["daily_stops"].to_dict() == {A: 3, B: 4, C: 5, D: 3, E: 1, F: 2}
    assert sorted(s["idx"]) == [1, 2, 3, 4, 5, 6]
    # min_runs drops rare edges (E → B has one run)
    assert gr.build_static_graph(con, [interim], st, ed, min_runs=2) == (6, 4)


def test_nodes_and_roles(built):
    g, _ = built
    assert g["node_n"][0] == 6 and g["node_x"].shape == (1, gr.N_MAX, len(gr.NODE_FEATURES))
    # route A, B, C in order, then neighbours by planned traffic: D (3), F (2), E (1)
    roles = [max(gr.ROLES, key=lambda r: node(g, p, f"is_{r}")) for p in range(6)]
    assert roles == ["current", "route", "target", "neighbour", "neighbour", "neighbour"]
    assert [node(g, p, "route_offset") for p in range(6)] == [0, 1, 2, 0, 0, 0]
    assert [node(g, p, "planned_min") for p in range(3)] == pytest.approx([-0.5, 7.5, 19.5])
    traffic = [node(g, p, "log_traffic") for p in range(6)]
    assert traffic == pytest.approx(np.log1p([3, 4, 5, 3, 2, 1]), rel=1e-3)
    # padding
    assert (g["node_x"][0, 6:] == 0).all() and (g["node_station"][0, 6:] == 0).all()
    assert (g["node_station"][0, :6] > 0).all()


def test_edges(built):
    g, _ = built
    assert g["edge_n"][0] == 5
    # route edges first, then by planned runs: C→D (3), F→C (2), E→B (1)
    assert g["edge_index"][0, :5].tolist() == [[0, 1], [1, 2], [2, 3], [4, 2], [5, 1]]
    assert [edge(g, e, "on_route") for e in range(5)] == [1, 1, 0, 0, 0]
    assert edge(g, 0, "run_planned_min") == 8 and edge(g, 3, "run_planned_min") == 10
    assert edge(g, 0, "log_runs") == pytest.approx(np.log1p(3), rel=1e-3)
    assert (g["edge_index"][0, 5:] == -1).all()


def test_window_features_match_network_features(built):
    g, net = built

    def same(value, expected):
        expected = 0.0 if pd.isna(expected) else expected  # no event: mean set to 0
        assert value == pytest.approx(expected, rel=2e-3, abs=2e-3)

    for pos, prefix in ((0, "cur"), (1, "next"), (2, "tgt")):
        same(node(g, pos, "log_n_30"), np.log1p(net[f"net_{prefix}_n"]))
        same(node(g, pos, "mean_delay_30"), net[f"net_{prefix}_mean_delay"])
        same(node(g, pos, "share_late_30"), net[f"net_{prefix}_share_late"])
    same(node(g, 0, "log_n_10"), np.log1p(net["net_cur_n_10"]))
    same(node(g, 0, "mean_delay_10"), net["net_cur_mean_delay_10"])
    same(node(g, 1, "mean_delay_10"), net["net_next_mean_delay_10"])
    same(node(g, 0, "mean_delay_60"), net["net_cur_mean_delay_60"])
    same(edge(g, 0, "log_n_30"), np.log1p(net["net_seg_n"]))
    same(edge(g, 0, "mean_gain_30"), net["net_seg_mean_gain"])
    same(edge(g, 0, "log_n_10"), np.log1p(net["net_seg_n_10"]))
    # hand check at B (30 min: arrivals and departures of T1-T3 and E's train before 11:58)
    assert node(g, 1, "share_late_30") > 0  # train 4 arrived 5 min late
    assert net["net_cur_n"] > 0 and net["net_seg_n"] > 0  # the test is not trivially empty


def test_events_after_cutoff_do_not_change_graphs(tmp_path, built):
    g, _ = built
    cutoff = (T_PRED - pd.Timedelta(minutes=nw.LAG_MIN)).floor("min")
    shifted = []
    for r in ROWS:
        r = dict(r)
        for k in ("arr", "dep"):
            if r[f"{k}_actual"] is not None and r[f"{k}_actual"] >= cutoff:
                r[f"{k}_actual"] += pd.Timedelta(minutes=7)
        shifted.append(r)
    other, _ = run(tmp_path, shifted)
    for name in gr.ARRAYS:
        np.testing.assert_array_equal(g[name], other[name], err_msg=name)


def test_node_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "N_MAX", 4)
    g, _ = run(tmp_path)
    assert g["node_x"].shape[1] == 4 and g["node_n"][0] == 4 and g["dropped_nodes"] == 2
    assert g["edge_n"][0] == 3  # A→B, B→C, C→D: edges to dropped nodes F, E are gone
