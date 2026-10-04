"""Tests for network-state features: all features against a plain-Python reference, the
reporting-lag boundary, segment gains, corridor, train ahead, and no use of events measured
after ``t − 2 min``."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from swissdelay.features import network as nw

DAY = pd.Timestamp("2026-03-10")
A, B, C, D, E = 8500001, 8500002, 8500003, 8500004, 8500005
FOREIGN = 8000001
ONE_MIN = pd.Timedelta(minutes=1)
ONE_S = pd.Timedelta(seconds=1)


def t(hhmmss: str) -> pd.Timestamp:
    parts = [int(x) for x in hhmmss.split(":")] + [0]
    return DAY + pd.Timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


def stop(trip, station, arr=None, arr_delay_s=None, dep=None, dep_delay_s=None, status="REAL"):
    """One interim row; delays in seconds, ``None`` = not measured."""
    return dict(
        operating_day=DAY.date(), trip_id=str(trip), station_id=station,
        arr_planned=arr, dep_planned=dep,
        arr_actual=None if arr is None or arr_delay_s is None
        else arr + pd.Timedelta(seconds=arr_delay_s),
        dep_actual=None if dep is None or dep_delay_s is None
        else dep + pd.Timedelta(seconds=dep_delay_s),
        arr_status=None if arr is None else status, dep_status=None if dep is None else status,
    )  # fmt: skip


TYPES = {
    "operating_day": "DATE", "trip_id": "VARCHAR", "station_id": "INTEGER",
    "arr_planned": "TIMESTAMP", "dep_planned": "TIMESTAMP", "arr_actual": "TIMESTAMP",
    "dep_actual": "TIMESTAMP", "arr_status": "VARCHAR", "dep_status": "VARCHAR",
}  # fmt: skip
POINT_TYPES = {
    "trip_key": "VARCHAR", "stop_seq": "INTEGER", "horizon_min": "INTEGER",
    "t_pred": "TIMESTAMP", "target_seq": "INTEGER", "cur_station_id": "INTEGER",
    "next_station_id": "INTEGER", "target_station_id": "INTEGER",
}  # fmt: skip
JOURNEY_TYPES = {"trip_key": "VARCHAR", "stop_seq": "INTEGER", "station_id": "INTEGER"}


def write(con, df: pd.DataFrame, path, types) -> None:
    con.register("df_tmp", df)
    cols = ", ".join(f"CAST({c} AS {types[c]}) AS {c}" for c in types)
    con.execute(f"COPY (SELECT {cols} FROM df_tmp) TO '{path}' (FORMAT parquet)")
    con.unregister("df_tmp")


def compute(tmp_path, rows: list[dict], points: list[dict]) -> pd.DataFrame:
    tmp_path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    interim = tmp_path / "trains_2026-03.parquet"
    write(con, pd.DataFrame(rows), interim, TYPES)
    paths = [tmp_path / f"{k}.parquet" for k in ("station", "segment", "all", "departures")]
    nw.build_minute_tables(con, [interim], *paths)
    points_df = pd.DataFrame(points).drop(columns="route")
    write(con, points_df, tmp_path / "points.parquet", POINT_TYPES)
    routes = {p["trip_key"]: p["route"] for p in points}
    journeys = pd.DataFrame(
        [dict(trip_key=k, stop_seq=i, station_id=s)
         for k, route in routes.items() for i, s in enumerate(route, start=1)]
    )  # fmt: skip
    write(con, journeys, tmp_path / "journeys.parquet", JOURNEY_TYPES)
    pts = f"SELECT * FROM read_parquet('{tmp_path / 'points.parquet'}')"
    nw.prepare(con, pts, tmp_path / "journeys.parquet", *paths)
    return con.sql(nw.network_sql()).df()


def point(key="P", seq=1, horizon=15, t_pred=None, route=(A, B, C), target_seq=3) -> dict:
    """A labelled point on ``route`` (stations by stop_seq), leaving stop ``seq``."""
    return dict(trip_key=key, stop_seq=seq, horizon_min=horizon, t_pred=t_pred,
                target_seq=target_seq, cur_station_id=route[seq - 1],
                next_station_id=route[seq], target_station_id=route[target_seq - 1],
                route=list(route))  # fmt: skip


# --------------------------------------------------------------------------- reference


def minute(ts: pd.Timestamp) -> int:
    return (ts - pd.Timestamp(0)) // ONE_MIN


def second(ts: pd.Timestamp) -> int:
    return (ts - pd.Timestamp(0)) // ONE_S


def valid(d) -> bool:
    return d is not None and not pd.isna(d) and nw.DELAY_MIN_MIN <= d <= nw.DELAY_MAX_MIN


def reference_events(rows: list[dict]):
    """Station events (station, minute, delay), segment events (from, to, minute, gain) and
    departures (from, to, dep second, dep delay, arr second or None, arr delay or None)."""
    df = pd.DataFrame(rows)
    df = df[df["station_id"] // 100000 == 85].copy()
    for k in ("arr", "dep"):
        d = (df[f"{k}_actual"] - df[f"{k}_planned"]).dt.total_seconds() / 60
        df[f"{k}_d"] = d.where((df[f"{k}_status"] == "REAL") & d.map(valid))
    station = []
    for k in ("arr", "dep"):
        ok = df[df[f"{k}_d"].notna()]
        cols = zip(ok["station_id"], ok[f"{k}_actual"], ok[f"{k}_d"], strict=True)
        station += [(s, minute(a), d) for s, a, d in cols]
    segment, departures = [], []
    df["o1"] = df["arr_planned"].fillna(df["dep_planned"])
    df["o2"] = df["dep_planned"].fillna(df["arr_planned"])
    for _, g in df.sort_values(["trip_id", "o1", "o2"]).groupby("trip_id"):
        recs = g.to_dict("records")
        for r, n in zip(recs, recs[1:], strict=False):
            if pd.isna(r["dep_d"]):
                continue
            arrived = pd.notna(n["arr_d"])
            departures.append((r["station_id"], n["station_id"], second(r["dep_actual"]),
                               r["dep_d"], second(n["arr_actual"]) if arrived else None,
                               n["arr_d"] if arrived else None))  # fmt: skip
            if arrived:
                segment.append((r["station_id"], n["station_id"], minute(n["arr_actual"]),
                                n["arr_d"] - r["dep_d"]))  # fmt: skip
    return station, segment, departures


def reference_features(rows: list[dict], p: dict) -> dict:
    """All network features of point ``p``, computed event by event."""
    station, segment, departures = reference_events(rows)
    m_end = minute(p["t_pred"] - pd.Timedelta(minutes=nw.LAG_MIN)) - 1
    cut_s = (m_end + 1) * 60

    def inside(m: int, window: int = nw.WINDOW_MIN) -> bool:
        return m_end - window < m <= m_end

    def at(sid, window=nw.WINDOW_MIN):
        return [d for s, m, d in station if s == sid and inside(m, window)]

    def everywhere(window=nw.WINDOW_MIN):
        return [d for _, m, d in station if inside(m, window)]

    def gains(frm, to, window=nw.WINDOW_MIN):
        return [g for f, t_, m, g in segment if f == frm and t_ == to and inside(m, window)]

    def mean(xs):
        return float(np.mean(xs)) if xs else np.nan

    def late(xs):
        return float(np.mean([x > nw.LATE_MIN for x in xs])) if xs else np.nan

    cur, nxt, tgt = p["cur_station_id"], p["next_station_id"], p["target_station_id"]
    out = {}
    for prefix, sid in (("cur", cur), ("next", nxt), ("tgt", tgt)):
        xs = at(sid)
        out |= {f"net_{prefix}_n": len(xs), f"net_{prefix}_mean_delay": mean(xs),
                f"net_{prefix}_share_late": late(xs)}  # fmt: skip
    g = gains(cur, nxt)
    out |= {"net_seg_n": len(g), "net_seg_mean_gain": mean(g)}
    xs = everywhere()
    out |= {"net_all_mean_delay": mean(xs), "net_all_share_late": late(xs)}
    # trend
    short, long_ = nw.SHORT_WINDOW_MIN, nw.LONG_WINDOW_MIN
    for prefix, sid in (("cur", cur), ("next", nxt)):
        xs = at(sid, short)
        out |= {f"net_{prefix}_n_10": len(xs), f"net_{prefix}_mean_delay_10": mean(xs)}
    g = gains(cur, nxt, short)
    out |= {"net_seg_n_10": len(g), "net_seg_mean_gain_10": mean(g)}
    xs = everywhere(short)
    out |= {"net_all_mean_delay_10": mean(xs), "net_all_share_late_10": late(xs)}
    out |= {"net_cur_mean_delay_60": mean(at(cur, long_)),
            "net_all_mean_delay_60": mean(everywhere(long_))}  # fmt: skip
    # corridor: stops after the current one up to the target
    route = p["route"]
    ks = range(p["stop_seq"] + 1, p["target_seq"] + 1)
    xs = [d for k in ks for d in at(route[k - 1])]
    per_seg = [gains(route[k - 2], route[k - 1]) for k in ks]
    all_g = [x for g in per_seg for x in g]
    seg_means = [mean(g) for g in per_seg if g]
    out |= {
        "net_cor_n": len(xs), "net_cor_mean_delay": mean(xs), "net_cor_share_late": late(xs),
        "net_cor_seg_n": len(all_g), "net_cor_seg_mean_gain": mean(all_g),
        "net_cor_gain_sum": float(np.sum(seg_means)) if seg_means else np.nan,
        "net_cor_seg_coverage": float(np.mean([bool(g) for g in per_seg])),
    }  # fmt: skip
    # train ahead
    cands = [d for d in departures if d[0] == cur and d[1] == nxt and d[2] < cut_s]
    ahead = dict.fromkeys(["net_ahead_headway_min", "net_ahead_dep_delay",
                           "net_ahead_seg_gain"], np.nan)  # fmt: skip
    tie = False
    if cands:
        best = max(d[2] for d in cands)
        tie = sum(d[2] == best for d in cands) > 1
        _, _, dep_s, dep_d, arr_s, arr_d = next(d for d in cands if d[2] == best)
        if dep_s > cut_s - nw.AHEAD_MAX_MIN * 60:
            ahead["net_ahead_headway_min"] = (second(p["t_pred"]) - dep_s) / 60
            ahead["net_ahead_dep_delay"] = dep_d
            if arr_s is not None and arr_s < cut_s:
                ahead["net_ahead_seg_gain"] = arr_d - dep_d
    out |= ahead
    out["_ahead_tie"] = tie
    return out


def random_rows(seed: int = 0, n_trips: int = 300) -> list[dict]:
    rng = np.random.default_rng(seed)
    swiss = [A, B, C, D, E]
    rows = []
    for trip in range(n_trips):
        route = list(rng.choice(swiss, size=3, replace=False))
        if rng.random() < 0.1:
            route = [FOREIGN] + route
        clock = t("10:30") + pd.Timedelta(minutes=int(rng.integers(0, 120)))
        for i, s in enumerate(route):
            first, last = i == 0, i == len(route) - 1
            arr = None if first else clock
            dep = None if last else clock + pd.Timedelta(minutes=int(rng.integers(1, 3)))

            def delay():
                if rng.random() < 0.05:
                    return int(rng.choice([-600, 400 * 60]))  # outside the valid range
                return int(rng.integers(-120, 900))

            status = "PROGNOSE" if rng.random() < 0.1 else "REAL"
            rows.append(stop(trip, int(s), arr, delay(), dep, delay(), status))
            clock = (dep or clock) + pd.Timedelta(minutes=int(rng.integers(5, 15)))
    return rows


RANDOM_POINTS = [
    point("P1", 1, 15, t("11:00:30"), (A, B, C, D), 3),
    point("P1", 1, 30, t("11:00:30"), (A, B, C, D), 4),
    point("P2", 2, 15, t("11:30:00"), (A, B, C, E), 4),
    point("P3", 1, 15, t("12:00:59"), (C, D, A), 3),
    point("P4", 3, 60, t("12:15:01"), (C, B, D, E), 4),  # target = next stop
    point("P5", 1, 15, t("12:40:00"), (E, A, B), 2),
    point("P6", 1, 15, t("10:00:00"), (A, B, C), 3),  # before any event
]


# --------------------------------------------------------------------------- tests


def test_matches_reference(tmp_path):
    rows = random_rows()
    got = compute(tmp_path, rows, RANDOM_POINTS).set_index(nw.KEYS)
    assert list(got.columns) == nw.ALL_NETWORK_FEATURES
    assert len(got) == len(RANDOM_POINTS)
    for p in RANDOM_POINTS:
        expected = reference_features(rows, p)
        assert set(expected) - {"_ahead_tie"} == set(nw.ALL_NETWORK_FEATURES)
        actual = got.loc[(p["trip_key"], p["stop_seq"], p["horizon_min"])]
        for col, value in expected.items():
            if col == "_ahead_tie" or (expected["_ahead_tie"] and col.startswith("net_ahead")):
                continue  # two trains left in the same second: either is a valid answer
            if np.isnan(value):
                assert pd.isna(actual[col]), (p["trip_key"], col)
            else:
                assert actual[col] == pytest.approx(value), (p["trip_key"], col)
    # the reference is not trivially empty
    assert got["net_all_mean_delay"].notna().sum() >= 5
    assert got["net_cor_seg_n"].gt(0).sum() >= 3
    assert got["net_ahead_dep_delay"].notna().sum() >= 3


def test_lag_and_window_boundaries(tmp_path):
    """t = 12:00:30 → t − 2 = 11:58:30; counted minutes are 11:28 … 11:57."""
    rows = [
        stop(1, A, dep=t("11:27"), dep_delay_s=59),       # 11:27:59 too old
        stop(2, A, arr=t("11:27"), arr_delay_s=60),       # 11:28:00 first counted
        stop(3, A, dep=t("11:55"), dep_delay_s=179),      # 11:57:59 last counted
        stop(4, A, dep=t("11:55"), dep_delay_s=180),      # 11:58:00 same minute as t − 2
        stop(5, A, dep=t("11:58"), dep_delay_s=29),       # 11:58:29 before t − 2, same minute
        stop(6, A, dep=t("12:05"), dep_delay_s=60),       # future
        stop(7, A, dep=t("11:40"), dep_delay_s=60, status="PROGNOSE"),  # not measured
    ]  # fmt: skip
    got = compute(tmp_path, rows, [point(t_pred=t("12:00:30"))]).iloc[0]
    assert got["net_cur_n"] == 2
    assert got["net_cur_mean_delay"] == pytest.approx((1.0 + 2.9833333) / 2)
    assert got["net_cur_share_late"] == pytest.approx(0.0)
    assert got["net_all_mean_delay"] == pytest.approx(got["net_cur_mean_delay"])
    assert got["net_next_n"] == 0 and pd.isna(got["net_next_mean_delay"])
    # 10-min window: 11:48 … 11:57; 60-min window: 10:58 … 11:57
    assert got["net_cur_n_10"] == 1
    assert got["net_cur_mean_delay_10"] == pytest.approx(179 / 60)
    assert got["net_cur_mean_delay_60"] == pytest.approx((59 / 60 + 1.0 + 179 / 60) / 3)
    assert got["net_all_mean_delay_60"] == pytest.approx(got["net_cur_mean_delay_60"])
    # no other station on the route has events, no train ran A -> B
    assert got["net_cor_n"] == 0 and got["net_cor_seg_coverage"] == 0.0
    assert pd.isna(got["net_ahead_headway_min"])


def test_segment_gain(tmp_path):
    rows = [
        # S1 runs A -> B and gains 2 min; arrival at B at 11:50 (counted)
        stop(1, A, dep=t("11:40"), dep_delay_s=60),
        stop(1, B, arr=t("11:47"), arr_delay_s=180),
        # S2 runs A -> B and loses 1 min (gain -1)
        stop(2, A, dep=t("11:30"), dep_delay_s=240),
        stop(2, B, arr=t("11:37"), arr_delay_s=180),
        # S3 runs B -> A: other direction, not counted on A -> B
        stop(3, B, dep=t("11:30"), dep_delay_s=0),
        stop(3, A, arr=t("11:37"), arr_delay_s=600),
        # S4 arrives at B after t - 2: not counted
        stop(4, A, dep=t("11:52"), dep_delay_s=0),
        stop(4, B, arr=t("11:59"), arr_delay_s=0),
    ]
    got = compute(tmp_path, rows, [point(t_pred=t("12:00:30"))]).iloc[0]
    assert got["net_seg_n"] == 2
    assert got["net_seg_mean_gain"] == pytest.approx(0.5)  # (2 + -1) / 2
    assert got["net_next_n"] == 3  # arrivals of S1, S2 and departure of S3 at B
    assert got["net_next_share_late"] == pytest.approx(0.0)
    # at A: departures of S1 (+1), S2 (+4), S4 (+0) and arrival of S3 (+10)
    assert got["net_cur_n"] == 4
    assert got["net_cur_share_late"] == pytest.approx(2 / 4)
    # corridor A -> B -> C: events at B (3) and C (0); segment A -> B seen, B -> C not
    assert got["net_cor_n"] == 3
    assert got["net_cor_seg_n"] == 2 and got["net_cor_seg_mean_gain"] == pytest.approx(0.5)
    assert got["net_cor_gain_sum"] == pytest.approx(0.5)
    assert got["net_cor_seg_coverage"] == pytest.approx(0.5)
    # train ahead on A -> B: S4 left at 11:52:00 (before the 11:58 cut-off), not yet arrived
    assert got["net_ahead_headway_min"] == pytest.approx(8.5)
    assert got["net_ahead_dep_delay"] == pytest.approx(0.0)
    assert pd.isna(got["net_ahead_seg_gain"])


def test_train_ahead_too_old_or_after_cutoff(tmp_path):
    rows = [
        stop(1, A, dep=t("10:50"), dep_delay_s=0),  # 68 min before the cut-off: ignored
        stop(1, B, arr=t("10:57"), arr_delay_s=0),
        stop(2, A, dep=t("11:58"), dep_delay_s=10),  # after the cut-off
        stop(2, B, arr=t("12:05"), arr_delay_s=0),
    ]
    got = compute(tmp_path, rows, [point(t_pred=t("12:00:30"))]).iloc[0]
    assert pd.isna(got["net_ahead_headway_min"]) and pd.isna(got["net_ahead_dep_delay"])


def test_events_after_lag_do_not_change_features(tmp_path):
    """Changing or adding any event measured at or after the minute of t − 2 min must not
    change any feature."""
    rows = random_rows(seed=1)
    p = point("P", 1, 15, t("11:45:40"), (A, B, C, D), 4)
    cutoff = (p["t_pred"] - pd.Timedelta(minutes=nw.LAG_MIN)).floor("min")
    base = compute(tmp_path / "a", rows, [p])

    shifted = []
    for r in rows:
        r = dict(r)
        for k in ("arr", "dep"):
            if r[f"{k}_actual"] is not None and r[f"{k}_actual"] >= cutoff:
                r[f"{k}_actual"] += pd.Timedelta(minutes=7)  # later, so still after cutoff
        shifted.append(r)
    shifted += [stop(9000 + i, s, arr=cutoff, arr_delay_s=600, dep=cutoff + ONE_MIN,
                     dep_delay_s=900) for i, s in enumerate((A, B, C))]  # fmt: skip
    other = compute(tmp_path / "b", shifted, [p])
    pd.testing.assert_frame_equal(base, other)


def test_points_sql(tmp_path):
    con = duckdb.connect()
    journeys, labels = tmp_path / "journeys.parquet", tmp_path / "labels.parquet"
    con.execute(f"""COPY (SELECT * FROM (VALUES ('k', 1, {A}), ('k', 2, {B}), ('k', 3, {C}))
                     j(trip_key, stop_seq, station_id)) TO '{journeys}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT * FROM (VALUES
                       ('k', 1, 15, TIMESTAMP '2026-03-10 10:00', {A}, 3, {C}, 'ok'),
                       ('k', 2, 15, TIMESTAMP '2026-03-10 10:10', {B}, 3, {C}, 'ok'),
                       ('k', 2, 30, TIMESTAMP '2026-03-10 10:10', {B}, NULL, NULL, 'no_target'))
                     l(trip_key, stop_seq, horizon_min, t_pred, station_id, target_seq,
                       target_station_id, label_status)) TO '{labels}' (FORMAT parquet)""")
    got = con.sql(nw.points_sql(journeys, labels) + " ORDER BY stop_seq").df()
    assert got["next_station_id"].tolist() == [B, C]
    assert got["cur_station_id"].tolist() == [A, B]
    assert len(got) == 2
