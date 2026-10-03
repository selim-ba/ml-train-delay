"""Tests for model features: values on a hand-computed route, and no use of the future."""

import duckdb
import pandas as pd
import pytest

from swissdelay.features import build as fb
from swissdelay.features import labels as lb

DAY0 = pd.Timestamp("2025-09-01")
STATIONS = [8500001, 8500002, 8500003, 8500004, 8500005]
# Planned timetable (minutes after 10:00): (arrival, departure) at each stop
PLAN = [(None, 0), (10, 12), (20, 21), (40, 41), (65, None)]
N_TRIPS = 30


def journey_rows(trip: int, day: pd.Timestamp, d0: float = 3.0) -> list[dict]:
    """One run of the route. Every segment is run 1 min faster than planned and every
    intermediate dwell 0.5 min shorter, so the reference times are known exactly."""
    rows, dep_delay = [], None
    base = day + pd.Timedelta(hours=10)
    for seq, (station, (arr, dep)) in enumerate(zip(STATIONS, PLAN, strict=True), start=1):
        arr_delay = None if arr is None else dep_delay - 1.0
        if dep is None:
            dep_delay = None
        elif arr is None:
            dep_delay = d0
        else:
            dep_delay = arr_delay - 0.5
        rows.append(
            dict(
                operating_day=day.date(), trip_key=f"{day.date()}|{trip}", trip_id=str(trip),
                stop_seq=seq, n_stops=len(STATIONS), station_id=station, category="IC",
                line_name="IC1", operator_abbr="SBB", train_number=str(trip),
                arr_planned=None if arr is None else base + pd.Timedelta(minutes=arr),
                dep_planned=None if dep is None else base + pd.Timedelta(minutes=dep),
                arr_delay_min=arr_delay, dep_delay_min=dep_delay,
                is_pass_through=False, is_cancelled=False, is_valid_station=True,
                is_plan_inconsistent=False, is_extra_trip=False, enters_from_abroad=False,
                is_prediction_point=dep_delay is not None,
            )
        )  # fmt: skip
        if dep is not None:
            rows[-1]["dep_actual"] = rows[-1]["dep_planned"] + pd.Timedelta(minutes=dep_delay)
    return rows


TYPES = {
    "operating_day": "DATE",
    "stop_seq": "INTEGER",
    "n_stops": "INTEGER",
    "station_id": "INTEGER",
    "arr_planned": "TIMESTAMP",
    "dep_planned": "TIMESTAMP",
    "dep_actual": "TIMESTAMP",
    "arr_delay_min": "DOUBLE",
    "dep_delay_min": "DOUBLE",
}
BOOL = ["is_pass_through", "is_cancelled", "is_valid_station", "is_plan_inconsistent",
        "is_extra_trip", "enters_from_abroad", "is_prediction_point"]  # fmt: skip


def write_parquet(con, rows, path):
    df = pd.DataFrame(rows)
    types = TYPES | dict.fromkeys(BOOL, "BOOLEAN")
    con.register("df_tmp", df)
    cols = ", ".join(f"CAST({c} AS {types.get(c, 'VARCHAR')}) AS {c}" for c in df.columns)
    con.execute(f"COPY (SELECT {cols} FROM df_tmp) TO '{path}' (FORMAT parquet)")
    con.unregister("df_tmp")


def build_all(tmp_path, rows, monkeypatch):
    monkeypatch.setattr(fb, "MIN_COUNT", 5)
    tmp_path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    j, lab = tmp_path / "journeys_2025-09.parquet", tmp_path / "labels_2025-09.parquet"
    seg, dwell = tmp_path / "segment_stats.parquet", tmp_path / "dwell_stats.parquet"
    out = tmp_path / "features_2025-09.parquet"
    write_parquet(con, rows, j)
    lb.build_month(con, j, lab)
    fb.build_reference_stats(con, [j], "2026-04-30", seg, dwell)
    fb.build_month(con, j, lab, out, seg, dwell, baseline=None)
    feats = con.sql(f"SELECT * FROM read_parquet('{out}')").df()
    stats = (con.sql(f"SELECT * FROM read_parquet('{seg}')").df(),
             con.sql(f"SELECT * FROM read_parquet('{dwell}')").df())  # fmt: skip
    return feats, stats


ROWS = [r for i in range(N_TRIPS) for r in journey_rows(i, DAY0 + pd.Timedelta(days=i % 28))]


@pytest.fixture()
def built(tmp_path, monkeypatch):
    return build_all(tmp_path, ROWS, monkeypatch)


def point(feats, stop_seq, horizon, trip="0"):
    sel = feats[(feats["trip_key"].str.endswith(f"|{trip}")) & (feats["stop_seq"] == stop_seq)]
    return sel[sel["horizon_min"] == horizon].iloc[0]


def test_all_feature_columns_exist(built):
    feats, _ = built
    missing = [c for c in fb.ALL_FEATURES + fb.KEY_COLUMNS + fb.META_COLUMNS if c not in feats]
    assert not missing


def test_reference_times(built):
    _, (seg, dwell) = built
    s = seg.set_index(["from_station_id", "to_station_id"])
    assert s.loc[(8500001, 8500002), "run_ref_min"] == pytest.approx(9.0)  # planned 10, -1
    assert s.loc[(8500003, 8500004), "run_ref_min"] == pytest.approx(18.0)  # planned 19, -1
    d = dwell.set_index("station_id")
    assert d.loc[8500002, "dwell_ref_min"] == pytest.approx(1.5)  # planned 2, -0.5


def test_current_and_history(built):
    feats, _ = built
    origin = point(feats, 1, 15)
    assert origin["d0_min"] == pytest.approx(3.0)
    assert pd.isna(origin["delay_lag1_min"]) and pd.isna(origin["arr_delay_now_min"])
    assert origin["n_observed_prev"] == 0
    s3 = point(feats, 3, 15)  # delays: dep1 3.0, dep2 1.5, arr3 0.5, dep3 0.0
    assert s3["d0_min"] == pytest.approx(0.0)
    assert s3["arr_delay_now_min"] == pytest.approx(0.5)
    assert s3["dwell_excess_now_min"] == pytest.approx(-0.5)
    assert s3["delay_lag1_min"] == pytest.approx(1.5)
    assert s3["delta_last1_min"] == pytest.approx(-1.5)
    assert s3["max_prev_delay_min"] == pytest.approx(3.0)
    assert s3["n_observed_prev"] == 2


def test_slack_to_target(built):
    feats, _ = built
    p = point(feats, 1, 15)  # origin -> target stop 3: segments 1-2 and 2-3, dwell at stop 2
    assert p["target_seq"] == 3
    assert p["run_reserve_to_target_min"] == pytest.approx(2.0)  # 1 min per segment
    assert p["dwell_reserve_to_target_min"] == pytest.approx(0.5)  # stop 2
    assert p["slack_to_target_min"] == pytest.approx(2.5)
    q = point(feats, 2, 15)  # stop 2 (dep 10:12) -> first arrival >= 10:27: stop 4 (10:40)
    assert q["target_seq"] == 4
    assert q["slack_to_target_min"] == pytest.approx(2.5)  # segments 2-3, 3-4 + dwell at 3


def test_timetable_and_context(built):
    feats, _ = built
    p = point(feats, 2, 15)
    assert p["minutes_since_origin"] == pytest.approx(12.0)
    assert p["planned_dwell_now_min"] == pytest.approx(2.0)
    assert p["stops_remaining"] == 3
    assert p["n_stops_to_target"] == p["target_seq"] - 2
    assert p["hour"] == 10 and p["day_type"] == "weekday" and p["split"] == "train"


def test_features_do_not_use_the_future(tmp_path, monkeypatch):
    """Changing actual delays AFTER the current stop must not change any feature."""
    base, _ = build_all(tmp_path / "a", ROWS, monkeypatch)
    for current in (1, 2, 3):
        shifted = []
        for r in ROWS:
            r = dict(r)
            if r["trip_key"].endswith("|0") and r["stop_seq"] > current:
                r["arr_delay_min"] = None if r["arr_delay_min"] is None else r["arr_delay_min"] + 7
                if r["dep_delay_min"] is not None:
                    r["dep_delay_min"] += 7
                    r["dep_actual"] += pd.Timedelta(minutes=7)
            shifted.append(r)
        other, _ = build_all(tmp_path / f"b{current}", shifted, monkeypatch)
        key = ["trip_key", "stop_seq", "horizon_min"]
        cols = [c for c in fb.ALL_FEATURES if c != "pred_historical"]
        a = base[base["trip_key"].str.endswith("|0") & (base["stop_seq"] == current)]
        b = other[other["trip_key"].str.endswith("|0") & (other["stop_seq"] == current)]
        a = a.sort_values(key)[cols].reset_index(drop=True)
        b = b.sort_values(key)[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
