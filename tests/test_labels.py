"""Tests for horizon labels on small synthetic journeys."""

import duckdb
import pandas as pd
import pytest

from swissdelay.features.labels import OUTPUT_COLUMNS, build_month

DAY = pd.Timestamp("2026-03-10")


def t(hhmm: str) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return DAY + pd.Timedelta(hours=h, minutes=m)


def row(trip, seq, arr=None, arr_delay=None, dep=None, dep_delay=None, **flags):
    r = dict(
        operating_day=DAY.date(), trip_key=f"2026-03-10|{trip}", stop_seq=seq,
        station_id=8500000 + seq, category="IC", line_name="IC1", operator_abbr="SBB",
        train_number="1", arr_planned=arr, arr_delay_min=arr_delay, dep_planned=dep,
        dep_actual=None if dep is None or dep_delay is None
        else dep + pd.Timedelta(minutes=dep_delay),
        dep_delay_min=dep_delay, is_pass_through=False, is_valid_station=True,
        is_plan_inconsistent=False, is_cancelled=False,
        is_prediction_point=dep is not None and dep_delay is not None,
    )  # fmt: skip
    r.update(flags)
    return r


def base_rows(scale: float = 1.0) -> list[dict]:
    """Trip T: 5 stops. ``scale`` multiplies every actual delay (for the leakage test)."""
    return [
        row("T", 1, dep=t("10:00"), dep_delay=2 * scale),
        row("T", 2, arr=t("10:10"), arr_delay=3 * scale, dep=t("10:12"), dep_delay=3 * scale),
        row("T", 3, arr=t("10:20"), arr_delay=4 * scale, dep=t("10:21"), dep_delay=4 * scale),
        row("T", 4, arr=t("10:40"), arr_delay=1 * scale, dep=t("10:41"), dep_delay=1 * scale),
        row("T", 5, arr=t("11:05"), arr_delay=5 * scale),
    ]


EXTRA = [
    # U: target at 15 min is cancelled; at 30 min not measured
    row("U", 1, dep=t("12:00"), dep_delay=0),
    row("U", 2, arr=t("12:20"), arr_delay=1, is_cancelled=True),
    row("U", 3, arr=t("12:35"), arr_delay=None),
    # V: pass-through and invalid station are skipped when choosing the target
    row("V", 1, dep=t("13:00"), dep_delay=1),
    row("V", 2, arr=t("13:16"), arr_delay=9, is_pass_through=True),
    row("V", 3, arr=t("13:17"), arr_delay=9, is_valid_station=False),
    row("V", 4, arr=t("13:18"), arr_delay=4, is_plan_inconsistent=True),
    row("V", 5, arr=t("13:20"), arr_delay=3),
]

TYPES = {"operating_day": "DATE", "stop_seq": "INTEGER", "station_id": "INTEGER"}
TS = ("arr_planned", "dep_planned", "dep_actual")
FLOAT = ("arr_delay_min", "dep_delay_min")
BOOL = ("is_pass_through", "is_valid_station", "is_plan_inconsistent", "is_cancelled",
        "is_prediction_point")  # fmt: skip


def build(tmp_path, rows) -> pd.DataFrame:
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    types = TYPES | dict.fromkeys(TS, "TIMESTAMP") | dict.fromkeys(FLOAT, "DOUBLE")
    types |= dict.fromkeys(BOOL, "BOOLEAN")
    con = duckdb.connect()
    con.register("df", df)
    cols = ", ".join(f"CAST({c} AS {types.get(c, 'VARCHAR')}) AS {c}" for c in df.columns)
    src, out = tmp_path / "journeys_2026-03.parquet", tmp_path / "labels_2026-03.parquet"
    con.execute(f"COPY (SELECT {cols} FROM df) TO '{src}' (FORMAT parquet)")
    build_month(con, src, out)
    return con.sql(f"SELECT * FROM read_parquet('{out}')").df()


def lab(df, trip, seq, h):
    sel = df[(df["trip_key"] == f"2026-03-10|{trip}") & (df["stop_seq"] == seq)]
    return sel[sel["horizon_min"] == h].iloc[0]


@pytest.fixture(scope="module")
def labels(tmp_path_factory):
    return build(tmp_path_factory.mktemp("labels"), base_rows() + EXTRA)


def test_columns_and_one_row_per_point_and_horizon(labels):
    assert list(labels.columns) == OUTPUT_COLUMNS
    # prediction points: T1-T4 and U1, V1 -> 6 points x 3 horizons
    assert len(labels) == 18
    assert set(labels["label_status"]) <= {"ok", "terminates", "cancelled", "not_measured"}


def test_target_stop_is_first_scheduled_at_least_h_ahead(labels):
    assert lab(labels, "T", 1, 15)["target_seq"] == 3  # 10:20 >= 10:15
    assert lab(labels, "T", 1, 30)["target_seq"] == 4  # 10:40 >= 10:30
    assert lab(labels, "T", 1, 60)["target_seq"] == 5  # 11:05 >= 11:00
    assert lab(labels, "T", 2, 15)["target_seq"] == 4  # dep 10:12 -> >= 10:27
    assert lab(labels, "T", 1, 15)["sched_gap_min"] == pytest.approx(20.0)


def test_delta_is_target_delay_minus_current_delay(labels):
    r = lab(labels, "T", 1, 15)
    assert r["d0_min"] == pytest.approx(2.0)
    assert r["target_delay_min"] == pytest.approx(4.0)
    assert r["delta_min"] == pytest.approx(2.0)
    assert lab(labels, "T", 1, 30)["delta_min"] == pytest.approx(-1.0)  # recovery


def test_terminates_and_common_subset(labels):
    r = lab(labels, "T", 2, 60)  # dep 10:12 -> needs >= 11:12, last stop 11:05
    assert r["label_status"] == "terminates" and pd.isna(r["delta_min"])
    t1 = labels[(labels["trip_key"] == "2026-03-10|T") & (labels["stop_seq"] == 1)]
    t2 = labels[(labels["trip_key"] == "2026-03-10|T") & (labels["stop_seq"] == 2)]
    assert t1["in_common_subset"].all()
    assert not t2["in_common_subset"].any()


def test_cancelled_and_not_measured_targets(labels):
    assert lab(labels, "U", 1, 15)["label_status"] == "cancelled"
    assert lab(labels, "U", 1, 30)["label_status"] == "not_measured"
    assert lab(labels, "U", 1, 60)["label_status"] == "terminates"
    assert pd.isna(lab(labels, "U", 1, 15)["delta_min"])


def test_never_targets_are_skipped(labels):
    r = lab(labels, "V", 1, 15)  # V2 pass-through, V3 invalid station, V4 inconsistent
    assert r["target_seq"] == 5 and r["delta_min"] == pytest.approx(2.0)


def test_target_choice_does_not_depend_on_actual_times(tmp_path):
    normal = build(tmp_path / "a", base_rows(scale=1.0))
    shifted = build(tmp_path / "b", base_rows(scale=20.0))  # every train 20x later
    key = ["trip_key", "stop_seq", "horizon_min"]
    a = normal.sort_values(key)[key + ["target_seq"]].reset_index(drop=True)
    b = shifted.sort_values(key)[key + ["target_seq"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
