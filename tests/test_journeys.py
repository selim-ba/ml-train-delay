"""Tests for journey reconstruction on a small synthetic month."""

import duckdb
import pandas as pd
import pytest

from swissdelay.data.ingest import ENGLISH_COLUMNS
from swissdelay.data.journeys import OUTPUT_COLUMNS, build_month, build_station_quality

DAY = pd.Timestamp("2025-09-01")


def t(hhmm: str, seconds: int = 0) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return DAY + pd.Timedelta(hours=h, minutes=m, seconds=seconds)


def stop(
    trip,
    station,
    arr=None,
    arr_act=None,
    arr_st="REAL",
    dep=None,
    dep_act=None,
    dep_st="REAL",
    category="IC",
    operator="SBB",
    **flags,
):
    row = {c: None for c in ENGLISH_COLUMNS}
    row.update(
        operating_day=DAY.date(),
        trip_id=trip,
        operator_id="85:11",
        operator_abbr=operator,
        operator_name=operator,
        transport_mode="Zug",
        train_number="1",
        line_name=category,
        category=category,
        is_extra_trip=False,
        is_cancelled=False,
        station_id=station,
        station_name=str(station),
        arr_planned=arr,
        arr_actual=arr_act,
        arr_status=arr_st if arr else None,
        dep_planned=dep,
        dep_actual=dep_act,
        dep_status=dep_st if dep else None,
        is_pass_through=False,
    )
    row.update(flags)
    return row


ROWS = [
    # A: domestic IC, rows given out of order on purpose
    stop("A", 8507000, arr=t("11:00"), arr_act=t("11:03")),
    stop("A", 8503000, dep=t("10:00"), dep_act=t("10:01")),
    stop("A", 8500218, arr=t("10:30"), arr_act=t("10:32"), dep=t("10:32"), dep_act=t("10:34")),
    # B: EC from Italy, foreign origin (PROGNOSE) then two Swiss stops
    stop("B", 8300046, dep=t("08:00"), dep_act=t("08:05"), dep_st="PROGNOSE", category="EC"),
    stop(
        "B",
        8505213,
        arr=t("09:00"),
        arr_act=t("09:10"),
        dep=t("09:05"),
        dep_act=t("09:12"),
        category="EC",
    ),
    stop("B", 8505000, arr=t("11:00"), arr_act=t("11:08"), category="EC"),
    # C: foreign-only RE
    stop("C", 8000001, dep=t("07:00"), dep_act=t("07:00"), category="RE", operator="DB"),
    stop("C", 8000002, arr=t("07:30"), arr_act=t("07:31"), category="RE", operator="DB"),
    # D: excluded operator
    stop("D", 8503000, dep=t("12:00"), dep_act=t("12:00"), category="RE", operator="FART"),
    stop("D", 8507000, arr=t("13:00"), arr_act=t("13:01"), category="RE", operator="FART"),
    # E: corrupted placeholder date on the first departure
    stop("E", 8503000, dep=t("14:00"), dep_act=pd.Timestamp("1899-12-30"), category="IR"),
    stop("E", 8500218, arr=t("14:30"), arr_act=t("14:31"), category="IR"),
    # F: duplicated row (same station twice in a row) -> invalid journey
    stop("F", 8503000, dep=t("15:00"), dep_act=t("15:00"), category="IR"),
    stop(
        "F",
        8500218,
        arr=t("15:30"),
        arr_act=t("15:30"),
        dep=t("15:31"),
        dep_act=t("15:31"),
        category="IR",
    ),
    stop(
        "F",
        8500218,
        arr=t("15:30"),
        arr_act=t("15:30"),
        dep=t("15:31"),
        dep_act=t("15:31"),
        category="IR",
    ),
    stop("F", 8507000, arr=t("16:00"), arr_act=t("16:00"), category="IR"),
    # G: S-Bahn, out of scope
    stop("G", 8503000, dep=t("16:00"), dep_act=t("16:00"), category="S"),
    stop("G", 8500218, arr=t("16:20"), arr_act=t("16:20"), category="S"),
    # H: pass-through in the middle, PROGNOSE departure at the origin
    stop("H", 8503000, dep=t("17:00"), dep_act=t("17:02"), dep_st="PROGNOSE", category="IR"),
    stop(
        "H",
        8500218,
        arr=t("17:30"),
        arr_act=t("17:30"),
        dep=t("17:30"),
        dep_act=t("17:30"),
        category="IR",
        is_pass_through=True,
    ),
    stop("H", 8507000, arr=t("18:00"), arr_act=t("18:01"), category="IR"),
    # I: departs from a poorly measured station
    stop("I", 8509999, dep=t("19:00"), dep_act=t("19:00"), dep_st="UNBEKANNT", category="RE"),
    stop("I", 8503000, arr=t("19:30"), arr_act=t("19:31"), category="RE"),
    # J, K: more measured departures, so Bern and Zürich HB are well-measured stations
    stop("J", 8507000, dep=t("20:00"), dep_act=t("20:00")),
    stop("J", 8503000, arr=t("21:00"), arr_act=t("21:00")),
    stop("K", 8503000, dep=t("21:30"), dep_act=t("21:30")),
    stop("K", 8500218, arr=t("22:00"), arr_act=t("22:00")),
]

# L: negative planned dwell at one stop (works timetable): only that stop is excluded
ROWS += [
    stop("L", 8503000, dep=t("06:00"), dep_act=t("06:00")),
    stop("L", 8500218, arr=t("06:30"), arr_act=t("06:30"), dep=t("06:28"), dep_act=t("06:31")),
    stop("L", 8507000, arr=t("07:00"), arr_act=t("07:01")),
    # M: German RE ending at its single Swiss stop: valid journey, nothing to predict
    stop("M", 8000001, dep=t("05:00"), dep_act=t("05:00"), category="RE", operator="DB Regio"),
    stop(
        "M",
        8503000,
        arr=t("05:40"),
        arr_act=t("05:41"),
        category="RE",
        operator="DB Regio",
    ),
]

TYPES = {
    "operating_day": "DATE",
    "station_id": "INTEGER",
    **{c: "BOOLEAN" for c in ("is_extra_trip", "is_cancelled", "is_pass_through")},
    **{c: "TIMESTAMP" for c in ("arr_planned", "arr_actual", "dep_planned", "dep_actual")},
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("journeys")
    df = pd.DataFrame(ROWS, columns=ENGLISH_COLUMNS)
    src = tmp / "trains_2025-09.parquet"
    con = duckdb.connect()
    con.register("df", df)
    cols = ", ".join(f"CAST({c} AS {TYPES.get(c, 'VARCHAR')}) AS {c}" for c in ENGLISH_COLUMNS)
    con.execute(f"COPY (SELECT {cols} FROM df) TO '{src}' (FORMAT parquet)")
    stations = tmp / "station_quality.parquet"
    build_station_quality(con, [src], stations, train_end="2026-04-30")
    out = tmp / "journeys_2025-09.parquet"
    stats = build_month(con, src, out, stations)
    j = con.sql(f"SELECT * FROM read_parquet('{out}')").df()
    st = con.sql(f"SELECT * FROM read_parquet('{stations}')").df()
    return j, st, stats, con, src, tmp


def trip(j, trip_id):
    return j[j["trip_id"] == trip_id].sort_values("stop_seq").reset_index(drop=True)


def test_columns_and_scope(built):
    j, *_ = built
    assert list(j.columns) == OUTPUT_COLUMNS
    assert set(j["trip_id"]) == set("ABDEFHIJKLM")  # C foreign-only, G S-Bahn


def test_domestic_journey_is_ordered_by_planned_time(built):
    a = trip(built[0], "A")
    assert list(a["station_id"]) == [8503000, 8500218, 8507000]
    assert list(a["stop_seq"]) == [1, 2, 3]
    assert (a["n_stops"] == 3).all()
    assert a.loc[0, "dep_delay_min"] == pytest.approx(1.0)
    assert a.loc[2, "arr_delay_min"] == pytest.approx(3.0)
    assert list(a["is_prediction_point"]) == [True, True, False]  # last stop: no departure
    assert a["is_valid_journey"].all() and not a["enters_from_abroad"].any()


def test_cross_border_keeps_swiss_section(built):
    b = trip(built[0], "B")
    assert list(b["station_id"]) == [8505213, 8505000]
    assert list(b["stop_seq"]) == [1, 2]
    assert b["enters_from_abroad"].all()
    assert b["is_valid_journey"].all()
    assert b.loc[0, "is_prediction_point"]


def test_excluded_operator_is_not_eligible(built):
    d = trip(built[0], "D")
    assert not d["is_target_operator"].any()
    assert not d["is_eligible_stop"].any() and not d["is_prediction_point"].any()


def test_invalid_delay_is_missing(built):
    e = trip(built[0], "E")
    assert pd.isna(e.loc[0, "dep_delay_min"])
    assert not e.loc[0, "is_prediction_point"]
    assert e.loc[1, "arr_delay_min"] == pytest.approx(1.0)


def test_duplicated_row_invalidates_journey(built):
    f = trip(built[0], "F")
    assert not f["is_valid_journey"].any()
    assert not f["is_prediction_point"].any()


def test_pass_through_and_non_real(built):
    h = trip(built[0], "H")
    assert pd.isna(h.loc[0, "dep_delay_min"])  # PROGNOSE is not observed
    assert not h.loc[0, "is_prediction_point"]
    assert h.loc[1, "is_pass_through"] and not h.loc[1, "is_eligible_stop"]
    assert h.loc[2, "is_eligible_stop"]


def test_station_quality(built):
    j, st, *_ = built
    q = st.set_index("station_id")
    assert not q.loc[8509999, "is_valid_station"]
    assert q.loc[8503000, "is_valid_station"]
    assert 8300046 not in q.index  # foreign stations are not rated
    i = trip(j, "I")
    assert not i.loc[0, "is_valid_station"] and not i.loc[0, "is_eligible_stop"]


def test_station_quality_uses_training_months_only(built):
    *_, con, src, tmp = built
    early = tmp / "station_quality_early.parquet"
    build_station_quality(con, [src], early, train_end="2025-08-31")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{early}')").fetchone()[0]
    assert n == 0


def test_summary_counts(built):
    j, _, stats, *_ = built
    assert stats["rows"] == len(j)
    assert stats["journeys"] == j["trip_key"].nunique()
    assert stats["prediction_points"] == int(j["is_prediction_point"].sum())


def test_negative_planned_dwell_excludes_only_that_stop(built):
    lj = trip(built[0], "L")
    assert lj["is_valid_journey"].all()
    assert list(lj["is_plan_inconsistent"]) == [False, True, False]
    assert lj.loc[0, "is_prediction_point"] and not lj.loc[1, "is_eligible_stop"]


def test_single_swiss_stop_is_valid_but_not_eligible(built):
    m = trip(built[0], "M")
    assert len(m) == 1 and m.loc[0, "n_stops"] == 1
    assert m.loc[0, "is_valid_journey"] and m.loc[0, "enters_from_abroad"]
    assert not m.loc[0, "is_eligible_stop"]
