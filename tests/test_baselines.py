import duckdb
import pandas as pd
import pytest

from swissdelay.models import baselines
from swissdelay.models.baselines import create_labelled_view, fit, predict_sql


@pytest.fixture()
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "MIN_COUNT", 3)
    rows = []
    # Training (Sep 2025, weekdays): line IC1 at station 1 at 08:00 always loses 2 min,
    # station 2 has only 2 labels (too few for its own group -> falls back).
    for d in range(1, 6):
        day = pd.Timestamp(f"2025-09-0{d}")  # Mon 1 - Fri 5
        rows.append(("IC1", 1, day + pd.Timedelta(hours=8), day, 15, 2.0))
        rows.append(("IC1", 3, day + pd.Timedelta(hours=9), day, 15, -1.0))
    for d in (1, 2):
        day = pd.Timestamp(f"2025-09-0{d}")
        rows.append(("IC1", 2, day + pd.Timedelta(hours=8), day, 15, 10.0))
    # A validation row (May 2026) at station 1, 08:00 on a weekday
    vday = pd.Timestamp("2026-05-05")
    rows.append(("IC1", 1, vday + pd.Timedelta(hours=8), vday, 15, 1.5))
    rows.append(("IC1", 2, vday + pd.Timedelta(hours=8), vday, 15, 0.0))
    df = pd.DataFrame(
        rows,
        columns=[
            "line_name",
            "station_id",
            "dep_planned",
            "operating_day",
            "horizon_min",
            "delta_min",
        ],
    )
    df["operating_day"] = df["operating_day"].dt.date
    for c, v in {
        "trip_key": "k", "stop_seq": 1, "category": "IC", "operator_abbr": "SBB",
        "sched_gap_min": 20.0, "d0_min": 1.0, "in_common_subset": True, "label_status": "ok",
    }.items():  # fmt: skip
        df[c] = v
    src = tmp_path / "labels_x.parquet"
    c = duckdb.connect()
    c.register("df", df)
    c.execute(f"COPY (SELECT * FROM df) TO '{src}' (FORMAT parquet)")
    create_labelled_view(c, f"'{src}'")
    fit(c)
    return c


def test_offset_is_training_median(con):
    p = con.sql(predict_sql(daily_stats=None)).df()
    # training deltas: 5 x 2.0, 5 x -1.0, 2 x 10.0 -> median 2.0
    assert (p["pred_offset"] == 2.0).all()
    assert (p["pred_persistence"] == 0.0).all()


def test_historical_uses_finest_group_then_falls_back(con):
    p = con.sql(predict_sql("split = 'valid'", daily_stats=None)).df().set_index("station_id")
    assert p.loc[1, "pred_historical"] == 2.0 and p.loc[1, "hist_level"] == 0
    # station 2: only 2 training labels < MIN_COUNT at every level -> horizon offset
    assert p.loc[2, "hist_level"] == 4 and p.loc[2, "pred_historical"] == 2.0


def test_fit_uses_training_rows_only(con):
    n_valid = con.execute("SELECT sum(n) FROM hist_4").fetchone()[0]
    assert n_valid == 12  # the 2 validation rows are not used
