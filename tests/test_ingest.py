"""Tests for the monthly ingestion on tiny synthetic Ist-Daten files."""

import zipfile

import duckdb
import pytest

from swissdelay.data.ingest import (
    ENGLISH_COLUMNS,
    RAW_COLUMNS,
    IngestError,
    days_of_month,
    ingest_source,
    month_range,
)


def _row(**overrides: str) -> str:
    values = {
        "BETRIEBSTAG": "01.08.2025",
        "FAHRT_BEZEICHNER": "85:11:1001:001",
        "BETREIBER_ID": "85:11",
        "BETREIBER_ABK": "SBB",
        "BETREIBER_NAME": "Schweizerische Bundesbahnen SBB",
        "PRODUKT_ID": "Zug",
        "LINIEN_ID": "1001",
        "LINIEN_TEXT": "IC1",
        "UMLAUF_ID": "",
        "VERKEHRSMITTEL_TEXT": "IC",
        "ZUSATZFAHRT_TF": "false",
        "FAELLT_AUS_TF": "false",
        "BPUIC": "8503000",
        "HALTESTELLEN_NAME": "Zürich HB",
        "ANKUNFTSZEIT": "01.08.2025 06:41",
        "AN_PROGNOSE": "01.08.2025 06:42:30",
        "AN_PROGNOSE_STATUS": "REAL",
        "ABFAHRTSZEIT": "01.08.2025 06:43",
        "AB_PROGNOSE": "01.08.2025 06:44:10",
        "AB_PROGNOSE_STATUS": "REAL",
        "DURCHFAHRT_TF": "false",
    }
    values.update(overrides)
    return ";".join(values[c] for c in RAW_COLUMNS)


def _day_csv(day: str, rows: list[str] | None = None) -> str:
    d, m, y = day.split(".")
    rows = rows or [
        # first stop: no arrival
        _row(BETRIEBSTAG=day, ANKUNFTSZEIT="", AN_PROGNOSE="", AN_PROGNOSE_STATUS=""),
        # bus: must be dropped
        _row(BETRIEBSTAG=day, PRODUKT_ID="Bus", VERKEHRSMITTEL_TEXT="B"),
        # upper-case product: must be kept
        _row(BETRIEBSTAG=day, PRODUKT_ID="ZUG", VERKEHRSMITTEL_TEXT="ZUG", FAELLT_AUS_TF="true"),
    ]
    rows = [r.replace("01.08.2025", day) for r in rows]
    return ";".join(RAW_COLUMNS) + "\n" + "\n".join(rows) + "\n"


def _zip(tmp_path, files: dict[str, str]):
    path = tmp_path / "month.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(f"ist-daten-v2-2025-08/{name}", text)
    return path


def _run(tmp_path, files, require_full_month=False):
    out = tmp_path / "trains_2025-08.parquet"
    summary = ingest_source(
        _zip(tmp_path, files), "2025-08", out, tmp_path / "work", require_full_month
    )
    return out, summary


def test_month_helpers():
    assert month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert len(days_of_month("2026-02")) == 28


def test_filters_trains_and_renames(tmp_path):
    files = {
        "2025-08-01_IstDaten.csv": _day_csv("01.08.2025"),
        "2025-08-02_IstDaten.csv": _day_csv("02.08.2025"),
    }
    out, summary = _run(tmp_path, files)

    rel = duckdb.sql(f"SELECT * FROM read_parquet('{out}')")
    assert rel.columns == ENGLISH_COLUMNS
    types = dict(zip(rel.columns, map(str, rel.types), strict=True))
    assert types["operating_day"] == "DATE"
    assert types["arr_planned"] == types["dep_actual"] == "TIMESTAMP"
    assert types["station_id"] == "INTEGER"
    assert types["is_cancelled"] == "BOOLEAN"

    modes = duckdb.sql(f"SELECT DISTINCT transport_mode FROM read_parquet('{out}')").fetchall()
    assert {m[0] for m in modes} == {"Zug", "ZUG"}
    assert summary["n_train_rows"] == 4
    assert summary["n_raw_rows"] == 6
    assert summary["n_days"] == 2
    assert not (tmp_path / "work").exists()

    delay = duckdb.sql(
        f"SELECT date_diff('second', dep_planned, dep_actual) FROM read_parquet('{out}') LIMIT 1"
    ).fetchone()[0]
    assert delay == 70


def test_bad_timestamp_fails(tmp_path):
    bad = _day_csv("01.08.2025", [_row(AB_PROGNOSE="2025-08-01T06:44:10")])
    with pytest.raises(IngestError, match="bad_dep_actual"):
        _run(tmp_path, {"2025-08-01_IstDaten.csv": bad})
    assert not (tmp_path / "trains_2025-08.parquet").exists()


def test_operating_day_must_match_file_name(tmp_path):
    with pytest.raises(IngestError, match="file date"):
        _run(tmp_path, {"2025-08-01_IstDaten.csv": _day_csv("02.08.2025")})


def test_missing_days_fail_for_full_month(tmp_path):
    with pytest.raises(IngestError, match="missing days"):
        _run(
            tmp_path,
            {"2025-08-01_IstDaten.csv": _day_csv("01.08.2025")},
            require_full_month=True,
        )


def test_missing_column_fails(tmp_path):
    text = _day_csv("01.08.2025").replace("DURCHFAHRT_TF", "SOMETHING_ELSE")
    with pytest.raises(IngestError, match="missing columns"):
        _run(tmp_path, {"2025-08-01_IstDaten.csv": text})


def test_known_extra_column_is_dropped(tmp_path):
    text = _day_csv("01.08.2025")
    lines = text.strip().split("\n")
    text = "\n".join([lines[0] + ";SLOID"] + [ln + ";ch:1:sloid:3000" for ln in lines[1:]]) + "\n"
    out, _ = _run(tmp_path, {"2025-08-01_IstDaten.csv": text})
    assert duckdb.sql(f"SELECT * FROM read_parquet('{out}')").columns == ENGLISH_COLUMNS
