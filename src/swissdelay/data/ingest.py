"""Monthly ingestion of Ist-Daten v2 archives into train-only Parquet files.

For each month, this module:

1. finds the month's raw input, in this order: ``--from-dir`` / ``--zip`` if given,
   else ``data/raw/ist-daten-v2-YYYY-MM/`` (unzipped folder), else
   ``data/raw/ist-daten-v2-YYYY-MM.zip``, else downloads that zip from the archive,
2. converts each daily CSV, one at a time, into a Parquet file that keeps every
   train row (``upper(PRODUKT_ID) = 'ZUG'``) and every column, renamed to English,
3. validates each day (schema, timestamp parsing, operating day = file date)
   and the month (every calendar day present),
4. merges the days into ``data/interim/trains_YYYY-MM.parquet``,
5. records provenance in ``data/dataset_manifest.json``,
6. deletes the raw zip / folder (unless ``--keep-raw``).

Only one daily CSV is unpacked at a time, so peak disk use is about the zip plus
~0.5 GB. If any check fails, nothing is deleted and the month's Parquet is not written.

Usage::

    uv run python -m swissdelay.data.ingest 2025-08 2026-09        # ingest a range
    uv run python -m swissdelay.data.ingest 2025-08 --from-dir path/to/folder
    uv run python -m swissdelay.data.ingest 2025-10 --zip ~/Downloads/ist-daten-v2-2025-10.zip

See docs/data_dictionary.md for the column meanings.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import re
import shutil
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import requests
from tqdm import tqdm

from swissdelay import config

log = logging.getLogger("swissdelay.ingest")

URL_TEMPLATE = (
    "https://archive.opentransportdata.swiss/istdaten/{year}/ist-daten-v2-{year}-{month:02d}.zip"
)
MANIFEST_PATH = config.DATA / "dataset_manifest.json"
SCHEMA_VERSION = "v2"

DAY_FMT = "%d.%m.%Y"
PLANNED_FMT = "%d.%m.%Y %H:%M"  # planned times: minute precision
ACTUAL_FMT = "%d.%m.%Y %H:%M:%S"  # actual / forecast times: second precision

TRAIN_FILTER = "upper(PRODUKT_ID) = 'ZUG'"

# (German source column, English name, DuckDB expression on the raw text column)
COLUMNS: list[tuple[str, str, str]] = [
    ("BETRIEBSTAG", "operating_day", f"try_strptime(BETRIEBSTAG, '{DAY_FMT}')::DATE"),
    ("FAHRT_BEZEICHNER", "trip_id", "FAHRT_BEZEICHNER"),
    ("BETREIBER_ID", "operator_id", "BETREIBER_ID"),
    ("BETREIBER_ABK", "operator_abbr", "BETREIBER_ABK"),
    ("BETREIBER_NAME", "operator_name", "BETREIBER_NAME"),
    ("PRODUKT_ID", "transport_mode", "PRODUKT_ID"),
    ("LINIEN_ID", "train_number", "LINIEN_ID"),
    ("LINIEN_TEXT", "line_name", "LINIEN_TEXT"),
    ("UMLAUF_ID", "rotation_id", "UMLAUF_ID"),
    ("VERKEHRSMITTEL_TEXT", "category", "VERKEHRSMITTEL_TEXT"),
    ("ZUSATZFAHRT_TF", "is_extra_trip", "lower(ZUSATZFAHRT_TF) = 'true'"),
    ("FAELLT_AUS_TF", "is_cancelled", "lower(FAELLT_AUS_TF) = 'true'"),
    ("BPUIC", "station_id", "TRY_CAST(BPUIC AS INTEGER)"),
    ("HALTESTELLEN_NAME", "station_name", "HALTESTELLEN_NAME"),
    ("ANKUNFTSZEIT", "arr_planned", f"try_strptime(ANKUNFTSZEIT, '{PLANNED_FMT}')"),
    ("AN_PROGNOSE", "arr_actual", f"try_strptime(AN_PROGNOSE, '{ACTUAL_FMT}')"),
    ("AN_PROGNOSE_STATUS", "arr_status", "AN_PROGNOSE_STATUS"),
    ("ABFAHRTSZEIT", "dep_planned", f"try_strptime(ABFAHRTSZEIT, '{PLANNED_FMT}')"),
    ("AB_PROGNOSE", "dep_actual", f"try_strptime(AB_PROGNOSE, '{ACTUAL_FMT}')"),
    ("AB_PROGNOSE_STATUS", "dep_status", "AB_PROGNOSE_STATUS"),
    ("DURCHFAHRT_TF", "is_pass_through", "lower(DURCHFAHRT_TF) = 'true'"),
]
RAW_COLUMNS = [raw for raw, _, _ in COLUMNS]
ENGLISH_COLUMNS = [eng for _, eng, _ in COLUMNS]

# Columns that exist in some files but are deliberately not kept.
# SLOID (added 2025-10-13) is a station-level Swiss Location ID that duplicates BPUIC
# (e.g. ch:1:sloid:3000 = 8503000 Zürich HB), so it adds no information.
KNOWN_DROPPED_COLUMNS = {"SLOID"}

# Raw columns whose non-empty values must all parse; a failure means the format changed.
PARSED_COLUMNS = [
    "BETRIEBSTAG",
    "BPUIC",
    "ANKUNFTSZEIT",
    "AN_PROGNOSE",
    "ABFAHRTSZEIT",
    "AB_PROGNOSE",
]

_SELECT_SQL = ",\n  ".join(f"{expr} AS {eng}" for _, eng, expr in COLUMNS)
_EXPR = {raw: (eng, expr) for raw, eng, expr in COLUMNS}
_CHECK_SQL = ",\n  ".join(
    f"count(*) FILTER (WHERE nullif(trim({raw}), '') IS NOT NULL AND ({_EXPR[raw][1]}) IS NULL)"
    f" AS bad_{_EXPR[raw][0]}"
    for raw in PARSED_COLUMNS
)
_ORDER_SQL = "operating_day, trip_id, coalesce(arr_planned, dep_planned)"
_DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


class IngestError(RuntimeError):
    """A validation check failed; raw inputs are kept for inspection."""


@dataclass
class DayStats:
    file: str
    file_date: str | None
    operating_days: list[str]
    n_raw_rows: int
    n_train_rows: int
    bad_values: dict[str, int]
    extra_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def month_range(start: str, end: str | None = None) -> list[str]:
    """Inclusive list of 'YYYY-MM' months."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, (end or start).split("-"))
    months = []
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def days_of_month(month: str) -> list[str]:
    y, m = map(int, month.split("-"))
    return [date(y, m, d).isoformat() for d in range(1, calendar.monthrange(y, m)[1] + 1)]


def archive_url(month: str) -> str:
    y, m = map(int, month.split("-"))
    return URL_TEMPLATE.format(year=y, month=m)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (via a .part file) and check it is a zip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for block in r.iter_content(chunk_size=8 << 20):
                f.write(block)
                bar.update(len(block))
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise IngestError(f"{url} did not return a zip file; check URL_TEMPLATE in ingest.py")
    tmp.replace(dest)


@contextmanager
def _daily_csvs(source: Path, work_dir: Path) -> Iterator[Iterator[tuple[Path, int]]]:
    """Yield (csv_path, n_raw_rows) one day at a time, from a zip or a folder.

    For a zip, each member is extracted to ``work_dir`` and deleted after use.
    """
    if source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() == ".csv")

        def from_dir() -> Iterator[tuple[Path, int]]:
            for p in files:
                yield p, _count_rows(p)

        yield from_dir()
        return

    with zipfile.ZipFile(source) as zf:
        members = sorted(
            m for m in zf.namelist() if m.lower().endswith(".csv") and "__MACOSX" not in m
        )

        def from_zip() -> Iterator[tuple[Path, int]]:
            for m in members:
                out = work_dir / Path(m).name
                n_lines = 0
                with zf.open(m) as src, open(out, "wb") as dst:
                    while block := src.read(8 << 20):
                        dst.write(block)
                        n_lines += block.count(b"\n")
                try:
                    yield out, max(n_lines - 1, 0)
                finally:
                    out.unlink(missing_ok=True)

        yield from_zip()


def _count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            n += block.count(b"\n")
    return max(n - 1, 0)


# --------------------------------------------------------------------------- core


def convert_day(
    con: duckdb.DuckDBPyConnection, csv_path: Path, out_path: Path, n_raw_rows: int = 0
) -> DayStats:
    """Filter one daily CSV to trains, rename to English and write Parquet."""
    con.execute(
        "CREATE OR REPLACE TEMP TABLE day_raw AS "
        f"SELECT * FROM read_csv('{_sql_path(csv_path)}', delim=';', header=true, "
        f"all_varchar=true) WHERE {TRAIN_FILTER}"
    )
    columns = [row[0] for row in con.execute("DESCRIBE day_raw").fetchall()]
    missing = [c for c in RAW_COLUMNS if c not in columns]
    if missing:
        raise IngestError(f"{csv_path.name}: missing columns {missing}")

    cursor = con.execute(f"SELECT count(*) AS n_train_rows, {_CHECK_SQL} FROM day_raw")
    names = [d[0] for d in cursor.description]
    checks = dict(zip(names, cursor.fetchone(), strict=True))
    n_train_rows = checks.pop("n_train_rows")

    days = con.execute(
        f"SELECT DISTINCT {_EXPR['BETRIEBSTAG'][1]} AS d FROM day_raw ORDER BY d"
    ).fetchall()

    con.execute(
        f"COPY (SELECT {_SELECT_SQL} FROM day_raw) "
        f"TO '{_sql_path(out_path)}' (FORMAT parquet, COMPRESSION zstd)"
    )
    match = _DATE_IN_NAME.search(csv_path.name)
    return DayStats(
        file=csv_path.name,
        file_date=match.group(1) if match else None,
        operating_days=[str(d[0]) for d in days],
        n_raw_rows=n_raw_rows,
        n_train_rows=n_train_rows,
        bad_values={k: v for k, v in checks.items() if v},
        extra_columns=[
            c for c in columns if c not in RAW_COLUMNS and c not in KNOWN_DROPPED_COLUMNS
        ],
    )


def validate(stats: list[DayStats], month: str, require_full_month: bool = True) -> list[str]:
    """Return a list of problems; empty means the month is good."""
    errors: list[str] = []
    seen: list[str] = []
    for s in stats:
        if s.bad_values:
            errors.append(f"{s.file}: unparseable values {s.bad_values}")
        if s.n_train_rows == 0:
            errors.append(f"{s.file}: no train rows")
        if s.file_date and s.operating_days != [s.file_date]:
            errors.append(f"{s.file}: operating days {s.operating_days} != file date {s.file_date}")
        if s.extra_columns:
            log.warning("%s: unexpected extra columns %s (ignored)", s.file, s.extra_columns)
        seen += s.operating_days
    outside = sorted(d for d in set(seen) if not d.startswith(month))
    if outside:
        errors.append(f"days outside {month}: {outside}")
    dupes = sorted({d for d in seen if seen.count(d) > 1})
    if dupes:
        errors.append(f"days present in several files: {dupes}")
    if require_full_month:
        missing = sorted(set(days_of_month(month)) - set(seen))
        if missing:
            errors.append(f"missing days: {missing}")
    return errors


def ingest_source(
    source: Path,
    month: str,
    out_path: Path,
    work_dir: Path,
    require_full_month: bool = True,
) -> dict:
    """Convert one month's zip or folder of daily CSVs into a single Parquet file."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    con = duckdb.connect()
    stats: list[DayStats] = []

    with _daily_csvs(source, work_dir) as days:
        for csv_path, n_raw in tqdm(days, desc=f"{month} days", unit="day"):
            stats.append(convert_day(con, csv_path, work_dir / f"{csv_path.stem}.parquet", n_raw))
    if not stats:
        raise IngestError(f"no CSV files found in {source}")

    errors = validate(stats, month, require_full_month)
    if errors:
        raise IngestError(f"{month} failed validation:\n  - " + "\n  - ".join(errors))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{_sql_path(work_dir)}/*.parquet') "
        f"ORDER BY {_ORDER_SQL}) TO '{_sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)"
    )
    n_written = con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(tmp)}')").fetchone()[0]
    n_expected = sum(s.n_train_rows for s in stats)
    if n_written != n_expected:
        tmp.unlink()
        raise IngestError(f"{month}: wrote {n_written} rows, expected {n_expected}")
    tmp.replace(out_path)
    con.close()
    shutil.rmtree(work_dir)

    return {
        "month": month,
        "n_days": len({d for s in stats for d in s.operating_days}),
        "n_raw_rows": sum(s.n_raw_rows for s in stats),
        "n_train_rows": n_expected,
        "train_rows_per_day": {d: s.n_train_rows for s in stats for d in s.operating_days},
    }


# --------------------------------------------------------------------------- manifest


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"schema_version": SCHEMA_VERSION, "filter": TRAIN_FILTER, "months": {}}


def save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    manifest["months"] = dict(sorted(manifest["months"].items()))
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- driver


def find_raw(month: str) -> Path | None:
    """Raw input already in data/raw: the unzipped folder first, then the zip."""
    for candidate in (
        config.RAW / f"ist-daten-v2-{month}",
        config.RAW / f"ist-daten-v2-{month}.zip",
    ):
        if candidate.exists():
            return candidate
    return None


def ingest_month(
    month: str,
    zip_path: Path | None = None,
    from_dir: Path | None = None,
    keep_raw: bool = False,
    force: bool = False,
    require_full_month: bool = True,
) -> dict | None:
    out_path = config.INTERIM / f"trains_{month}.parquet"
    manifest = load_manifest()
    if out_path.exists() and not force:
        if month not in manifest["months"]:
            log.warning("%s exists but is not in the manifest; rerun with --force", out_path.name)
        else:
            log.info("%s already ingested, skipping", month)
        return None

    source = from_dir or zip_path or find_raw(month)
    if source is None:
        source = config.RAW / f"ist-daten-v2-{month}.zip"
        download(archive_url(month), source)

    entry: dict = {"input": source.name, "source_url": archive_url(month)}
    if source.is_file():
        entry["zip_sha256"] = sha256_file(source)
        entry["zip_bytes"] = source.stat().st_size

    summary = ingest_source(
        source, month, out_path, config.RAW / f"_work_{month}", require_full_month
    )
    entry |= summary
    entry |= {
        "parquet": str(out_path.relative_to(config.ROOT)),
        "parquet_sha256": sha256_file(out_path),
        "parquet_bytes": out_path.stat().st_size,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest["months"][month] = entry
    save_manifest(manifest)

    if not keep_raw and source.resolve().is_relative_to(config.RAW.resolve()):
        log.info("Deleting raw input %s", source)
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
    log.info(
        "%s: %d days, %d train rows, %.1f MB",
        month,
        entry["n_days"],
        entry["n_train_rows"],
        entry["parquet_bytes"] / 1e6,
    )
    return entry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("start", help="first month, YYYY-MM")
    parser.add_argument("end", nargs="?", help="last month, YYYY-MM (default: start)")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--zip", type=Path, help="use this local zip (single month)")
    src.add_argument("--from-dir", type=Path, help="use this folder of daily CSVs (single month)")
    parser.add_argument("--keep-raw", action="store_true", help="do not delete raw inputs")
    parser.add_argument("--force", action="store_true", help="re-ingest existing months")
    parser.add_argument(
        "--allow-partial-month", action="store_true", help="do not require every calendar day"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    months = month_range(args.start, args.end)
    if (args.zip or args.from_dir) and len(months) > 1:
        parser.error("--zip / --from-dir take a single month")

    failed = []
    for month in months:
        try:
            ingest_month(
                month,
                zip_path=args.zip,
                from_dir=args.from_dir,
                keep_raw=args.keep_raw,
                force=args.force,
                require_full_month=not args.allow_partial_month,
            )
        except (IngestError, requests.RequestException) as exc:
            log.error("%s", exc)
            failed.append(month)
    if failed:
        raise SystemExit(f"Failed months (raw inputs kept): {', '.join(failed)}")


if __name__ == "__main__":
    main()
