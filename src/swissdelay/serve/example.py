"""Real trains from the production period, as API requests and as data for the dashboard.

- ``reports/example_request.json``: one train 3–8 min late, as a ``POST /predict`` body
  (with the realised changes in delay alongside, for comparison);
- ``src/swissdelay/serve/showcase.json`` (committed): the dashboard's data bundle:
  - a handful of real departures of September 2026, from on time to very late, each with
    its features, station names, planned times, what actually happened, and the answer of
    ``POST /predict`` (so the online dashboard works without the 700 MB of models);
  - the daily average delay over the whole period;
  - ``GET /health`` and ``GET /metrics/daily`` of the simulated production.

Needs the exported model (``models.registry``) and the replay outputs (``pipeline.replay``).

Usage::

    uv run python -m swissdelay.serve.example
    curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \\
         -d @reports/example_request.json | python3 -m json.tool
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from swissdelay import config
from swissdelay.evaluation.days import DAILY_STATS_PATH
from swissdelay.models import registry
from swissdelay.models import tabular as tb

OUT = config.REPORTS / "example_request.json"
SHOWCASE = Path(__file__).parent / "showcase.json"

# (name, current delay from, to, number of trains) for the dashboard's examples
SHOWCASE_BUCKETS = (("on time", -1, 1, 2), ("a little late", 1, 3, 2), ("late", 3, 10, 3),
                    ("very late", 10, 30, 1))  # fmt: skip
LABEL_COLUMNS = ["trip_key", "stop_seq", "horizon_min", "category", "line_name",
                 "train_number", "station_id", "dep_planned", "target_station_id",
                 "target_arr_planned", "target_delay_min"]  # fmt: skip


def _points(rows: pd.DataFrame, columns: list[str]) -> list[dict]:
    """``POST /predict`` points of one departure (one per horizon); NaN → null."""
    return [{"horizon_min": int(r["horizon_min"]),
             "features": json.loads(r[columns].drop("horizon_min").to_json())}
            for _, r in rows.sort_values("horizon_min").iterrows()]  # fmt: skip


def _complete(rows: pd.DataFrame) -> pd.Series:
    """Departures with a measured outcome at all three horizons."""
    return rows.groupby(["trip_key", "stop_seq"])["horizon_min"].transform("nunique") == 3


def build(day: str = "2026-09-30", seed: int = 0) -> dict:
    columns = registry.load().columns
    rows = tb.load(duckdb.connect(), f"operating_day = DATE '{day}' AND delta_min IS NOT NULL",
                   ["trip_key", "stop_seq", "delta_min", "station_id", *columns])  # fmt: skip
    late = rows["d0_min"].between(3, 8)
    keys = rows.loc[_complete(rows) & late, ["trip_key", "stop_seq"]].drop_duplicates()
    trip_key, stop_seq = keys.sample(1, random_state=seed).iloc[0]
    pick = rows[(rows["trip_key"] == trip_key) & (rows["stop_seq"] == stop_seq)]
    realised = zip(pick["horizon_min"], pick["delta_min"], strict=True)
    about = {"trip_key": trip_key, "stop_seq": int(stop_seq),
             "station_id": int(pick["station_id"].iloc[0]),
             "realised_delta_min": {int(h): round(float(d), 2) for h, d in realised}}  # fmt: skip
    return {"points": _points(pick, columns), "_about": about}


def _delay(v: float) -> str:
    if v >= 0.05:
        return f"{v:.1f} min late"
    return f"{-v:.1f} min early" if v <= -0.05 else "on time"


def _hhmm(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def showcase_trains(start: str = "2026-09-01", end: str = "2026-09-30",
                    seed: int = 0) -> list[dict]:  # fmt: skip
    """A few real departures, spread over current-delay buckets, days and lines."""
    con = duckdb.connect()
    columns = registry.load().columns
    where = f"operating_day BETWEEN DATE '{start}' AND DATE '{end}' AND delta_min IS NOT NULL"
    rows = tb.load(con, where, ["trip_key", "stop_seq", "delta_min", *columns])
    labels = con.sql(f"""
        SELECT {", ".join(LABEL_COLUMNS)}
        FROM read_parquet('{config.PROCESSED}/labels_????-??.parquet')
        WHERE {where} AND label_status = 'ok'
    """).df()  # fmt: skip
    names = con.sql(f"""
        SELECT station_id, mode(station_name) AS station_name
        FROM read_parquet('{config.PROCESSED}/journeys_????-??.parquet')
        WHERE operating_day BETWEEN DATE '{start}' AND DATE '{end}'
        GROUP BY station_id
    """).df().set_index("station_id")["station_name"]  # fmt: skip
    keys = ["trip_key", "stop_seq", "horizon_min"]
    labels = labels[[c for c in labels.columns if c in keys or c not in rows.columns]]
    rows = rows.merge(labels, on=keys, how="inner")
    rows = rows[_complete(rows) & rows["line_name"].notna()]
    rows = rows[pd.to_datetime(rows["dep_planned"]).dt.hour.between(6, 21)]

    departures = rows.drop_duplicates(["trip_key", "stop_seq"]).sample(frac=1, random_state=seed)
    chosen, used_days, used_lines = [], set(), set()
    for name, low, high, n in SHOWCASE_BUCKETS:
        pool = departures[departures["d0_min"].between(low, high, inclusive="left")]
        taken = 0
        for _, d in pool.iterrows():
            day, line = d["trip_key"].split("|")[0], f"{d['category']} {d['line_name']}"
            if day in used_days or line in used_lines:
                continue
            chosen.append((name, d["trip_key"], d["stop_seq"]))
            used_days.add(day)
            used_lines.add(line)
            taken += 1
            if taken == n:
                break

    out = []
    for bucket, trip_key, stop_seq in chosen:
        pick = rows[(rows["trip_key"] == trip_key) & (rows["stop_seq"] == stop_seq)]
        pick = pick.sort_values("horizon_min")
        first = pick.iloc[0]
        day = pd.Timestamp(trip_key.split("|")[0])
        d0 = float(first["d0_min"])
        origin = names.get(first["station_id"], str(first["station_id"]))
        train = f"{first['category']} {first['line_name']}"
        out.append({
            "label": f"{train} from {origin} at {_hhmm(first['dep_planned'])} on "
                     f"{day:%a %d %b}, {_delay(d0)}",
            "bucket": bucket,
            "day": f"{day:%Y-%m-%d}",
            "weekday": f"{day:%A}",
            "train": train,
            "train_number": str(first["train_number"]),
            "from": {"station": origin, "planned": _hhmm(first["dep_planned"]),
                     "delay_min": round(d0, 2)},
            "targets": [{
                "horizon_min": int(r["horizon_min"]),
                "station": names.get(r["target_station_id"], str(r["target_station_id"])),
                "planned": _hhmm(r["target_arr_planned"]),
                "actual_delay_min": round(float(r["target_delay_min"]), 2),
                "realised_delta_min": round(float(r["delta_min"]), 2),
            } for _, r in pick.iterrows()],
            "points": _points(pick, columns),
        })  # fmt: skip
    return out


def daily_delay() -> list[dict]:
    """Average delay and disruption flag of every day of the study period."""
    d = pd.read_parquet(DAILY_STATS_PATH)
    d["operating_day"] = pd.to_datetime(d["operating_day"]).dt.strftime("%Y-%m-%d")
    keep = ["operating_day", "split", "runs", "mean_delay_min", "cancelled_runs_share",
            "is_disruption_day"]  # fmt: skip
    return json.loads(d[keep].to_json(orient="records"))


def bundle(trains: list[dict]) -> dict:
    """The dashboard's data, with the API's own answers (same code path as the service)."""
    from fastapi.testclient import TestClient

    from swissdelay.serve.api import create_app

    client = TestClient(create_app())

    def call(method: str, path: str, **kwargs):
        r = getattr(client, method)(path, **kwargs)
        r.raise_for_status()
        return r.json()

    for t in trains:
        t["predictions"] = call("post", "/predict", json={"points": t["points"]})
    health = {k: v for k, v in call("get", "/health").items() if k != "features"}
    return {"trains": trains, "daily": daily_delay(), "health": health,
            "metrics": call("get", "/metrics/daily")}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--day", default="2026-09-30")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    req = build(args.day, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(req, indent=1))
    print(f"Wrote {OUT}: {req['_about']}")
    trains = showcase_trains(seed=args.seed)
    SHOWCASE.write_text(json.dumps(bundle(trains), indent=1))
    print(f"Wrote {SHOWCASE}: {len(trains)} trains")
    for t in trains:
        print(f"  [{t['bucket']}] {t['label']}")


if __name__ == "__main__":
    main()
