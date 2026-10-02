"""Daily statistics and the definition of disruption days.

A disruption day is one of the worst 5 % of days of the study period by a combination of
mean delay and cancellations (equal weight, z-scores). It is a reporting stratum, not a
model feature, so it is computed on all days at once.

Output: ``data/processed/daily_stats.parquet``, one row per operating day.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd

from swissdelay import config
from swissdelay.evaluation.splits import split_of

DISRUPTION_SHARE = 0.05
DAILY_STATS_PATH = config.PROCESSED / "daily_stats.parquet"


def daily_stats_sql(source: str) -> str:
    """Per-day delay and cancellation statistics over eligible in-scope journeys."""
    return f"""
    SELECT operating_day,
           count(DISTINCT trip_key)                                           AS runs,
           count(*) FILTER (WHERE is_prediction_point)                        AS prediction_points,
           avg(dep_delay_min) FILTER (WHERE is_prediction_point)              AS mean_delay_min,
           quantile_cont(dep_delay_min, 0.9) FILTER (WHERE is_prediction_point)
                                                                              AS p90_delay_min,
           count(DISTINCT trip_key) FILTER (WHERE is_cancelled)
             / count(DISTINCT trip_key) AS cancelled_runs_share
    FROM read_parquet({source})
    WHERE is_valid_journey AND is_target_operator
    GROUP BY operating_day
    ORDER BY operating_day
    """


def flag_disruption_days(stats: pd.DataFrame, share: float = DISRUPTION_SHARE) -> pd.DataFrame:
    """Add ``disruption_score`` and ``is_disruption_day`` (top ``share`` of days)."""
    out = stats.copy()

    def z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std(ddof=0)

    out["disruption_score"] = (z(out["mean_delay_min"]) + z(out["cancelled_runs_share"])) / 2
    n_flag = math.ceil(share * len(out))
    top = out["disruption_score"].rank(ascending=False, method="first") <= n_flag
    out["is_disruption_day"] = top
    return out


def build_daily_stats(
    con: duckdb.DuckDBPyConnection, sources: list[Path], out_path: Path = DAILY_STATS_PATH
) -> pd.DataFrame:
    source = "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in sources) + "]"
    stats = con.sql(daily_stats_sql(source)).df()
    stats["operating_day"] = pd.to_datetime(stats["operating_day"]).dt.date  # DATE in Parquet
    stats = flag_disruption_days(stats)
    stats["split"] = stats["operating_day"].map(split_of)
    stats["weekday"] = pd.to_datetime(stats["operating_day"]).dt.day_name()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(out_path, index=False)
    return stats


def main() -> None:
    sources = sorted(config.PROCESSED.glob("journeys_*.parquet"))
    if not sources:
        raise SystemExit("No journeys: run swissdelay.data.journeys first")
    stats = build_daily_stats(duckdb.connect(), sources)
    flagged = stats[stats["is_disruption_day"]].sort_values("disruption_score", ascending=False)
    print(f"{len(flagged)} disruption days out of {len(stats)}:")
    cols = ["operating_day", "weekday", "split", "mean_delay_min", "cancelled_runs_share"]
    print(flagged[cols].to_string(index=False))


if __name__ == "__main__":
    main()
