"""Daily statistics and the definition of disruption days.

Each day gets a disruption score: the average of the z-scores of its mean departure
delay and of its share of runs with a cancelled stop. The z-scores and the threshold are
fitted on the **training days only**: a disruption day is a day whose score is above the
95th percentile of training days, i.e. "worse than 95 % of training days". Nothing is
learned from validation, test or production days, and each split gets as many disruption
days as it really has. It is a reporting stratum, not a model feature.

Output: ``data/processed/daily_stats.parquet``, one row per operating day.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from swissdelay import config
from swissdelay.evaluation.splits import split_of

DISRUPTION_QUANTILE = 0.95
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


def flag_disruption_days(
    stats: pd.DataFrame, reference: pd.Series, quantile: float = DISRUPTION_QUANTILE
) -> pd.DataFrame:
    """Add ``disruption_score``, ``disruption_threshold`` and ``is_disruption_day``.

    ``reference`` is a boolean mask of the days used to fit the z-scores and the
    threshold (the training days).
    """
    out = stats.copy()
    ref = out[reference]

    def z(col: str) -> pd.Series:
        return (out[col] - ref[col].mean()) / ref[col].std(ddof=0)

    out["disruption_score"] = (z("mean_delay_min") + z("cancelled_runs_share")) / 2
    threshold = float(out.loc[reference, "disruption_score"].quantile(quantile))
    out["disruption_threshold"] = threshold
    out["is_disruption_day"] = out["disruption_score"] > threshold
    return out


def build_daily_stats(
    con: duckdb.DuckDBPyConnection, sources: list[Path], out_path: Path = DAILY_STATS_PATH
) -> pd.DataFrame:
    source = "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in sources) + "]"
    stats = con.sql(daily_stats_sql(source)).df()
    stats["operating_day"] = pd.to_datetime(stats["operating_day"]).dt.date  # DATE in Parquet
    stats["split"] = stats["operating_day"].map(split_of)
    stats = flag_disruption_days(stats, reference=stats["split"] == "train")
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
    print(f"{len(flagged)} disruption days of {len(stats)} (threshold fitted on training days):")
    print(stats.groupby("split")["is_disruption_day"].agg(["sum", "count"]).to_string())
    cols = ["operating_day", "weekday", "split", "mean_delay_min", "cancelled_runs_share"]
    print(flagged[cols].to_string(index=False))


if __name__ == "__main__":
    main()
