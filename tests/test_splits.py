import duckdb
import pytest

from swissdelay.evaluation.splits import split_of, split_sql


@pytest.mark.parametrize(
    ("day", "split"),
    [
        ("2025-08-01", "train"),
        ("2026-04-30", "train"),
        ("2026-05-01", "valid"),
        ("2026-05-31", "valid"),
        ("2026-06-01", "test"),
        ("2026-06-30", "test"),
        ("2026-07-01", "production"),
        ("2026-09-30", "production"),
    ],
)
def test_split_of_and_sql_agree(day, split):
    assert split_of(day) == split
    expr = split_sql(f"DATE '{day}'")
    assert duckdb.sql(f"SELECT {expr}").fetchone()[0] == split


def test_outside_period():
    with pytest.raises(ValueError):
        split_of("2025-07-31")


def test_timestamps_are_handled():
    import pandas as pd

    assert split_of(pd.Timestamp("2026-04-30")) == "train"
    assert split_of(pd.Timestamp("2026-05-31 23:59")) == "valid"
