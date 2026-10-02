"""Time-based splits: one definition used everywhere."""

from datetime import date

from swissdelay import config

SPLITS = ("train", "valid", "test", "production")


def split_of(day: date | str) -> str:
    """Split of an operating day (ISO dates compare correctly as strings)."""
    d = (day.isoformat() if isinstance(day, date) else str(day))[:10]  # also for Timestamps
    if d < config.PERIOD_START or d > config.PERIOD_END:
        raise ValueError(f"{d} is outside the study period")
    if d <= config.TRAIN_END:
        return "train"
    if d <= config.VALID_END:
        return "valid"
    if d <= config.TEST_END:
        return "test"
    return "production"


def split_sql(column: str = "operating_day") -> str:
    """SQL CASE expression giving the split of a DATE column."""
    return (
        f"CASE WHEN {column} <= DATE '{config.TRAIN_END}' THEN 'train' "
        f"WHEN {column} <= DATE '{config.VALID_END}' THEN 'valid' "
        f"WHEN {column} <= DATE '{config.TEST_END}' THEN 'test' "
        f"ELSE 'production' END"
    )
