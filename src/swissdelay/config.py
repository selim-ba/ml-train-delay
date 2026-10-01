"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"  # untouched downloads (monthly archives, daily CSVs)
INTERIM = DATA / "interim"  # filtered train-only Parquet
PROCESSED = DATA / "processed"  # prediction points, features, labels
EXTERNAL = DATA / "external"  # GTFS stops.txt, station metadata
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Scope (see docs/roadmap.md)
TRAIN_CATEGORIES = ("IC", "IR", "RE", "EC")
PERIOD_START = "2025-08-01"  # first full month after the v2 format change
PERIOD_END = "2026-09-30"
HORIZONS_MIN = (15, 30, 60)
REPORTING_LAG_MIN = 2

# Time-based splits (inclusive end dates)
TRAIN_END = "2026-04-30"  # train:      Aug 2025 - Apr 2026
VALID_END = "2026-05-31"  # validation: May 2026 (model selection, Day-4 gate)
TEST_END = "2026-06-30"  # test:       Jun 2026 (headline benchmark)
# Jul - Sep 2026: simulated production, replayed day by day, never used for development

# Delay severity buckets (minutes)
SEVERITY_BINS = (1, 3, 10)
