# SwissDelay

**How much do a train's own history and the state of the network improve short-term delay forecasts over simple baselines?**

SwissDelay is a leakage-safe benchmark of train-delay propagation on Swiss operating data. At every measured departure of an IC, IR, RE or EC train, it predicts how the delay will change **15, 30 and 60 minutes ahead**, and compares persistence, historical means, Ridge, XGBoost, a sequence model and a GNN.

> **Status:** 🚧 project setup. Results, figures and the project website will be linked here as they land.

## Results

_Coming soon._ The headline table will report MAE and P90 absolute error on a common subset of prediction points labelled at all three horizons, with day-clustered bootstrap confidence intervals.

## Data

| Source | Use |
|---|---|
| [Swiss actual data (Ist-Daten v2)](https://data.opentransportdata.swiss/fr/dataset/ist-daten-v2) | Planned and actual times per stop, recent days |
| [opentransportdata.swiss archive](https://archive.opentransportdata.swiss) | Monthly archives for past months and years |
| GTFS `stops.txt` | Station coordinates for maps only |

- **Scope:** IC, IR, RE and EC trains at every station they serve, nationwide.
- **Period:** August 2025 to September 2026, v2 format only (after the July 2025 format change).
- **Kept at ingestion:** every train row (S-Bahn included) and every column, renamed to English. The IC/IR/RE/EC scope is applied in code. See [`docs/data_dictionary.md`](docs/data_dictionary.md).
- **Ground truth:** only times with status `REAL` count as observed values or labels.

| Period | Role |
|---|---|
| Aug 2025 – Apr 2026 | Train |
| May 2026 | Validation |
| Jun 2026 | Test (headline results) |
| Jul – Sep 2026 | Simulated production: replayed day by day |
| Oct 2026 → | Nightly pipeline on new daily files |

### Building the dataset

Raw data is not committed. One command downloads each monthly archive, keeps the train rows, validates them, writes `data/interim/trains_YYYY-MM.parquet` and deletes the raw files:

```bash
uv run python -m swissdelay.data.ingest 2025-08 2026-09
```

Each month's source URL, checksums and row counts are recorded in [`data/dataset_manifest.json`](data/dataset_manifest.json). A month of raw CSVs (~16 GB) becomes a ~60 MB Parquet file.

## Method

- **Target:** the change in delay `Δd = d(t+h) − d(t)`, so persistence is `Δd = 0`.
- **Target stop:** the first downstream stop whose **scheduled** time is at least `h` after the prediction time. It is never chosen from actual times.
- **No leakage:** network features only use events measured at or before `t − 2 min`. Automated tests enforce this.
- **Splits:** by time only (see the table above).

The full plan is in [`docs/roadmap.md`](docs/roadmap.md).

## Project structure

```
train-delay/
├── data/               # raw / interim / processed / external (git-ignored)
├── docs/               # roadmap and design notes
├── notebooks/          # exploration (numbered, e.g. 01-pilot-eda.ipynb)
├── reports/figures/    # generated figures
├── site/               # minimal results website
├── src/swissdelay/
│   ├── config.py       # paths, scope, horizons, splits
│   ├── data/           # ingest.py (download → Parquet), journey reconstruction
│   ├── features/       # prediction points, labels, features
│   ├── models/         # baselines, XGBoost, sequence, GNN
│   └── evaluation/     # metrics, common subset, bootstrap CIs
└── tests/
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 itself if needed.

```bash
git clone https://github.com/selim-ba/ml-train-delay.git
cd ml-train-delay
uv sync                 # core + dev tools
uv sync --all-groups    # also PyTorch / PyG and the serving stack
make test
```

On macOS, XGBoost needs OpenMP: `brew install libomp`.

| Command | What it does |
|---|---|
| `make lint` | ruff check + format check |
| `make format` | auto-fix lint issues and format code |
| `make test` | run pytest |
| `make reproduce` | rebuild baselines and the report _(coming soon)_ |

## Roadmap

- [x] Project setup
- [x] Monthly ingestion to Parquet (`swissdelay.data.ingest`)
- [ ] Day 1: 7-day pilot, train filtering, journey reconstruction, data quality report
- [ ] Day 2: prediction points, labels, persistence and historical-mean baselines
- [ ] Day 3: Ridge and train-only XGBoost
- [ ] Day 4: network features, leakage tests, **gate** for the GNN
- [ ] Day 5: sequence model
- [ ] Day 6–7: GNN (if the gate passes)
- [ ] Day 8: final evaluation
- [ ] Day 9: nightly replay pipeline, API, dashboard
- [ ] Day 10: write-up and website

## License

Code is MIT-licensed. The data comes from [opentransportdata.swiss](https://opentransportdata.swiss) and is subject to its terms of use.
