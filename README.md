# SwissDelay

**How much do a train's own history and the state of the network improve short-term delay forecasts over simple baselines?**

SwissDelay is a leakage-safe benchmark of train-delay propagation on Swiss operating data. At every measured departure of an IC, IR, RE or EC train, it predicts how the delay will change **15, 30 and 60 minutes ahead**, and compares persistence, historical means, Ridge, XGBoost, a sequence model and a GNN.

> **Status:** 🚧 Day 2 done: labels and baselines. Next: Ridge and XGBoost (Day 3). Results, figures and the project website will be linked here as they land.

## Results

Validation month (May 2026), common subset of 183 k prediction points labelled at all three horizons. MAE in minutes of the predicted change in delay; 95 % confidence intervals from a bootstrap over whole days.

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Persistence (delay stays the same) | 1.33 | 1.47 | 1.64 |
| Constant offset (training median Δd) | 1.06 | 1.25 | 1.44 |
| **Historical median** (line × station × hour × day type) | **0.88** [0.86, 0.91] | **1.07** [1.04, 1.11] | **1.28** [1.22, 1.33] |

- Δd has a structural offset of ≈ −0.8 min (trains tend to arrive slightly early), so the **constant offset** is the reference a model must beat to show real skill.
- The historical median beats it significantly at every horizon (−0.16 to −0.18 min). It ignores the current delay, where most of the remaining error lies (trains already > 10 min late: 2.9 / 4.5 / 6.3 min).
- Details: [`notebooks/02-labels-baselines.ipynb`](notebooks/02-labels-baselines.ipynb).

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

A second command rebuilds the journeys of in-scope trains, applying the data-quality decisions below, and writes `data/processed/journeys_YYYY-MM.parquet`:

```bash
uv run python -m swissdelay.data.journeys
uv run python -m swissdelay.features.labels      # horizon labels (Δd at 15 / 30 / 60 min)
uv run python -m swissdelay.models.baselines     # disruption days, baselines, validation / test tables
```

### Data quality

The audit is in [`notebooks/01-data-quality.ipynb`](notebooks/01-data-quality.ipynb) (section 9 summarises it).

| | |
|---|---|
| Dataset | 426 days (1 Aug 2025 – 30 Sep 2026), 76.0 M train rows, no missing day |
| Scope | 1.16 M IC / IR / RE / EC train runs (≈ 2,700 per day) |
| Measured (`REAL`) times | 95.7 % at Swiss stops, 0.7 % abroad; no feed outage |
| Delays | median < 1 min (IC / IR / RE), 1.5 min (EC); > 3 min late: 8–10 % (IC / IR / RE), 29 % (EC) |
| Cancellations | 1.0 % of runs fully, 6.1 % partially; twice as many at weekends (engineering works) |
| Prediction points | 7.38 M eligible measured departures in 1.11 M journeys (before horizon labels) |

Main cleaning rules, all implemented in `swissdelay.data.journeys`:

- **Swiss stops only**: foreign stops have almost no measured times. Cross-border runs keep their Swiss section, flagged `enters_from_abroad`.
- **Keep every row, but only `REAL` times count**; a `REAL` time giving a delay outside [−5 min, +6 h] is a data error and is treated as missing.
- **Eligible prediction points and targets** exclude three poorly measured operators (FART, DB, DB Regio), stations with < 80 % measured departures over the training months, pass-through and cancelled stops, and anomalous journeys (0.06 %).

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
├── notebooks/          # 01 data quality, 02 labels and baselines
├── reports/figures/    # generated figures
├── site/               # minimal results website
├── src/swissdelay/
│   ├── config.py       # paths, scope, horizons, splits
│   ├── data/           # ingest.py (download → Parquet), journeys.py (journey reconstruction)
│   ├── features/       # labels.py (horizon labels), features
│   ├── models/         # baselines.py, then XGBoost, sequence, GNN
│   └── evaluation/     # splits, disruption days, metrics with day-bootstrap CIs
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
- [x] Day 1: train filtering, data quality report, journey reconstruction (`swissdelay.data.journeys`)
- [x] Day 2: labels, coverage, persistence / offset / historical baselines (`swissdelay.features.labels`, `swissdelay.models.baselines`)
- [ ] Day 3: Ridge and train-only XGBoost
- [ ] Day 4: network features, leakage tests, **gate** for the GNN
- [ ] Day 5: sequence model
- [ ] Day 6–7: GNN (if the gate passes)
- [ ] Day 8: final evaluation
- [ ] Day 9: nightly replay pipeline, API, dashboard
- [ ] Day 10: write-up and website

## License

Code is MIT-licensed. The data comes from [opentransportdata.swiss](https://opentransportdata.swiss) and is subject to its terms of use.
