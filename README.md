# SwissDelay

**How much do a train's own history and the state of the network improve short-term delay forecasts over simple baselines?**

SwissDelay is a leakage-safe benchmark of train-delay propagation on Swiss operating data. At every measured departure of an IC, IR, RE or EC train, it predicts how the delay will change **15, 30 and 60 minutes ahead**, and compares persistence, historical means, Ridge, XGBoost with and without network features, and a graph transformer.

> **Status:** 🚧 models and final evaluation done (test month June 2026). Next: the nightly replay of Jul – Sep 2026 as simulated production, the API and dashboard, and the results website.

## Results

**Test month (June 2026), used once after every choice was made on May.** Common subset of 175 k prediction points labelled at all three horizons. MAE in minutes of the predicted change in delay; 95 % confidence intervals from a bootstrap over whole days.

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Persistence (delay stays the same) | 1.40 | 1.58 | 1.80 |
| Constant offset (training median Δd) | 1.16 | 1.38 | 1.65 |
| Historical median (line × station × hour × day type) | 0.99 | 1.23 | 1.49 |
| XGBoost (train, timetable and context features) | 0.83 | 1.02 | 1.22 |
| **XGBoost + network features** (deployed) | **0.805** [0.774, 0.834] | **0.990** [0.943, 1.032] | **1.200** [1.134, 1.264] |
| Average of graph transformer and XGBoost (research) | 0.797 (−1.0 %) | — | — |

Prediction intervals (P10–P90, conformal calibration): **77 % coverage** on the test month (target 80 %).

- **XGBoost cuts the error by 19 % vs the historical median** and by 33–42 % vs persistence, at every horizon. The validation month (May) gave the same picture (−21 %), so the model choice did not overfit it. The gain is largest for trains 3–10 min late: −31 to −42 %.
- **The state of the network helps, and helps more when things go wrong.** 33 features on all trains measured before `t − 2 min` improve XGBoost by −2.8 / −2.7 / −1.8 %. Those features cover:
  - traffic and delays at the current, next and target stations;
  - delay gained on the route to the target;
  - the train just ahead;
  - 10 / 30 / 60-min trends.

  On the 13 disruption days of June the gain is −3.0 / −3.1 / −2.6 %, against −2.6 / −2.3 / −1.1 % on normal days.
- **An explicit graph model does not beat hand-made network features on its own.** A Graphormer-style transformer reads a subgraph per prediction: the stations from the current one to the target and their neighbours, with their recent traffic and delays.
  - At 15 min it beat XGBoost on May (−0.5 %) but tied it on June.
  - Averaging the two helps on both months (−1.3 % and −1.0 %).
  - Randomly rewired edges keep 90 % of the graph's gain: what helps is station-by-station detail, not the network's topology.
  - The gain is below the 2 % promotion rule, so XGBoost stays the deployed model.
- **The train's trajectory before its current stop adds nothing** once its current delay, arrival delay and dwell are known; **timetable slack** adds a small, significant gain.
- **Ridge, with the same features, barely beats the historical median**: the relationships are non-linear (recovery depends on how late the train already is).
- **Intervals under-cover in a disrupted month.** They are calibrated on April; on June 77 % of outcomes fall inside them (76 % on disruption days), and the misses are mostly trains ending later than P90.
- Δd has a structural offset of ≈ −0.8 min (trains tend to arrive slightly early); the constant offset captures it, so it is not counted as skill.
- Details: [`notebooks/02-labels-baselines.ipynb`](notebooks/02-labels-baselines.ipynb), [`notebooks/03-tabular-models.ipynb`](notebooks/03-tabular-models.ipynb), [`notebooks/04-network-gate.ipynb`](notebooks/04-network-gate.ipynb), [`notebooks/05-graph-transformer.ipynb`](notebooks/05-graph-transformer.ipynb), [`notebooks/06-test-results.ipynb`](notebooks/06-test-results.ipynb) (final results).

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
uv run python -m swissdelay.features.build       # model features (33 per point and horizon)
uv run python -m swissdelay.features.network     # network-state features (all trains, events before t − 2 min)
uv run python -m swissdelay.models.tabular       # Ridge + XGBoost; --sets all --sample 0.3 for the ablations
uv run python -m swissdelay.models.tabular --models xgb --sets full full_network full_network_plus
uv run python -m swissdelay.models.quantile      # P10 / P90 with conformal calibration
uv run python -m swissdelay.features.graph       # per-point subgraphs for the graph transformer (15 min)
uv sync --group dev --group dl                   # PyTorch
uv run python -m swissdelay.models.graph_transformer --variant graph --refit   # also: nograph, random
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
├── notebooks/          # 01 data quality, 02 labels and baselines, 03 Ridge and XGBoost, 04 network features, 05 graph transformer, 06 test results
├── reports/figures/    # generated figures
├── site/               # minimal results website
├── src/swissdelay/
│   ├── config.py       # paths, scope, horizons, splits
│   ├── data/           # ingest.py (download → Parquet), journeys.py (journey reconstruction)
│   ├── features/       # labels.py, build.py (model features), network.py (network state), graph.py (subgraphs)
│   ├── models/         # baselines.py, tabular.py (Ridge, XGBoost), quantile.py, graph_transformer.py
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
- [x] Data quality report and journey reconstruction (`swissdelay.data.journeys`)
- [x] Horizon labels, coverage and baselines: persistence, constant offset, historical median (`swissdelay.features.labels`, `swissdelay.models.baselines`)
- [x] Ridge and train-only XGBoost, with history and slack ablations (`swissdelay.models.tabular`)
- [x] Network features, leakage tests, gate for a graph model (`swissdelay.features.network`)
- [x] Graph transformer on per-point subgraphs, vs no-graph and random-graph controls, 15 min (`swissdelay.models.graph_transformer`)
- [x] Prediction intervals: quantile XGBoost with conformal calibration (`swissdelay.models.quantile`)
- [x] Final evaluation on the test month: severity and disruption breakdowns, confidence intervals
- [ ] Nightly replay pipeline, API, dashboard
- [ ] Write-up and results website

## License

Code is MIT-licensed. The data comes from [opentransportdata.swiss](https://opentransportdata.swiss) and is subject to its terms of use.
