# SwissDelay

**Can we predict how a train's delay will change?** · [Live dashboard](https://ml-train-delay.streamlit.app/)

Your train leaves its station 4 minutes late. Will you still be late at your stop in half an hour? Will the train make up time, or lose more? SwissDelay predicts, for every long-distance train in Switzerland (IC, IR, RE and EC), **how its delay will change over the next 15, 30 and 60 minutes**, and gives a likely range around each prediction. The models were trained on 14 months of open Swiss railway data, compared fairly against simple rules, tested once on a month they had never seen, and then run for three months as if in service.

## In short

- **Data:** 14 months (August 2025 – September 2026) of the official open record of every train stop in Switzerland: 76 million train records, 1.1 million long-distance train runs, 7.4 million examples to learn from.
- **Result:** 15 minutes ahead, the deployed model is typically **0.8 minutes off**. That is **19 % less error than the best simple rule** (*"the train will do what trains usually do on this line, at this station and hour"*), and 42 % less than assuming the delay stays the same. The gain over the best simple rule is the same 30 and 60 minutes ahead (−19 %).
- **Likely range:** each prediction comes with a range built so that the real delay falls inside it for 8 trains out of 10. Over three months of simulated production, **80 %** did.
- **Reliability over time:** run day by day from July to September 2026 **without retraining**, the model kept the same advantage (−21 %), with no sign of ageing.
- **Research:** a graph neural network (a graph transformer reading the railway map around each train) tied the deployed model; averaging the two was only 1 % better, so the simpler, faster model stays.

**Dashboard: [ml-train-delay.streamlit.app](https://ml-train-delay.streamlit.app/)**. It explains the project for non-specialists, shows every result, and lets you see the model's predictions on real trains. You can also run it locally with the live model (see [Run it](#run-it)).

## The methods and their tags

Every method has a short tag, used in the tables below and in the dashboard.

| Tag | Method | What it uses to predict the change in delay |
|---|---|---|
| **R1** | Persistence | nothing: it predicts no change |
| **R2** | Typical change | the median change of all training examples (about −0.8 min), the same for every train |
| **R3** | Historical median (best simple rule) | the median change of past trains on the same line, at the same station, horizon, hour and type of day |
| **M1** | XGBoost (`xgb_full`) | 32 inputs on the train itself: current delay, delays at its last stops, spare time in the timetable, stops ahead, hour, day, type of train, operator, and R3 |
| **M2** | XGBoost with network data (`xgb_full_network_plus`), **deployed** | M1's inputs plus 33 inputs on the other trains around it: their delays at its stations and on its route, the train just ahead, 10 / 30 / 60-minute trends (65 inputs) |
| P10 / P90 | Quantile XGBoost | M2's inputs, trained to predict the low and high ends of the likely range |
| **G1–G4** | Graph transformer variants (research, 15 min) | M1's inputs plus a subgraph of the stations around the train: G1 without the stations (control), G2 with a random map, G3 with the real map, G4 = G3 retrained on all training months |
| **A** | Average of M2 and G4 (research) | the mean of the two predictions |

## Results

### Test month (June 2026)

The test month was used **once**, after every choice had been made on the validation month (May). All methods are scored on the same 175,420 departures per horizon. The metric is the **MAE** (mean absolute error): the average gap, in minutes, between the predicted and the real change in delay. Lower is better. Brackets: 95 % confidence intervals from a bootstrap over whole days.

| | 15 min | 30 min | 60 min |
|---|---|---|---|
| R1 · Persistence | 1.40 | 1.58 | 1.80 |
| R2 · Typical change | 1.16 | 1.38 | 1.65 |
| R3 · Historical median | 0.99 | 1.23 | 1.49 |
| M1 · XGBoost | 0.83 | 1.02 | 1.22 |
| **M2 · XGBoost with network data** (deployed) | **0.805** [0.774, 0.834] | **0.990** [0.943, 1.032] | **1.200** [1.134, 1.264] |
| A · Average of M2 and G4 (research) | 0.797 (−1.0 %) | — | — |

**Likely range (P10–P90).** Ranges calibrated once on April covered **77 %** of real delays in June (target 80 %); with margins recomputed every night from the last 14 days, they covered **80 %** in simulated production.

### Simulated production (July – September 2026)

The frozen M2 was replayed day by day for 92 days (2.8 M predictions), exactly as a nightly job would run it: predict, score the previous day, update the likely ranges, check for drift and raise alerts.

| | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE of M2, normal days (79) | 0.690 | 0.865 | 1.056 |
| MAE of M2, disrupted days (13) | 0.780 | 1.002 | 1.253 |
| M2 vs R3, normal / disrupted days | −22 / −20 % | −21 / −20 % | −21 / −20 % |
| Real delays inside the likely range, normal / disrupted days | 80 / 79 % | 80 / 79 % | 80 / 79 % |

A **disrupted day** is a day with unusually high delays and cancellations: its score, combining average delay and share of cancelled trains, exceeds the level of the worst 5 % of training days.

### What we learned

- **M2 cuts the error by 19 % vs R3** and by 33–42 % vs R1, at every horizon. The validation month gave the same picture (−21 %), so the model choice did not overfit it. The gain is largest for trains 3–10 minutes late (−31 to −42 %).
- **Knowing what happens around the train helps, especially when things go wrong.** The 33 network inputs (M1 → M2), all measured from events at least 2 minutes old, cut the error by 2.8 / 2.7 / 1.8 %; on the 13 disrupted days of June, by 3.0 / 3.1 / 2.6 %, against 2.6 / 2.3 / 1.1 % on normal days.
- **An explicit graph model does not beat well-designed network inputs on its own.** The graph transformer (G4) beat M2 on May (−0.5 %) but tied it on June. A random map (G2) keeps 90 % of the real map's gain over G1: what helps is station-by-station detail, not the shape of the network. Averaging (A) helps a little (−1.3 % on May, −1.0 % on June), below the 2 % rule set beforehand for running a second, much heavier model.
- **What the model relies on** (XGBoost total-gain importance): R3 matters most 15 minutes ahead; the train's current delay matters more and more further ahead.
- **The train's history before its current stop adds nothing** once its current delay, arrival delay and dwell are known; **spare time in the timetable** adds a small, significant gain.
- **A linear model (Ridge) with the same inputs barely beats R3:** recovery is non-linear (it depends on how late the train already is).
- **Ranges calibrated on a calm month under-cover in a disrupted one:** 77 % on June, with the misses mostly on the high side (trains ending later than P90). Separate nightly margins for each end fixed it.
- **No ageing without retraining:** the gain over R3 is flat across July, August and September, and no input drifted from training (PSI ≤ 0.08 every day). 5 alerts in 92 days, all on genuinely bad days.
- The change in delay has a structural offset of about −0.8 min (trains tend to arrive slightly early); R2 captures it, so it is not counted as skill.

Analysis notebooks, for details: [data quality](notebooks/01-data-quality.ipynb), [labels and baselines](notebooks/02-labels-baselines.ipynb), [Ridge and XGBoost](notebooks/03-tabular-models.ipynb), [network inputs](notebooks/04-network-gate.ipynb), [graph transformer](notebooks/05-graph-transformer.ipynb), [test results](notebooks/06-test-results.ipynb), [simulated production](notebooks/07-production-replay.ipynb).

## How it works

- **Target:** the change in delay `Δd = d(t+h) − d(t)` between the departure now (`t`) and the target stop, so R1 is `Δd = 0`. Knowing the current delay, an error on the change is also the error on the delay at the stop.
- **Target stop:** for a horizon `h` of 15, 30 or 60 minutes, the first stop the train is **scheduled** to reach at least `h` after leaving. It is never chosen from actual times.
- **No peeking at the future:** every input uses only events measured at or before `t − 2 min`, because train positions reach the system with a small lag. Automated tests enforce this.
- **Splits by date only**, so a model is always judged on days after the ones it learned from:

| Period | Role |
|---|---|
| Aug 2025 – Apr 2026 | **Training**: the models learn |
| May 2026 | **Validation**: models and settings are compared, one is chosen |
| Jun 2026 | **Test**: the chosen model is scored once, nothing changed afterwards |
| Jul – Sep 2026 | **Simulated production**: replayed day by day, without retraining |

- **Metric:** MAE in minutes; every comparison is paired (same departures) with a bootstrap confidence interval over whole days.
- **Likely range:** two quantile XGBoost models give P10 and P90; each end is then widened by a split-conformal margin, per current-delay bucket (on time, 1–3, 3–10, > 10 min late), so that 10 % of real delays fall below and 10 % above. In production, the margins are recomputed every night from the last 14 days.
- **Monitoring (per horizon, every day):** an alert is raised when the day's MAE exceeds 1.25 × the median of the previous 14 days, when fewer than 70 % of real delays fall inside the range, or when the drift score (PSI) of the current delays, hours of departure or traffic at stations exceeds 0.25.

## Data

| Source | Use |
|---|---|
| [Swiss actual data (Ist-Daten v2)](https://data.opentransportdata.swiss/fr/dataset/ist-daten-v2) | Planned and actual times per stop, recent days |
| [opentransportdata.swiss archive](https://archive.opentransportdata.swiss) | Monthly archives for past months |
| GTFS `stops.txt` | Station coordinates, for maps only |

- **Scope:** IC, IR, RE and EC trains at every Swiss station they serve.
- **Period:** August 2025 to September 2026, v2 format only (after the July 2025 format change). About 16 GB of raw files per month, all public transport; each month becomes a ~60 MB Parquet file of train rows.
- **Kept at ingestion:** every train row (S-Bahn included) and every column, renamed to English; the IC/IR/RE/EC scope is applied in code. See [`docs/data_dictionary.md`](docs/data_dictionary.md).
- **Ground truth:** only times with status `REAL` (really measured, not estimated) count as observations or labels.

| Data quality | |
|---|---|
| Dataset | 426 days (1 Aug 2025 – 30 Sep 2026), 76.0 M train rows, no missing day |
| Scope | 1.16 M IC / IR / RE / EC train runs (≈ 2,700 per day) |
| Measured (`REAL`) times | 95.7 % at Swiss stops, 0.7 % abroad; no feed outage |
| Delays | median < 1 min (IC / IR / RE), 1.5 min (EC); > 3 min late: 8–10 % (IC / IR / RE), 29 % (EC) |
| Cancellations | 1.0 % of runs fully, 6.1 % partially; twice as many at weekends (engineering works) |
| Prediction points | 7.38 M eligible measured departures in 1.11 M journeys |

Main cleaning rules (all in `swissdelay.data.journeys`):

- **Swiss stops only:** foreign stops have almost no measured times. Cross-border runs keep their Swiss section, flagged `enters_from_abroad`.
- **Every row is kept, but only `REAL` times count;** a `REAL` time giving a delay outside [−5 min, +6 h] is a data error and is treated as missing.
- **Eligible prediction points and targets** exclude three poorly measured operators (FART, DB, DB Regio), stations with < 80 % measured departures over the training months, pass-through and cancelled stops, and anomalous journeys (0.06 %).

## Run it

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed). On macOS, XGBoost needs OpenMP: `brew install libomp`.

```bash
git clone https://github.com/selim-ba/ml-train-delay.git
cd ml-train-delay
uv sync --all-groups    # core, dev tools, PyTorch, serving stack
make test
```

The raw data is not committed, so the API and dashboard need the pipeline below to have been run once (it writes the model to `models/champion/` and the production metrics to `data/processed/replay/`).

### 1. Build the dataset

```bash
uv run python -m swissdelay.data.ingest 2025-08 2026-09   # download, keep train rows, validate → data/interim/
uv run python -m swissdelay.data.journeys                 # cleaned journeys → data/processed/journeys_YYYY-MM.parquet
uv run python -m swissdelay.features.labels               # change in delay at 15 / 30 / 60 min
uv run python -m swissdelay.models.baselines              # disrupted days, R1–R3
uv run python -m swissdelay.features.build                # inputs on the train itself
uv run python -m swissdelay.features.network              # network inputs (events before t − 2 min)
```

Each month's source URL, checksums and row counts are recorded in [`data/dataset_manifest.json`](data/dataset_manifest.json).

### 2. Train and evaluate

```bash
uv run python -m swissdelay.models.tabular                # Ridge + M1; --sets all --sample 0.3 for the ablations
uv run python -m swissdelay.models.tabular --models xgb --sets full full_network full_network_plus
uv run python -m swissdelay.models.quantile               # P10 / P90 with conformal calibration
uv run python -m swissdelay.features.graph                # subgraphs for the graph transformer (15 min)
uv run python -m swissdelay.models.graph_transformer --variant graph --refit   # also: nograph, random
```

Add `--test` to score the test month (once, at the end).

### 3. Deploy and replay production

```bash
uv run python -m swissdelay.models.registry               # export M2 to models/champion/ (verified)
uv run python -m swissdelay.pipeline.replay               # Jul – Sep 2026 day by day → reports/daily_metrics.csv
uv run python -m swissdelay.serve.example                 # dashboard data bundle: example trains + their predictions
```

### 4. Serve

```bash
make api          # prediction API on http://localhost:8000 (interactive docs at /docs)
make dashboard    # dashboard on http://localhost:8501, in a second terminal
# or both in Docker:
docker compose up --build
```

Example request: `curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @reports/example_request.json | python3 -m json.tool`.

| Command | What it does |
|---|---|
| `make install` / `make install-all` | install the core and dev tools / plus PyTorch and the serving stack |
| `make lint` | ruff check + format check |
| `make format` | auto-fix lint issues and format code |
| `make test` | run pytest (the PyTorch tests run in a separate process) |
| `make api` / `make dashboard` | start the API / the dashboard |
| `make docker` | build and start both in Docker |

## Project structure

```
train-delay/
├── data/               # raw / interim / processed / external (git-ignored)
├── docs/               # roadmap, data dictionary
├── models/champion/    # exported M2 + P10 / P90 models, manifest, nightly calibration (git-ignored)
├── notebooks/          # 01 data quality … 07 simulated production
├── reports/            # result tables, daily metrics, example requests
├── src/swissdelay/
│   ├── config.py       # paths, scope, horizons, splits
│   ├── data/           # ingest.py (download → Parquet), journeys.py (journey reconstruction)
│   ├── features/       # labels.py, build.py (train inputs), network.py (network inputs), graph.py (subgraphs)
│   ├── models/         # baselines.py (R1–R3), tabular.py (Ridge, M1, M2), quantile.py, registry.py, graph_transformer.py
│   ├── evaluation/     # splits, disrupted days, metrics with day-bootstrap CIs
│   ├── pipeline/       # replay.py (simulated production: daily metrics, drift, alerts)
│   └── serve/          # api.py (FastAPI), dashboard.py (Streamlit), example.py, showcase.json (data bundle)
├── tests/
├── Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml   # lint + tests on every push
```

## License

Code is MIT-licensed. The data comes from [opentransportdata.swiss](https://opentransportdata.swiss) and is subject to its terms of use.
