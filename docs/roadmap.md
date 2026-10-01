# Roadmap (condensed)

Source: *SwissDelay — Merged Project Roadmap* (30 Sep 2026).

## Objective

Predict how a running train's delay will change over the next 15, 30 and 60 minutes, and measure which information improves that forecast: the train's own history, timetable slack, or the state of the network. "Explicit graph modelling adds nothing over engineered features" is a valid result.

## Scope and data

- **Trains:** IC, IR, RE and EC, nationwide, keeping whole trips and the full graph.
- **Period:** Aug 2025 – Sep 2026 (v2 format only). Updated 1 Oct 2026, was Nov 2025 – Jun 2026.
- **Source:** Ist-Daten (actual data). GTFS `stops.txt` for coordinates only.
- **Ground truth:** only `REAL` measured times count.
- **Station IDs:** `station_id` (BPUIC) is already station-level. The SLOID column (from Oct 2025) duplicates it and is dropped.

## Prediction task

1. **Prediction points:** every measured departure event.
2. **Current delay:** `d` = departure delay at that event.
3. **Target stop for horizon h:** the first downstream stop whose *scheduled* time ≥ t + h.
4. **Target:** `Δd = d(t+h) − d(t)`.
5. **No label:** the train terminates before h, is cancelled, or the target time isn't measured. These cases are counted, not silently dropped.

## Evaluation controls

| Risk | Control |
|---|---|
| Samples change with horizon | Main table on the common subset labelled at all horizons |
| Cancellations drop out | Separate outcome; coverage split normal vs disruption days |
| `REAL` not random | Coverage by operator and station in the quality report |
| Network-state leakage | Events measured ≤ t − lag (2 min); automated test |
| Stale GNN snapshots | Build graph state at each prediction time |
| CIs too narrow | Bootstrap by whole day (1,000 resamples) |

- **Splits:** train Aug 2025–Apr 2026, validate May, test June. Rolling 3-fold backtest for XGBoost.
- **Simulated production:** Jul–Sep 2026 replayed day by day through the nightly pipeline (daily metrics, drift, weekly retrain rule) before going live in October.
- **Metrics:** MAE (primary), P90 absolute error, quantile XGBoost P10/P50/P90 coverage.
- **Breakdowns:** severity (<1, 1–3, 3–10, >10 min), normal vs disruption days (worst 5 %), major hubs.

## Models

| Model | Train history | Slack | Network |
|---|---|---|---|
| Persistence | current delay | – | – |
| Historical mean Δd | – | implicit | – |
| Ridge | last stops | ✓ | – |
| XGBoost train-only | last 1/3/5 stops, slopes | ✓ | – |
| XGBoost + network | ✓ | ✓ | hand-made |
| Sequence (GRU/transformer, d=128) | full trajectory | ✓ | – |
| GNN (GraphSAGE, 2 layers) | ✓ | ✓ | learned |

**Gate (Day 4):** if XGBoost + network does not beat train-only XGBoost on validation (day-clustered CI excludes 0), the GNN becomes a short negative-result section.

## Production

A nightly GitHub Actions job replays yesterday's trains through the champion model and appends MAE, P90 and coverage to `daily_metrics`, with PSI drift monitoring. The project also ships a FastAPI service (`/health`, `/predict`, `/metrics/daily`) and a Streamlit dashboard. This is a replay of past days, not live inference.

## Never cut

Persistence and historical baselines, common-subset evaluation with coverage, leakage tests, day-clustered CIs, the nightly replay pipeline.
