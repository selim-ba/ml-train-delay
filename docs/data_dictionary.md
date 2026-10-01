# Data dictionary

Source: Ist-Daten v2 (opentransportdata.swiss), one CSV per operating day, `;`-separated.
Cleaned Parquet files in `data/interim/` use the English names below.

| Original (German) | English name | Type | Meaning |
|---|---|---|---|
| `BETRIEBSTAG` | `operating_day` | date | Operating day. A train running after midnight still belongs to the day it started |
| `FAHRT_BEZEICHNER` | `trip_id` | str | Trip identifier, unique within a day. `(operating_day, trip_id)` = one train run |
| `BETREIBER_ID` | `operator_id` | str | Operator code (e.g. `80:07____` = DB Regio Bayern) |
| `BETREIBER_ABK` | `operator_abbr` | str | Operator abbreviation (SBB, BLS, SOB, DB…) |
| `BETREIBER_NAME` | `operator_name` | str | Full operator name |
| `PRODUKT_ID` | `transport_mode` | str | `Zug` (train), `Bus`, `Tram`, `Metro`, `Zahnradbahn` (rack railway), `Schiff` (boat)… Case varies (`Bus`/`BUS`) |
| `LINIEN_ID` | `train_number` | str | For trains, the train number (e.g. 3280) |
| `LINIEN_TEXT` | `line_name` | str | Line shown to passengers (e.g. `RE7`, `IC5`) |
| `UMLAUF_ID` | `rotation_id` | str | Vehicle rotation (which vehicle runs a sequence of trips). Often empty |
| `VERKEHRSMITTEL_TEXT` | `category` | str | Train category: `IC`, `IR`, `RE`, `EC`, `S`, `R`, `ICE`… |
| `ZUSATZFAHRT_TF` | `is_extra_trip` | bool | Trip added outside the planned timetable |
| `FAELLT_AUS_TF` | `is_cancelled` | bool | Trip or this stop is cancelled |
| `BPUIC` | `station_id` | int | Station number (Swiss stations start with `85`) |
| `HALTESTELLEN_NAME` | `station_name` | str | Station name |
| `ANKUNFTSZEIT` | `arr_planned` | timestamp | Planned arrival, minute precision. Empty at first stop |
| `AN_PROGNOSE` | `arr_actual` | timestamp | Actual or forecast arrival, second precision |
| `AN_PROGNOSE_STATUS` | `arr_status` | str | Quality of `arr_actual` (see below) |
| `ABFAHRTSZEIT` | `dep_planned` | timestamp | Planned departure. Empty at last stop |
| `AB_PROGNOSE` | `dep_actual` | timestamp | Actual or forecast departure |
| `AB_PROGNOSE_STATUS` | `dep_status` | str | Quality of `dep_actual` |
| `DURCHFAHRT_TF` | `is_pass_through` | bool | Train passes without stopping |

## Status values

| Value | Meaning | Use |
|---|---|---|
| `REAL` | Measured | Only value counted as observed / label |
| `PROGNOSE` | Last forecast, never confirmed | Unobserved |
| `GESCHAETZT` | Estimated | Unobserved |
| `UNBEKANNT` | Unknown | Unobserved |
| empty | No arrival/departure at this stop | — |

## Notes

- Filter kept at ingestion: `upper(transport_mode) = 'ZUG'` (all trains, ~10 % of rows).
  IC/IR/RE/EC scope is applied later in code.
- `SLOID` appears from 2025-10-13 onwards. It is a station-level Swiss Location ID that duplicates
  `station_id` (e.g. `ch:1:sloid:3000` = 8503000 Zürich HB), so ingestion drops it on purpose.
- Profile of 2025-08-01: 1.66 M rows, 169 k trains, of which IC/IR/RE/EC ≈ 25 k.
  Train arrivals: 73 % `REAL`, 7 % `PROGNOSE`, 11 % `UNBEKANNT`, 8 % empty.
- `category` also contains generic labels: `ZUG` (TRN local trains, Neuchâtel, ~360 rows/day) and
  `Zug` (feed from operator `D`, mostly German stations, `line_name` RE87 / IC, ~400 rows/day).
  Category is missing there, not a real train type. Negligible for the IC/IR/RE/EC scope.
- August 2025 trains as zstd Parquet: 60 MB (vs 16 GB of raw CSV for all modes).

## Dataset built (1 Oct 2026)

14 months (Aug 2025 – Sep 2026), 425 days, all present. 76.0 M train rows
(158 k – 188 k per day), 840 MB of Parquet in `data/interim/`. Details per month in
`data/dataset_manifest.json`.
