"""SwissDelay showcase: the project explained for everyone, its results, three months of
(simulated) production, and live predictions for real trains.

Usage::

    uv run uvicorn swissdelay.serve.api:app            # in one terminal
    uv run python -m swissdelay.serve.example          # once: writes reports/showcase.json
    uv run streamlit run src/swissdelay/serve/dashboard.py

``SWISSDELAY_API`` sets the API address (default ``http://localhost:8000``) and
``SWISSDELAY_SHOWCASE`` the example trains (default ``reports/showcase.json``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("SWISSDELAY_API", "http://localhost:8000")
SHOWCASE = Path(os.environ.get("SWISSDELAY_SHOWCASE", "reports/showcase.json"))
GITHUB = "https://github.com/selim-ba/ml-train-delay"

# validated categorical palette and neutrals
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
LIGHT_BLUE, INK = "#9cc2ec", "#52514e"

# Methods compared on the Results page: tag → (name, what it uses)
R1, R2, R3 = "R1 · Delay stays the same", "R2 · Typical change", "R3 · Historical median"
M1, M2 = "M1 · XGBoost", "M2 · XGBoost with network data"
# Final results on the test month (June 2026, used once), common subset, MAE in minutes
RESULTS = pd.DataFrame(
    [
        (R1, "simple rule", 1.399, 1.581, 1.797),
        (R2, "simple rule", 1.155, 1.379, 1.649),
        (R3, "simple rule", 0.992, 1.228, 1.488),
        (M1, "model", 0.829, 1.016, 1.222),
        (M2 + " (deployed)", "deployed model", 0.805, 0.990, 1.200),
    ],
    columns=["method", "kind", "15 min", "30 min", "60 min"],
)
# June, normal vs disrupted days: R3 / M1 / M2
BY_DAY_TYPE = pd.DataFrame(
    [
        ("Normal days", 15, 0.953, 0.792, 0.771), ("Normal days", 30, 1.165, 0.959, 0.937),
        ("Normal days", 60, 1.394, 1.128, 1.115), ("Disrupted days", 15, 1.045, 0.879, 0.853),
        ("Disrupted days", 30, 1.314, 1.095, 1.062), ("Disrupted days", 60, 1.616, 1.351, 1.316),
    ],
    columns=["day type", "horizon", R3, M1, M2],
)  # fmt: skip
# Share of the model's decisions based on each kind of input (XGBoost gain), %
ATTENTION = pd.DataFrame(
    [
        ("What usually happens here", 44.3, 34.6, 31.1),
        ("The train's current delay", 13.9, 26.9, 39.3),
        ("The network around it", 14.8, 17.5, 13.6),
        ("Spare time in the timetable", 13.7, 8.8, 5.4),
        ("The timetable (stops, hour, day)", 6.4, 4.7, 4.0),
        ("Delays at its earlier stops", 3.8, 4.5, 3.8),
        ("Context (type of train, operator)", 3.0, 3.1, 2.8),
    ],
    columns=["input", "15 min", "30 min", "60 min"],
)
# Graph transformer experiment, validation month (May 2026), 15 min, MAE in minutes
GT = pd.DataFrame(
    [
        ("G1 · No map", "graph transformer", 0.734),
        ("G2 · Random map", "graph transformer", 0.708),
        ("G3 · Real map", "graph transformer", 0.705),
        ("G4 · Real map, retrained", "graph transformer", 0.697),
        (M2 + " (deployed)", "deployed model", 0.701),
        ("A · Average of M2 and G4", "average", 0.692),
    ],
    columns=["model", "kind", "mae"],
)
# A real departure of 30 September 2026 (reports/example_request.json), predicted by the API
OVERVIEW_EXAMPLE = (
    ("Departure", "now", 3.6),
    [{"title": "+15 min", "subtitle": "", "p50": 2.55, "lo": 1.72, "hi": 3.44, "actual": 3.5},
     {"title": "+30 min", "subtitle": "", "p50": 1.47, "lo": 0.53, "hi": 2.86, "actual": 1.8},
     {"title": "+60 min", "subtitle": "", "p50": 0.77, "lo": -0.71, "hi": 3.00, "actual": 3.4}],
)  # fmt: skip

st.set_page_config(page_title="How will this train's delay change", page_icon="🚆",
                   layout="wide")  # fmt: skip


# =========================================================================== helpers
@st.cache_data(ttl=300)
def get(path: str):
    r = requests.get(f"{API}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def post(path: str, body: str):
    r = requests.post(f"{API}{path}", data=body, timeout=30,
                      headers={"Content-Type": "application/json"})  # fmt: skip
    r.raise_for_status()
    return r.json()


@st.cache_data
def showcase() -> dict | None:
    return json.loads(SHOWCASE.read_text()) if SHOWCASE.exists() else None


def svg(markup: str) -> None:
    st.markdown(f'<div style="overflow-x:auto">{markup}</div>', unsafe_allow_html=True)


def delay_text(v: float) -> str:
    if v >= 0.05:
        return f"{v:.1f} min late"
    if v <= -0.05:
        return f"{-v:.1f} min early"
    return "on time"


def api_down(e: Exception) -> None:
    st.error(f"The prediction service is not reachable at {API} ({e}). Start it with "
             "`make api` (or `docker compose up`) and reload the page.")  # fmt: skip


def trip_svg(start: tuple[str, str, float], stops: list[dict], show_actual: bool = True) -> str:
    """A train's journey: delay now, then the predicted delay (dot) and likely range (bar)
    at the stops 15, 30 and 60 minutes ahead; orange rings: what actually happened.

    ``start`` is (station, planned time, delay now); each stop has ``title``, ``subtitle``,
    ``p50``, ``lo``, ``hi`` and ``actual`` (delays at the stop, minutes)."""
    values = [0.0, start[2], *(s[k] for s in stops for k in ("p50", "lo", "hi"))]
    if show_actual:
        values += [s["actual"] for s in stops if s.get("actual") is not None]
    low, high = min(values), max(values)
    pad = max(0.6, 0.12 * (high - low))
    low, high = low - pad, high + pad
    top, bottom, track = 40, 215, 248

    def y(v: float) -> float:
        return bottom - (v - low) / (high - low) * (bottom - top)

    xs = [90 + i * 620 / len(stops) for i in range(len(stops) + 1)]
    step = next(s for s in (0.5, 1, 2, 5, 10, 20, 50) if (high - low) / s <= 7)
    parts = ['<svg viewBox="-10 0 810 310" width="100%" style="max-width:800px;font-family:'
             'sans-serif;font-size:13px" role="img" aria-label="Predicted delay along the '
             'journey">']  # fmt: skip
    tick = step * int(low / step)
    while tick <= high:
        if tick >= low:
            zero = abs(tick) < 1e-9
            dash = ' stroke-dasharray="4 4"' if zero else ""
            label = "on time" if zero else f"{tick:g}"
            parts.append(f'<line x1="50" y1="{y(tick):.1f}" x2="770" y2="{y(tick):.1f}" '
                         f'stroke="{GRAY}" stroke-opacity="{0.6 if zero else 0.15}"{dash}/>'
                         f'<text x="44" y="{y(tick) + 4:.1f}" text-anchor="end" '
                         f'fill="currentColor" opacity="0.6">{label}</text>')  # fmt: skip
        tick += step
    parts.append('<text x="50" y="22" fill="currentColor" opacity="0.7">Delay (minutes)</text>'
                 f'<line x1="{xs[0]}" y1="{track}" x2="{xs[-1]}" y2="{track}" stroke="{GRAY}" '
                 'stroke-width="4" stroke-linecap="round"/>')  # fmt: skip
    titles = [(start[0], start[1])] + [(s["title"], s["subtitle"]) for s in stops]
    for x, (title, subtitle) in zip(xs, titles, strict=True):
        parts.append(f'<circle cx="{x}" cy="{track}" r="8" fill="white" stroke="{GRAY}" '
                     f'stroke-width="3"/><text x="{x}" y="{track + 28}" text-anchor="middle" '
                     f'fill="currentColor" font-weight="600">{title[:24]}</text>'
                     f'<text x="{x}" y="{track + 46}" text-anchor="middle" fill="currentColor" '
                     f'opacity="0.7">{subtitle}</text>')  # fmt: skip
    parts.append(f'<circle cx="{xs[0]}" cy="{y(start[2]):.1f}" r="7" fill="{INK}"/>'
                 f'<text x="{xs[0] + 14}" y="{y(start[2]) + 4:.1f}" fill="currentColor">'
                 f'{delay_text(start[2])} now</text>')  # fmt: skip
    for x, s in zip(xs[1:], stops, strict=True):
        parts.append(f'<rect x="{x - 10}" y="{y(s["hi"]):.1f}" width="20" '
                     f'height="{y(s["lo"]) - y(s["hi"]):.1f}" rx="5" fill="{BLUE}" '
                     f'opacity="0.25"/><circle cx="{x}" cy="{y(s["p50"]):.1f}" r="7" '
                     f'fill="{BLUE}" stroke="white" stroke-width="2"/>')  # fmt: skip
        if show_actual and s.get("actual") is not None:
            parts.append(f'<circle cx="{x}" cy="{y(s["actual"]):.1f}" r="6.5" fill="none" '
                         f'stroke="{ORANGE}" stroke-width="2.5"/>')  # fmt: skip
        anchor, dx = ("end", -16) if x == xs[-1] else ("start", 16)
        parts.append(f'<text x="{x + dx}" y="{y(s["p50"]) + 4:.1f}" text-anchor="{anchor}" '
                     f'fill="currentColor">{delay_text(s["p50"])}</text>')  # fmt: skip
    return "".join(parts) + "</svg>"


def legend(show_actual: bool = True, range_text: str = "likely range (8 times out of 10)"):
    items = [
        f'<span style="color:{BLUE}">●</span> most likely delay',
        f'<span style="color:{BLUE};opacity:.45">▮</span> {range_text}',
    ]
    if show_actual:
        items.append(f'<span style="color:{ORANGE}">◯</span> what actually happened')
    st.markdown(" &nbsp;·&nbsp; ".join(items), unsafe_allow_html=True)


def timeline_svg(
    labels: tuple[str, ...] = ("Training", "Validation", "Test", "Simulated production"),
) -> str:
    months = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep"]  # fmt: skip
    spans = [(0, 9, BLUE), (9, 10, ORANGE), (10, 11, AQUA), (11, 14, YELLOW)]
    phases = [(*span, label) for span, label in zip(spans, labels, strict=True)]
    w = 52
    parts = ['<svg viewBox="0 0 760 120" width="100%" style="max-width:760px;font-family:'
             'sans-serif;font-size:12px" role="img" aria-label="Timeline">']  # fmt: skip
    for a, b, color, label in phases:
        parts.append(f'<rect x="{10 + a * w + 1}" y="30" width="{(b - a) * w - 2}" height="34" '
                     f'rx="4" fill="{color}"/><text x="{10 + (a + b) / 2 * w}" y="20" '
                     f'text-anchor="middle" fill="currentColor" font-weight="600">{label}'
                     "</text>")  # fmt: skip
    for i, mth in enumerate(months):
        parts.append(f'<text x="{10 + (i + 0.5) * w}" y="82" text-anchor="middle" '
                     f'fill="currentColor" opacity="0.75">{mth}</text>')  # fmt: skip
    parts.append('<text x="10" y="108" fill="currentColor" opacity="0.75">2025</text>'
                 f'<text x="{10 + 5 * w}" y="108" fill="currentColor" opacity="0.75">2026</text>'
                 "</svg>")  # fmt: skip
    return "".join(parts)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table (keeps bold and code formatting, unlike a dataframe)."""
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


def range_svg() -> str:
    """The likely range: 10 % of outcomes below, 80 % inside, 10 % above."""
    lo, mid, hi = 220, 380, 580
    return (
        '<svg viewBox="0 0 800 130" width="100%" style="max-width:800px;font-family:sans-serif;'
        'font-size:13px" role="img" aria-label="Likely range">'
        f'<line x1="40" y1="60" x2="760" y2="60" stroke="{GRAY}" stroke-width="2"/>'
        f'<rect x="{lo}" y="45" width="{hi - lo}" height="30" rx="6" fill="{BLUE}" '
        'opacity="0.25"/>'
        f'<circle cx="{mid}" cy="60" r="8" fill="{BLUE}" stroke="white" stroke-width="2"/>'
        f'<text x="{lo}" y="98" text-anchor="middle" fill="currentColor" font-weight="600">'
        "P10</text>"
        f'<text x="{mid}" y="98" text-anchor="middle" fill="currentColor" font-weight="600">'
        "P50 (best guess)</text>"
        f'<text x="{hi}" y="98" text-anchor="middle" fill="currentColor" font-weight="600">'
        "P90</text>"
        f'<text x="{(40 + lo) / 2}" y="35" text-anchor="middle" fill="currentColor">'
        "10 % of trains end here</text>"
        f'<text x="{(lo + hi) / 2}" y="35" text-anchor="middle" fill="currentColor" '
        'font-weight="600">80 % of trains end inside the range</text>'
        f'<text x="{(hi + 760) / 2}" y="35" text-anchor="middle" fill="currentColor">'
        "10 % of trains end here</text>"
        '<text x="40" y="122" fill="currentColor" opacity="0.7">less delay</text>'
        '<text x="760" y="122" text-anchor="end" fill="currentColor" opacity="0.7">'
        "more delay</text></svg>"
    )


def bar_chart(df: pd.DataFrame, y: str, x: str, colors: dict[str, str], title: str,
              fmt: str = ".2f", height: int = 230) -> alt.LayerChart:  # fmt: skip
    """Horizontal bars, coloured by ``kind``, with the value at the end of each bar."""
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22).encode(
        y=alt.Y(f"{y}:N", sort=None, title=None, axis=alt.Axis(labelLimit=360)),
        x=alt.X(f"{x}:Q", title=title),
        color=alt.Color("kind:N", scale=alt.Scale(domain=list(colors),
                        range=list(colors.values())),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[y, alt.Tooltip(f"{x}:Q", format=fmt)],
    )  # fmt: skip
    text = bars.mark_text(align="left", dx=6).encode(text=alt.Text(f"{x}:Q", format=fmt),
                                                     color=alt.value(INK))  # fmt: skip
    return (bars + text).properties(height=height)


def no_showcase() -> None:
    st.warning(f"No example data at {SHOWCASE}: run "
               "`uv run python -m swissdelay.serve.example` once.")  # fmt: skip


# Pages, filled in at the bottom of the file (navigation), in reading order
PAGES: dict = {}
ORDER = ["overview", "data", "method", "results", "production", "try", "engineering",
         "glossary"]  # fmt: skip


def link(key: str, label: str) -> None:
    """A link to another page, under the paragraph that refers to it."""
    st.page_link(PAGES[key], label=label, icon=":material/arrow_forward:")


def pager(key: str) -> None:
    """Previous / next page, and the glossary, at the bottom of every page."""
    st.divider()
    i = ORDER.index(key)
    c = st.columns(3)
    if i > 0:
        prev = PAGES[ORDER[i - 1]]
        c[0].page_link(prev, label=f"Previous: {prev.title}", icon=":material/arrow_back:")
    if i < len(ORDER) - 1:
        nxt = PAGES[ORDER[i + 1]]
        c[1].page_link(nxt, label=f"Next: {nxt.title}", icon=":material/arrow_forward:")
    if key != "glossary":
        c[2].page_link(PAGES["glossary"], label="Glossary of all terms",
                       icon=":material/menu_book:")  # fmt: skip


def data_split() -> None:
    """Timeline and description of the four periods (Overview and How it works)."""
    svg(timeline_svg())
    st.markdown(
        """
The 14 months are split **by date**, so that the model is always judged on days that come
after the ones it learned from, as it would be in real life:

- **Training** (August 2025 – April 2026, 4.8 million examples): the models learn how
  delays usually evolve.
- **Validation** (May 2026, 0.5 million): the models and their settings are compared, and
  the best one is chosen.
- **Test** (June 2026, 0.5 million): the chosen model is scored **once**, on a month it has
  never seen. Nothing is changed afterwards.
- **Simulated production** (July – September 2026, 1.6 million): the model is run day by
  day, as it would be in service, **without being retrained**, to check that it keeps
  working over time.
"""
    )


# =========================================================================== pages
def overview() -> None:
    st.title("Can we predict how a train's delay will change?")
    st.markdown("##### XGBoost models and graph neural networks trained on 14 months of Swiss "
                f"railway data  \n[Code on GitHub]({GITHUB})")  # fmt: skip
    st.markdown(
        """
This project predicts **how the delay of every long-distance train in Switzerland (IC, IR, RE
and EC) will change over the next 15, 30 and 60 minutes**.

For instance, your train leaves its station **4 minutes late**. Will you still be late at your
stop in half an hour? Will the train make up time, or lose more?
"""
    )

    st.subheader("Which data was used for what")
    data_split()

    st.subheader("Results in a nutshell")
    st.markdown(
        """
For every train, we compare the predicted delay at the stop with the real one: the gap, in
minutes, is the **error**. Averaged over all trains, it tells how far off a method is in a
typical case (lower is better). To know whether a model is worth it, we compare it with
simple rules anyone could apply without machine learning. Each method has a short tag, used
throughout this site:

| Average error (min), test month (June 2026) | 15 min ahead | 30 min ahead | 60 min ahead |
|---|---|---|---|
| **R1** · *"the delay will stay the same"* | 1.40 | 1.58 | 1.80 |
| **R3** · historical median: *"the train will do what it usually does"* | 0.99 | 1.23 | 1.49 |
| **M2** · our model: XGBoost with network data | **0.81** | **0.99** | **1.20** |
| **Error reduction of M2 vs R3** | **−19 %** | **−19 %** | **−19 %** |

R3, the best of the simple rules, predicts for each train the typical change in delay of
past trains on the same line, at the same station, hour and type of day.

In other words, 15 minutes ahead the model's prediction is typically **less than a minute
off**, and it makes about a fifth less error than R3. Over the three months
of simulated production, this advantage held steady (−21 %), without any retraining.

Besides its best guess, the model gives a **likely range** of delays. It is built so that
the real delay falls inside it for **8 trains out of 10**, and over the three months of
simulated production it did: **80 %** of the real delays were inside their range.
"""
    )
    link("results", "All methods and results")

    st.subheader("What it predicts, on a real train")
    svg(trip_svg(*OVERVIEW_EXAMPLE))
    legend(range_text="likely range (the real delay falls inside it for 8 trains out of 10)")
    st.markdown(
        "This train left 3.6 minutes late on 30 September 2026. For each of the next stops, "
        "the model gives its best guess (blue dot) and a likely range (blue bar); the orange "
        "rings show what really happened."
    )
    link("try", "Try it on more trains")

    st.subheader("What I built, end to end")
    st.graphviz_chart(
        """
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25;
  node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=11,
        color="#8a8984", fillcolor="#f4f4f2", margin="0.15,0.08"];
  edge [color="#8a8984"];
  a [label="1. Open data\\n76 million train\\nstop records"];
  b [label="2. Clean journeys\\n1.1 million\\ntrain runs"];
  c [label="3. 7.4 million\\nexamples\\n(no future info)"];
  node [fillcolor="#dbe8f8", color="#2a78d6"];
  d [label="4. Models\\ncompared fairly"];
  e [label="5. Test on\\nan unseen month"];
  f [label="6. 3 months\\nreplayed as if live"];
  g [label="7. Prediction API\\n+ this dashboard"];
  a -> b -> c -> d -> e -> f -> g;
}
"""
    )
    st.markdown(
        """
1. **Open data.** I downloaded about **220 GB** of raw files (14 months × ~16 GB) from the
   Swiss open transport data platform: the planned and actual time of every train, bus and
   tram at every stop in the country. I kept the **76 million train records**.
2. **Clean journeys.** I kept the long-distance trains (IC, IR, RE, EC) and their Swiss
   stops, rebuilt each run stop by stop, and only trusted times that were really measured
   (not estimated). Implausible delays, inconsistent runs, cancelled stops and stations with
   unreliable measurements were set aside: **1.1 million train runs** remain.
3. **Examples.** Each departure becomes up to three examples (15, 30 and 60 minutes ahead):
   what was known **at least 2 minutes before** the train left, and how its delay then
   changed. **7.4 million examples** in all, with nothing taken from the future.
4. **Models.** Three simple rules as references, then **XGBoost** (with and without
   information on the other trains of the network), two extra XGBoost models for the likely
   range, and a **graph neural network** (a graph transformer) that reads the railway map.
5. **Test on an unseen month.** The best model, chosen on May, was scored once on June.
6. **Three months replayed as if live.** July to September were replayed day by day, as a
   nightly job would run in service: predict, check the predictions against what happened,
   update the likely ranges, watch for changes in traffic and raise alerts.
7. **Prediction API and this dashboard.** The model runs as a web service: send it a
   train's situation, it answers with the predicted delays and their ranges. This dashboard
   reads from it, and both are packaged with Docker.
"""
    )

    st.subheader("Technical stack")
    st.markdown(
        """
| Area | Tools |
|---|---|
| Languages | Python (PyTorch, scikit-learn), SQL |
| Data processing | DuckDB, Parquet, pandas, NumPy |
| Models | XGBoost, Graph Neural Network |
| Evaluation | time-based splits, bootstrap confidence intervals, conformal calibration |
| Serving | FastAPI, Streamlit, Graphviz, Altair, Uvicorn |
| Engineering | Docker Compose, GitHub Actions CI, pytest, ruff, uv |
"""
    )
    link("engineering", "How it is built: Under the hood")
    pager("overview")


def data_page() -> None:
    st.title("Data")
    st.markdown(
        """
Every day, the Swiss railways publish the **planned and actual time of every train at every
stop** in the country (*Ist-Daten*, open data from SBB). I downloaded **14 months** of it,
from August 2025 to September 2026, and kept the long-distance trains: **InterCity (IC),
InterRegio (IR), RegioExpress (RE) and EuroCity (EC)**.
"""
    )
    k = st.columns(4)
    k[0].metric("Days", "426")
    k[1].metric("Train stop records", "76.0 M")
    k[2].metric("Train runs kept", "1.11 M")
    k[3].metric("Times measured, not estimated", "95.7 %")

    st.subheader("From raw records to examples to learn from")
    st.graphviz_chart(
        """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=11,
        color="#8a8984", fillcolor="#f4f4f2", margin="0.15,0.08"];
  edge [color="#8a8984", fontname="sans-serif", fontsize=10];
  a [label="76.0 M records\\none train at one stop"];
  b [label="1.11 M train runs\\nIC / IR / RE / EC,\\nconsistent and complete"];
  c [label="7.38 M examples\\na train leaving a stop,\\nlooking 15, 30 or 60 min ahead"];
  a -> b [label=" clean, keep\\n long-distance"];
  b -> c [label=" one per departure\\n and horizon"];
}
"""
    )
    st.markdown(
        "Each **example** is one train leaving one station, with what was known at that "
        "moment and what then happened 15, 30 or 60 minutes later. That is what the model "
        "learns from: **4.8 M** examples for training, **0.5 M** for validation, **0.5 M** "
        "for the test, and **1.6 M** for the three months of simulated production."
    )
    link("method", "How the periods are used: How it works")

    st.subheader("Fourteen months of delays")
    data = showcase()
    if data is None:
        no_showcase()
        pager("data")
        return
    d = pd.DataFrame(data["daily"])
    d["day"] = pd.to_datetime(d["operating_day"])
    phase = {"train": "Training", "valid": "Validation", "test": "Test",
             "production": "Simulated production"}  # fmt: skip
    d["phase"] = d["split"].map(phase)
    st.caption("Average departure delay of the long-distance trains, each day. Dots: "
               "disrupted days, with unusually high delays and cancellations (storms, "
               "incidents, works).")  # fmt: skip
    base = alt.Chart(d).encode(x=alt.X("day:T", title=None))
    line = base.mark_line(strokeWidth=1.5).encode(
        y=alt.Y("mean_delay_min:Q", title="Average delay (min)"),
        color=alt.Color("phase:N", scale=alt.Scale(domain=list(phase.values()),
                        range=[BLUE, ORANGE, AQUA, YELLOW]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("day:T", format="%a %d %b %Y"), "phase",
                 alt.Tooltip("mean_delay_min:Q", format=".2f", title="average delay (min)"),
                 alt.Tooltip("cancelled_runs_share:Q", format=".1%", title="runs cancelled")],
    )  # fmt: skip
    dots = (
        base.transform_filter("datum.is_disruption_day")
        .mark_point(filled=True, size=40, color=INK)
        .encode(y="mean_delay_min:Q")
    )
    st.altair_chart((line + dots).properties(height=300), width="stretch")
    st.markdown(
        "Most days, long-distance trains leave **1 to 2 minutes late** on average, and a few "
        "days stand out sharply. A good forecast must work on calm days and on bad ones, so "
        "every result is also checked separately on the most disrupted days."
    )
    pager("data")


def method_page() -> None:
    st.title("How it works")
    st.subheader("1. The question")
    st.markdown(
        "A train leaves a station now. For **15, 30 and 60 minutes ahead**, take the first "
        "stop it is scheduled to reach at least that much later. How will its delay **change** "
        "by then? Negative = it makes up time; positive = it loses more."
    )
    xs = [80, 300, 500, 720]
    parts = ['<svg viewBox="0 0 800 110" width="100%" style="max-width:800px;font-family:'
             'sans-serif;font-size:13px" role="img" aria-label="Horizons">',
             f'<line x1="{xs[0]}" y1="55" x2="{xs[-1]}" y2="55" stroke="{GRAY}" '
             'stroke-width="4" stroke-linecap="round"/>']  # fmt: skip
    labels = [("Leaving now", "4 min late"), ("Stop at +15 min", "? min late"),
              ("Stop at +30 min", "? min late"), ("Stop at +60 min", "? min late")]  # fmt: skip
    for i, (x, (a, b)) in enumerate(zip(xs, labels, strict=True)):
        fill = INK if i == 0 else BLUE
        parts.append(f'<circle cx="{x}" cy="55" r="9" fill="{fill}"/>'
                     f'<text x="{x}" y="30" text-anchor="middle" fill="currentColor" '
                     f'font-weight="600">{a}</text><text x="{x}" y="88" text-anchor="middle" '
                     f'fill="currentColor" opacity="0.75">{b}</text>')  # fmt: skip
    svg("".join(parts) + "</svg>")

    st.subheader("2. No peeking at the future")
    st.markdown(
        "In real life, a forecast can only use what is known at the moment it is made, and "
        "train positions reach the system with a small lag. So every input is computed from "
        "events that happened **at least 2 minutes before** the prediction. Automated tests "
        "check this rule."
    )
    parts = ['<svg viewBox="0 0 800 110" width="100%" style="max-width:800px;font-family:'
             'sans-serif;font-size:13px" role="img" aria-label="Information cut-off">',
             f'<rect x="40" y="40" width="460" height="30" rx="4" fill="{BLUE}" opacity="0.8"/>',
             f'<rect x="500" y="40" width="80" height="30" fill="{GRAY}" opacity="0.35"/>',
             f'<rect x="580" y="40" width="180" height="30" rx="4" fill="{ORANGE}" '
             'opacity="0.35"/>',
             '<text x="270" y="30" text-anchor="middle" fill="currentColor" font-weight="600">'
             'Known: used by the model</text>',
             '<text x="540" y="30" text-anchor="middle" fill="currentColor">2 min</text>',
             '<text x="670" y="30" text-anchor="middle" fill="currentColor" font-weight="600">'
             'Future: never used</text>',
             f'<line x1="580" y1="34" x2="580" y2="80" stroke="{INK}" stroke-width="2"/>',
             '<text x="580" y="98" text-anchor="middle" fill="currentColor">prediction time'
             '</text></svg>']  # fmt: skip
    svg("".join(parts))

    st.subheader("3. What the model looks at")
    st.markdown(
        """
| Kind of input | Examples |
|---|---|
| **The train now** | its current delay, delays at its last stops |
| **Spare time in the timetable** | extra time planned before the target stop |
| **The timetable** | stops ahead, time of day, weekday |
| **Context** | type of train (IC, IR…), operator |
| **What usually happens here** | the typical change on that line, at that station and hour |
| **The network around it** | delays of nearby trains, the train just ahead, trends |

65 inputs in all.
"""
    )
    link("glossary", "The network inputs in detail: Glossary")
    st.subheader("4. The reference rules")
    st.markdown(
        "A model is only worth using if it beats simple rules that need no machine learning. "
        "Three rules serve as references. Each method gets a short tag, used on every page:"
    )
    st.markdown(md_table(
        ["Tag", "Rule", "What it predicts for the change in delay",
         "Example: a train leaving 4 min late"],
        [["**R1**", "**Persistence**", "no change: the delay stays the same",
          "still 4 min late at the stop"],
         ["**R2**", "**Typical change**", "the same change for every train: the median "
          "change of all training examples (about −0.8 min, thanks to spare time in "
          "timetables)", "about 3.2 min late"],
         ["**R3**", "**Historical median**", "the median change of past trains on the same "
          "line, at the same station, horizon, hour and type of day",
          "if trains there usually make up 1.5 min: 2.5 min late"]],
    ))  # fmt: skip
    st.markdown("R3 is the strongest of the three, and the main reference for the "
                "models.")  # fmt: skip

    st.subheader("5. The models")
    st.markdown(md_table(
        ["Tag", "Model", "Inputs", "Role"],
        [["**M1**", "**XGBoost** (`xgb_full`)", "the train now, spare time, timetable, "
          "context and R3 (32 inputs)", "first model"],
         ["", "**XGBoost + network** (`xgb_full_network`)", "+ 13 inputs on the other "
          "trains over the last 30 minutes", "intermediate step"],
         ["**M2**", "**XGBoost with network data** (`xgb_full_network_plus`)", "+ 20 more: "
          "trends over 10 and 60 minutes, the route to the target stop, the train just "
          "ahead (65 inputs)", "**deployed**"],
         ["", "**XGBoost P10 and P90**", "the same 65 inputs, trained to predict the low "
          "and high ends of the likely range", "likely range"],
         ["**G1–G4**", "**Graph transformer** (graph neural network), four variants", "the "
          "same inputs, plus a small map of the stations around the train, each with its "
          "own traffic and delays", "research, 15 min ahead"]],
    ))  # fmt: skip
    st.markdown("**A** denotes the average of the predictions of M2 and G4 (research). The "
                "four graph transformer variants are detailed on the Results page.")  # fmt: skip

    st.subheader("6. Which data was used for what")
    data_split()

    st.subheader("7. How the score works")
    st.markdown("The score is the **mean absolute error** (MAE), in minutes:")
    st.latex(r"\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| \hat{y}_i - y_i \right|")
    c = st.columns([3, 2])
    c[0].markdown(
        r"""
- $N$: the number of examples scored (train departures, for one horizon, over the period);
- $y_i$: the **actual** change in delay of example $i$, the delay at the target stop minus
  the delay at departure, in minutes;
- $\hat{y}_i$: the **predicted** change in delay for the same example;
- $\left| \hat{y}_i - y_i \right|$: the error of that example, ignoring its sign: 1 minute
  too early or 1 minute too late counts the same.

The delay at departure is known, so an error on the change is also the error on the delay
at the stop. The MAE is in minutes, so anyone can read it ("off by 0.8 minutes on
average"), and a few extreme delays do not dominate it. Lower is better.

Every comparison also comes with a **confidence interval**: the score is recomputed on
thousands of resampled sets of days, to check that a difference between two models is not
luck.
"""
    )
    c[1].markdown(
        """
| Train | Predicted change $\\hat{y}_i$ | Actual change $y_i$ | Error |
|---|---|---|---|
| A | −1.0 | −0.5 | 0.5 |
| B | +0.5 | +1.5 | 1.0 |
| C | −2.0 | −2.3 | 0.3 |
| | | **MAE** | **0.6 min** |
"""
    )

    st.subheader("8. The likely range")
    st.markdown(
        "Besides its best guess, the model gives a **likely range** for each prediction. It is "
        "built so that the real delay falls **below it for 10 % of trains, inside it for 80 %, "
        "and above it for 10 %**."
    )
    svg(range_svg())
    st.markdown(
        """
It is built in three steps:

1. **Two extra models.** Two more XGBoost models are trained on the same inputs, not to
   predict the most likely change but the **P10** (the value the change stays below for 10 %
   of trains) and the **P90** (below for 90 %). Together with the best guess (the P50, or
   median), they give a first range.
2. **Calibration.** On their own, these ranges were too narrow: only 74 % of real delays
   fell inside. Each end is therefore pushed outwards by a margin measured on recent
   outcomes, just enough to have 10 % of real delays below and 10 % above. The margins are
   computed separately for trains on time, 1–3, 3–10 and more than 10 minutes late, as
   their delays vary very differently.
3. **Nightly update.** The margins are recomputed every night from the last 14 days, so the
   range stays honest when the network goes through a bad spell.

| Version of the range | Real delays inside (target 80 %) |
|---|---|
| Two extra models only (June) | 74 % |
| Margins fixed once, on April (June) | 77 % |
| **Margins updated every night (July – September)** | **80 %** |
"""
    )
    link("results", "All results: Results")
    pager("method")


def results_page() -> None:
    st.title("Results")
    st.markdown(
        "All results below are measured on the **test month, June 2026**, which none of the "
        "methods saw during training or selection. The scoring metric is the **MAE** (mean "
        "absolute error, in minutes): lower is better."
    )
    link("method", "How the methods and the MAE are defined: How it works")
    st.subheader("The methods compared")
    st.markdown("As a reminder, each method has a short tag, used in the charts below.")
    st.markdown(md_table(
        ["Tag", "Method", "What it uses to predict the change in delay"],
        [["**R1**", "Delay stays the same (persistence)", "nothing: it predicts no change"],
         ["**R2**", "Typical change", "the median change of all training examples "
          "(about −0.8 min), the same for every train"],
         ["**R3**", "Historical median (best simple rule)", "the median change of past trains "
          "on the same line, at the same station, horizon, hour and type of day"],
         ["**M1**", "XGBoost", "32 inputs on the train itself: current delay, delays at its "
          "last stops, spare time in the timetable, stops ahead, hour, day, type of train, "
          "and R3"],
         ["**M2**", "XGBoost with network data (**deployed**)", "M1's inputs plus 33 inputs on "
          "the other trains around it: their delays at its stations and on its route, the "
          "train just ahead, recent trends (65 inputs)"],
         ["**G1–G4**", "Graph transformer variants (research)", "M1's inputs plus a small map "
          "of the stations around the train (detailed further down)"],
         ["**A**", "Average of M2 and G4 (research)", "the mean of the two predictions"]],
    ))  # fmt: skip

    st.subheader("Simple rules versus models")
    st.markdown(
        "All methods are scored on the same departures (175,420 per horizon). From top to "
        "bottom, each method uses more information than the one before; the bars show "
        "its MAE."
    )
    horizon = st.radio("How far ahead", ["15 min", "30 min", "60 min"], horizontal=True)
    r = RESULTS.assign(error=RESULTS[horizon])
    colors = {"simple rule": GRAY, "model": LIGHT_BLUE, "deployed model": BLUE}
    st.altair_chart(bar_chart(r, "method", "error", colors, "MAE (minutes)"),
                    width="stretch")  # fmt: skip
    persistence, best_rule, model = r["error"].iloc[0], r["error"].iloc[2], r["error"].iloc[4]
    st.markdown(
        f"At **{horizon}**, the deployed model M2 is off by **{model:.2f} min** on average. "
        f"That is **{100 * (1 - model / best_rule):.0f} % less error** than R3, the historical "
        "median (predicting, for each train, the typical change of past trains on the same "
        "line, at the same station, hour and type of day), and "
        f"**{100 * (1 - model / persistence):.0f} % less** than R1 (assuming the delay stays "
        "the same). The gains of M1 over R3 and of M2 over M1 are both confirmed by confidence "
        "intervals."
    )

    st.subheader("The network helps most on disrupted days")
    st.markdown(
        "A **disrupted day** is a day with unusually high delays and cancellations. Each day "
        "gets a score that combines its average delay and its share of cancelled trains, "
        "compared with the training months; a day is called disrupted when its score is "
        "above the level reached by only the worst 5 % of training days. June 2026 was a "
        "difficult month: **13 of its 30 days** were disrupted."
    )
    h = int(horizon.split()[0])
    methods = [R3, M1, M2]
    b = BY_DAY_TYPE[BY_DAY_TYPE["horizon"] == h].melt(
        id_vars="day type", value_vars=methods, var_name="method", value_name="error"
    )
    bars = alt.Chart(b).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("day type:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("method:N", sort=methods),
        y=alt.Y("error:Q", title="MAE (minutes)"),
        color=alt.Color("method:N", sort=methods, scale=alt.Scale(domain=methods,
                        range=[GRAY, LIGHT_BLUE, BLUE]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["day type", "method", alt.Tooltip("error:Q", format=".3f")],
    )  # fmt: skip
    text = bars.mark_text(dy=-6).encode(text=alt.Text("error:Q", format=".2f"),
                                         color=alt.value(INK))  # fmt: skip
    st.altair_chart((bars + text).properties(height=280), width="stretch")
    g = b.pivot_table(index="day type", columns="method", values="error")
    gain = 100 * (1 - g[M2] / g[M1])
    st.markdown(
        f"Adding the network data (M1 → M2) cuts the error by **{gain['Normal days']:.1f} %** "
        f"on normal days and **{gain['Disrupted days']:.1f} %** on disrupted days ({horizon} "
        "ahead). When things go wrong, knowing what happens around the train is worth more."
    )

    st.subheader("What the model relies on")
    st.markdown(
        """
**How this is measured.** XGBoost builds thousands of small decision trees. Each split in a
tree tests one input and reduces the model's error by some amount during training. Adding
up these reductions for every input (XGBoost's *total gain* importance) and grouping the
inputs by kind gives the share of the error reduction owed to each kind of input, for the
deployed model M2 at each horizon. It shows what the model uses most overall.
"""
    )
    a = ATTENTION.melt(id_vars="input", var_name="horizon", value_name="share")
    horizons = ["15 min", "30 min", "60 min"]
    chart = alt.Chart(a).mark_bar(cornerRadiusEnd=3, height=9).encode(
        y=alt.Y("input:N", sort=list(ATTENTION["input"]), title=None,
                axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("horizon:N", sort=horizons),
        x=alt.X("share:Q", title="Share of the total gain (%)"),
        color=alt.Color("horizon:N", sort=horizons, scale=alt.Scale(domain=horizons,
                        range=[BLUE, ORANGE, AQUA]),
                        legend=alt.Legend(title="Looking ahead", orient="bottom")),
        tooltip=["input", "horizon", alt.Tooltip("share:Q", format=".1f", title="%")],
    ).properties(height=360)  # fmt: skip
    st.altair_chart(chart, width="stretch")
    st.markdown(
        "For the next 15 minutes, the model relies most on **what usually happens here** "
        "(R3, used as an input). The further ahead it looks, the more the **train's current "
        "delay** matters: a very late train has more room to make up time, and the "
        "timetable's spare time gets used up along the way. The train's earlier stops add "
        "little once its current delay is known."
    )

    st.subheader("Trying a Graph Neural Network: Graph Transformer")
    st.markdown(
        "Railways are a network, so I also built a **graph transformer**, a type of graph "
        "neural network. For each departure, it reads a small map of the stations around the "
        "train (each station with its own traffic and delays, linked by the tracks between "
        "them), and lets every station weigh the information of the others. To understand "
        "where its gains come from, four variants were trained:"
    )
    st.markdown(md_table(
        ["Tag", "Variant", "What it reads", "MAE, May, 15 min"],
        [["**G1**", "No map", "M1's inputs only, same network without the stations "
          "(control)", "0.734"],
         ["**G2**", "Random map", "the same stations, but linked at random", "0.708"],
         ["**G3**", "Real map", "the stations around the train and the real links between "
          "them", "0.705"],
         ["**G4**", "Real map, retrained", "G3, retrained on all training months, like M2",
          "0.697"],
         ["**M2**", "XGBoost with network data (deployed)", "65 inputs, no map", "0.701"],
         ["**A**", "Average of M2 and G4", "the mean of the two predictions", "0.692"]],
    ))  # fmt: skip
    st.caption("These scores are on the validation month (May 2026), 15 minutes ahead, where "
               "the variants were compared; only G4 and A were then scored on June.")  # fmt: skip
    colors = {"graph transformer": LIGHT_BLUE, "deployed model": BLUE, "average": GRAY}
    st.altair_chart(bar_chart(GT, "model", "mae", colors, "MAE (minutes), May, 15 min ahead",
                              fmt=".3f", height=250), width="stretch")  # fmt: skip
    st.markdown(
        """
- **G1 → G3:** adding the stations cuts the error by 4 %, so the extra information helps.
- **G2 ≈ G3:** a random map works almost as well as the real one: the gain comes from the
  per-station information, not from the map's structure.
- **G4 vs M2:** on the June test month, the two **tied** (difference 0.000 min).
- **A:** averaging the two was **1 % better** than M2 on June. I had set a **2 % rule**
  beforehand for running a second, much heavier model every night, so **M2 stays deployed**.
"""
    )

    st.subheader("Are the likely ranges reliable?")
    st.markdown(
        """
**What they are.** With each prediction, the model also gives a **likely range**. It is
designed so that the real delay ends **below the range for 10 % of trains, inside it for
80 %, and above it for 10 %** (the P10–P90 range).

**How they are built.** Two extra XGBoost models predict the P10 and the P90. Each end is
then widened by a margin learned from real outcomes, just enough to reach 10 % below and
10 % above, separately for trains on time, 1–3, 3–10 and more than 10 minutes late.

**What we measured.** The share of real delays that fell inside, below and above the range:
"""
    )
    st.markdown(md_table(
        ["Version of the range", "Scored on", "Inside (target 80 %)", "Below (target 10 %)",
         "Above (target 10 %)"],
        [["Two extra models only", "June", "74 %", "", ""],
         ["Margins learned once, on April", "June", "77 %", "10 %", "13 %"],
         ["**Margins updated every night from the last 14 days, one per end (deployed)**",
          "July – September", "**80 %**", "**10 %**", "**10 %**"]],
    ))  # fmt: skip
    link("method", "How the likely range is built, step by step: How it works")
    st.markdown(
        "**What we observed.** Margins learned once, on April (a calm month), were too narrow "
        "by June, mostly on the high side: 13 % of trains ended later than their range. "
        "Updating the margins every night, with a separate margin for each end, brought the "
        "range back on target over the three months of simulated production, on normal days "
        "(80 %) and almost on disrupted days (79 %)."
    )

    st.subheader("Limitations")
    st.markdown(
        """
- Only long-distance trains (IC, IR, RE, EC); regional and S-Bahn trains are out of scope.
- "Production" is a **faithful replay** of July–September 2026 from the published data, not
  a live connection to the railway's systems.
- The gains are a fraction of a minute on average: useful at scale (connections, passenger
  information), but delays remain partly unpredictable.
- The graph transformer was only tested 15 minutes ahead.
"""
    )
    link("production", "Three months of simulated production: In production")
    pager("results")


ALERT_NAMES = {"mae_jump": "Error jump", "low_coverage": "Low coverage",
               "drift:d0_min": "Drift: current delays", "drift:hour": "Drift: hours",
               "drift:net_cur_n": "Drift: traffic at stations"}  # fmt: skip


def production_page() -> None:
    st.title("In production")
    try:
        daily = pd.DataFrame(get("/metrics/daily"))
        health = get("/health")
    except requests.RequestException as e:
        api_down(e)
        pager("production")
        return
    daily["day"] = pd.to_datetime(daily["operating_day"])
    st.markdown(
        "From **1 July to 30 September 2026**, the deployed model **M2** was run day by day, "
        "exactly as it would run every night in service, **without being retrained**. Each "
        "morning, its predictions of the previous day were checked against what actually "
        "happened, the likely ranges were updated, and alerts were raised if something "
        "looked wrong."
    )
    st.markdown(md_table(
        ["Tag", "Method", "Role on this page"],
        [["**M2**", "XGBoost with network data (deployed)", "the model running in production"],
         ["**R3**", "Historical median (best simple rule): the typical change of past trains "
          "on the same line, at the same station, hour and type of day",
          "the reference M2 must keep beating"]],
    ))  # fmt: skip
    horizon = st.radio("How far ahead", health["horizons_min"], horizontal=True,
                       format_func=lambda h: f"{h} min")  # fmt: skip
    d = daily[daily["horizon_min"] == horizon].sort_values("day")
    w = d["n"] / d["n"].sum()
    mae, hist = (w * d["mae"]).sum(), (w * d["mae_historical"]).sum()
    cov = (w * d["coverage_rolling_tails"]).sum()
    k = st.columns(4)
    k[0].metric("MAE of M2", f"{mae:.2f} min", help="Over the three months")
    k[1].metric("Less error than R3", f"{100 * (1 - mae / hist):.0f} %")
    k[2].metric("Real delays inside the likely range", f"{100 * cov:.0f} %",
                help="Target: 80 %")  # fmt: skip
    k[3].metric("Alerts in 3 months", int((d["alerts"].fillna("") != "").sum()))

    disrupted = d.loc[d["is_disruption_day"], ["day"]].assign(
        end=lambda x: x["day"] + pd.Timedelta(days=1)
    )
    shade = (
        alt.Chart(disrupted)
        .mark_rect(opacity=0.15, color=GRAY)
        .encode(x=alt.X("day:T"), x2=alt.X2("end:T"))
    )

    st.subheader("Daily error")
    st.caption("MAE of each day. Grey bands: disrupted days (unusually high delays and "
               "cancellations). M2 (blue) stays below R3 (orange) every single "
               "day.")  # fmt: skip
    series = {"mae": M2 + " (deployed)", "mae_historical": R3}
    long = d.melt(id_vars="day", value_vars=list(series), var_name="series",
                  value_name="minutes")  # fmt: skip
    long["series"] = long["series"].map(series)
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("day:T", title=None), y=alt.Y("minutes:Q", title="MAE (minutes)"),
        color=alt.Color("series:N", scale=alt.Scale(domain=list(series.values()),
                        range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("day:T", format="%a %d %b"), "series",
                 alt.Tooltip("minutes:Q", format=".2f")],
    )  # fmt: skip
    st.altair_chart((shade + lines).properties(height=280), width="stretch")

    st.subheader("Is the likely range reliable?")
    st.caption("Share of trains whose real delay ended inside M2's likely range, each day. It "
               "should be close to 80 %. The deployed version updates the range's margins "
               "every night from the last 14 days; the other kept the margins learned on "
               "April.")  # fmt: skip
    names = {"coverage_rolling_tails": "Margins updated nightly (deployed)",
             "coverage_static": "Margins fixed since April"}  # fmt: skip
    cv = d.melt(id_vars="day", value_vars=list(names), var_name="method", value_name="share")
    cv["method"] = cv["method"].map(names)
    cov_lines = alt.Chart(cv).mark_line(strokeWidth=2).encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("share:Q", title="Inside the range", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0.65, 0.9])),
        color=alt.Color("method:N", scale=alt.Scale(domain=list(names.values()),
                        range=[BLUE, GRAY]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("day:T", format="%a %d %b"), "method",
                 alt.Tooltip("share:Q", format=".1%")],
    )  # fmt: skip
    target = alt.Chart(pd.DataFrame({"y": [0.8]})).mark_rule(strokeDash=[4, 4], color=GRAY
                                                             ).encode(y="y:Q")  # fmt: skip
    st.altair_chart((shade + cov_lines + target).properties(height=260), width="stretch")
    link("method", "How the likely range is built: How it works")

    st.subheader("Has the traffic changed since training?")
    st.caption("A drift score (PSI, population stability index) compares the distribution of "
               "three inputs each day with the training months: current delays, hours of "
               "departure and traffic at stations. Below 0.1 means 'same as during "
               "training'; above 0.25 triggers an alert.")  # fmt: skip
    psi_names = {"psi_d0_min": "current delays", "psi_hour": "hours of departure",
                 "psi_net_cur_n": "traffic at stations"}  # fmt: skip
    ps = d.melt(id_vars="day", value_vars=list(psi_names), var_name="input", value_name="score")
    ps["input"] = ps["input"].map(psi_names)
    st.altair_chart(alt.Chart(ps).mark_line(strokeWidth=2).encode(
        x=alt.X("day:T", title=None), y=alt.Y("score:Q", title="Drift score",
                                              scale=alt.Scale(domain=[0, 0.3])),
        color=alt.Color("input:N", scale=alt.Scale(range=[BLUE, ORANGE, AQUA]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("day:T", format="%a %d %b"), "input",
                 alt.Tooltip("score:Q", format=".3f")],
    ).properties(height=220), width="stretch")  # fmt: skip

    st.subheader("Alerts")
    st.markdown(
        "Every morning, three checks are run on the previous day, separately for each "
        "horizon (15, 30 and 60 minutes). Each failed check raises an alert:"
    )
    st.markdown(md_table(
        ["Alert", "What is checked", "Raised when"],
        [["**Error jump**", "M2's MAE of the day, compared with the median of its daily MAE "
          "over the previous 14 days (once 7 days of history exist)",
          "the day's MAE is more than **25 % above** that median"],
         ["**Low coverage**", "the share of real delays inside M2's likely range that day",
          "it falls **below 70 %** (target 80 %)"],
         ["**Drift**", "the drift score (PSI) of each of the three inputs above",
          "any score is **above 0.25**"]],
    ))  # fmt: skip
    st.caption("An alert means a person should look at what happened; it does not change the "
               "model by itself.")  # fmt: skip
    a = daily[daily["alerts"].fillna("") != ""].copy()
    if a.empty:
        st.success("No alerts in the whole period.")
    else:
        a["alert"] = a["alerts"].map(
            lambda x: ", ".join(ALERT_NAMES.get(t, t) for t in x.split(","))
        )
        kinds = sorted({k for x in a["alert"] for k in x.split(", ")})
        st.markdown(
            f"**Result: {len(a)} alerts on {a['operating_day'].nunique()} days**, of type: "
            f"{', '.join(kinds)}. They all fell on genuinely bad days for the network; even "
            "then, M2 stayed well below R3."
        )  # fmt: skip
        better = 100 * (1 - a["mae"] / a["mae_historical"])
        st.dataframe(
            a.assign(better=better)[["operating_day", "horizon_min", "alert", "mae", "better"]],
            hide_index=True, column_config={
                "operating_day": "day", "horizon_min": "minutes ahead",
                "mae": st.column_config.NumberColumn("MAE of M2 (min)", format="%.2f"),
                "better": st.column_config.NumberColumn("less error than R3",
                                                        format="%.0f %%")},
        )  # fmt: skip
    st.caption(
        f"Model in service: trained on {health['trained_on'][0]} to {health['trained_on'][1]}; "
        f"likely ranges last updated from {health['interval_calibration']['window'][0]} to "
        f"{health['interval_calibration']['window'][1]}."
    )
    pager("production")


def try_page() -> None:
    st.title("Try it")
    st.markdown(
        "Pick a real train from September 2026, during the simulated production. Its situation "
        "at departure (its 65 inputs) is sent to the **live prediction service**, which runs "
        "the deployed model **M2** (XGBoost with network data), exactly as a travel app would "
        "do it. Make your own guess first, then reveal what actually happened."
    )
    data = showcase()
    if data is None:
        no_showcase()
        pager("try")
        return
    trains = data["trains"]

    def label(j: int) -> str:
        t = trains[j]
        day = pd.Timestamp(t["day"]).strftime("%a %d %b")
        return (f"{t['train']} from {t['from']['station']} at {t['from']['planned']} on {day}, "
                f"{delay_text(t['from']['delay_min'])}")  # fmt: skip

    i = st.selectbox("Train", range(len(trains)), format_func=label)
    t = trains[i]
    try:
        preds = post("/predict", json.dumps({"points": t["points"]}))
    except requests.RequestException as e:
        api_down(e)
        pager("try")
        return
    by_h = {p["horizon_min"]: p for p in preds}
    reveal = st.toggle("Show what actually happened", value=False)

    stops = []
    for tg in t["targets"]:
        p = by_h[tg["horizon_min"]]
        stops.append({
            "title": tg["station"], "subtitle": f"{tg['planned']} · +{tg['horizon_min']} min",
            "p50": p["delay_at_target_p50"], "lo": p["delay_at_target_p10"],
            "hi": p["delay_at_target_p90"], "actual": tg["actual_delay_min"],
        })  # fmt: skip
    start = (t["from"]["station"], t["from"]["planned"], t["from"]["delay_min"])
    st.markdown(f"**{t['train']}** (train {t['train_number']}), {t['weekday']} {t['day']}: "
                f"leaving **{t['from']['station']}** at {t['from']['planned']}, "
                f"**{delay_text(t['from']['delay_min'])}**.")  # fmt: skip
    svg(trip_svg(start, stops, show_actual=reveal))
    legend(show_actual=reveal, range_text="M2's likely range (the real delay should fall "
                                          "inside it for 8 trains out of 10)")  # fmt: skip

    rows = []
    for s, tg in zip(stops, t["targets"], strict=True):
        row = {"looking ahead": f"{tg['horizon_min']} min", "stop": s["title"],
               "planned arrival": tg["planned"], "M2's best guess": delay_text(s["p50"]),
               "M2's likely range (min)": f"{s['lo']:.1f} to {s['hi']:.1f}"}  # fmt: skip
        if reveal:
            inside = s["lo"] <= s["actual"] <= s["hi"]
            row |= {"actual": delay_text(s["actual"]),
                    "error (min)": round(abs(s["actual"] - s["p50"]), 1),
                    "inside the range": "yes" if inside else "no"}  # fmt: skip
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Delays at the stop, in minutes (negative = early). The error is the gap between "
               "M2's best guess and the real delay; its average over all trains is the MAE. "
               "Two horizons can point to the same stop when the train has no stop in "
               "between.")  # fmt: skip
    st.markdown(
        "**How to read it.** A single train can land outside its range: that should happen "
        "for about 2 trains in 10. What matters is how often it happens over many trains, "
        "which is measured every day in simulated production."
    )
    c = st.columns(2)
    with c[0]:
        link("production", "Coverage day by day: In production")
    with c[1]:
        link("method", "How the likely range is built: How it works")
    with st.expander("What was sent to the service, and its answer"):
        st.markdown("For each horizon, the request carries M2's 65 inputs, all computed from "
                    "what was known at departure (first horizon shown):")  # fmt: skip
        st.json({"points": t["points"][:1]}, expanded=False)
        st.markdown("The answer: the change in delay (`delta_`) and the delay at the stop "
                    "(`delay_at_target_`), each as P50 (best guess), P10 and P90 (the ends of "
                    "the likely range).")  # fmt: skip
        st.json(preds, expanded=False)
    pager("try")


def engineering_page() -> None:
    st.title("Under the hood")
    st.markdown("How the project is built. Method tags (R1–R3, M1–M2, G1–G4) are defined on "
                "*How it works* and in the Glossary.")  # fmt: skip
    st.subheader("Architecture")
    st.graphviz_chart(
        """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=11,
        color="#8a8984", fillcolor="#f4f4f2", margin="0.15,0.08"];
  edge [color="#8a8984", fontname="sans-serif", fontsize=9];
  src [label="SBB open data\\n(daily files)"];
  subgraph cluster_offline { label="Offline: Python, DuckDB, Parquet"; fontname="sans-serif";
    fontsize=10; color="#cccccc";
    ing [label="Ingest & clean\\njourneys"];
    feat [label="Labels &\\nfeatures"];
    train [label="Train & evaluate\\nR1–R3, M1–M2,\\nP10 / P90, G1–G4"];
  }
  reg [label="Model registry\\nmodels + manifest", fillcolor="#dbe8f8", color="#2a78d6"];
  subgraph cluster_online { label="Served: Docker Compose"; fontname="sans-serif";
    fontsize=10; color="#cccccc";
    job [label="Nightly job\\nscore, recalibrate,\\ndrift, alerts"];
    api [label="FastAPI\\n/predict, /metrics", fillcolor="#dbe8f8", color="#2a78d6"];
    ui [label="Streamlit\\nthis website"];
  }
  src -> ing -> feat -> train -> reg -> api -> ui;
  reg -> job;
  job -> api [label=" metrics,\\n ranges"];
}
"""
    )
    c = st.columns(2)
    c[0].markdown(
        """
**Stack**
- **Languages:** Python (PyTorch, scikit-learn), SQL
- **Data:** DuckDB SQL over Parquet, pandas, NumPy; time-safe *as-of* joins for the network
  inputs (only events at least 2 minutes old)
- **Models:** XGBoost with a median loss (M1, M2) and quantile losses (P10 / P90),
  split-conformal calibration of the likely range, a Graphormer-style graph transformer in
  PyTorch on the Apple GPU (G1–G4)
- **Evaluation:** time-based splits, test month used once, day-clustered bootstrap
  confidence intervals, separate scores on disrupted days
- **Serving:** model registry with a manifest, nightly replay job, FastAPI, Uvicorn,
  Streamlit, Altair, Graphviz
- **Hardware:** every model was trained locally on a MacBook Pro with an Apple M4 Pro chip
  and 24 GB of RAM: XGBoost on the CPU, the graph transformer on the chip's GPU (PyTorch MPS)
"""
    )
    c[1].markdown(
        """
**Engineering practices**
- **95 automated tests**, including leakage tests (no input may use information from the
  last 2 minutes or the future) and a check that the exported M2 reproduces the training
  run's predictions exactly
- **CI** on every push with GitHub Actions (ruff lint, pytest); dependencies locked with uv
- **Docker Compose** runs the API and this dashboard with one command
- **Reproducible pipeline:** one command per step, everything rebuilt from the raw data
- **Monitoring:** drift scores, range coverage and error jumps raise alerts
"""
    )
    link("production", "The monitoring in action: In production")
    st.subheader("The prediction service")
    st.markdown("The service runs M2. One request per departure, one point per horizon; the "
                "answer gives the change in delay and the delay at the stop, each as P50 "
                "(best guess), P10 and P90 (the likely range).")  # fmt: skip
    st.code(
        """curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \\
     -d @reports/example_request.json

[{"horizon_min": 15, "delay_now_min": 3.6,
  "delta_p50": -1.05, "delta_p10": -1.88, "delta_p90": -0.16,
  "delay_at_target_p50": 2.55, "delay_at_target_p10": 1.72, "delay_at_target_p90": 3.44},
 ...]""",
        language="bash",
    )
    st.markdown("Other endpoints: `GET /health` (model version, training period, calibration "
                "window), `GET /metrics/daily` and `GET /metrics/summary` (production "
                "metrics). Interactive documentation at `/docs` on the service.")  # fmt: skip
    st.subheader("Run it yourself")
    st.code(
        """git clone https://github.com/selim-ba/ml-train-delay && cd ml-train-delay
docker compose up --build      # API on :8000, this site on :8501""",
        language="bash",
    )
    st.markdown(f"Full instructions, from downloading the data to every result, are in the "
                f"[README]({GITHUB}#readme).")  # fmt: skip
    link("try", "See the service answer live: Try it")
    pager("engineering")


def glossary_page() -> None:
    st.title("Glossary")
    st.markdown(
        """
#### Question
- **Delay**: how many minutes after the timetable a train actually leaves or arrives.
- **Change in delay**: the delay at a later stop minus the delay now. Negative = the train
  makes up time; positive = it loses more.
- **Horizon (15, 30, 60 min)**: how far ahead we look. For "15 min", the prediction is for the
  first stop the train is scheduled to reach at least 15 minutes after leaving now; same for 30
  and 60. The further ahead, the harder.
- **Example (prediction point)**: one train leaving one station, for one horizon. The test
  month alone has about 175,000 per horizon.

#### Data periods
- **Training** (August 2025 – April 2026): the months the models learn from.
- **Validation** (May 2026): the month used to compare the models and choose one.
- **Test** (June 2026): the month the chosen model is scored on, **once**, with no change
  afterwards.
- **Simulated production** (July – September 2026): the months replayed day by day, as if the
  model ran every night in service, without retraining.

#### Methods, models and their tags
- **R1 · Persistence**: "the delay stays the same". The simplest possible forecast.
- **R2 · Typical change**: every train changes by the same amount, the median change seen in
  training (about −0.8 minutes, as timetables include some spare time).
- **R3 · Historical median**: the median change of past trains on the same line, at the same
  station, horizon, hour and type of day. The best simple rule, and the main reference.
- **M1 · XGBoost** (`xgb_full`): XGBoost with 32 inputs on the train itself: current delay,
  delays at its last stops, spare time before the target stop, stops ahead, hour, weekday,
  type of train, operator, and R3.
- **XGBoost + network** (`xgb_full_network`): M1 plus 13 summaries of the network over the
  last 30 minutes. An intermediate step, not tagged.
- **M2 · XGBoost with network data** (`xgb_full_network_plus`, **deployed**): M1 plus 33
  network inputs, 65 inputs in all.
- **P10 / P90 models**: two extra XGBoost models, with M2's inputs, that predict the low and
  high ends of the likely range.
- **G1–G4 · Graph transformer variants** (research, 15 minutes ahead): a graph neural network
  that reads the stations around the train as a small map. G1 without the stations (control),
  G2 with a random map, G3 with the real map, G4 the real map retrained on all training months.
- **A**: the average of the predictions of M2 and G4 (research).
- **XGBoost**: a machine-learning method that combines thousands of small decision trees.
- **Refit**: after choosing the settings, a model is trained once more on all training months;
  this gave a small, consistent gain.

#### How we score
- **MAE (mean absolute error)**: the average gap, in minutes, between the predicted and the
  real change in delay, ignoring the sign. An MAE of 0.8 means "off by 0.8 minutes on
  average". It is in minutes, so anyone can read it, and a few extreme delays do not dominate
  it. The models predict the *median* outcome, which is exactly what minimises this score.
- **Confidence interval**: the range in which a score difference would fall if the test were
  repeated on other days; computed by resampling whole days thousands of times.
- **Likely range (P10–P90)**: the range in which the real delay should fall for **8 trains
  out of 10**. P10 is the value the change stays below for 10 % of trains; P90 for 90 %.
- **Coverage**: the share of real delays that actually fell inside the range. 80 % means the
  range keeps its promise.
- **Total gain importance**: how much each input reduced XGBoost's error during training,
  summed over all the trees; used to show what M2 relies on.
- **Disrupted day**: a day with unusually high delays and cancellations. Each day's score
  combines its average delay and its share of cancelled trains, compared with the training
  months; the day is disrupted when the score exceeds the level of the worst 5 % of training
  days.

#### Network inputs (M2)
All measured from trains that had already passed **at least 2 minutes before** the
prediction, so nothing from the future is used:
- how many trains passed the current, next and target stations, how late they were, and the
  share more than 3 minutes late;
- how much delay trains gained or lost on the next segment, and on the whole route to the
  target stop;
- the train just ahead on the same segment: how long ago it left, how late it was, and how
  much delay it gained;
- the same summaries over the last 10 and 60 minutes, to see whether things get better or
  worse, and the average delay over the whole Swiss network.

#### In production
- **Calibration of the likely range**: each end of the range is widened by a margin learned
  from real outcomes, just enough to have 10 % of real delays below and 10 % above,
  separately for trains on time, 1–3, 3–10 and more than 10 minutes late.
- **Nightly update**: each night, these margins are recomputed from the last 14 days.
- **Drift score (PSI, population stability index)**: measures how different the day's trains
  are from the training months, for three inputs (current delays, hours of departure, traffic
  at stations). Below 0.1 = no change; above 0.25 = alert.
- **Alerts**: raised, per horizon, when the day's MAE is more than 25 % above the median of
  the previous 14 days (error jump), when fewer than 70 % of real delays fall inside the range
  (low coverage), or when a drift score exceeds 0.25 (drift).
- **Data**: SBB "actual data" (*Ist-Daten*), the official open record of every scheduled and
  actual arrival and departure in Switzerland.
"""
    )
    pager("glossary")


# =========================================================================== navigation
PAGES.update({
    "overview": st.Page(overview, title="Overview", icon=":material/home:", default=True),
    "data": st.Page(data_page, title="Data", icon=":material/database:", url_path="data"),
    "method": st.Page(method_page, title="How it works", icon=":material/schema:",
                      url_path="method"),
    "results": st.Page(results_page, title="Results", icon=":material/leaderboard:",
                       url_path="results"),
    "production": st.Page(production_page, title="In production", icon=":material/monitoring:",
                          url_path="production"),
    "try": st.Page(try_page, title="Try it", icon=":material/train:", url_path="try"),
    "engineering": st.Page(engineering_page, title="Under the hood", icon=":material/build:",
                           url_path="engineering"),
    "glossary": st.Page(glossary_page, title="Glossary", icon=":material/menu_book:",
                        url_path="glossary"),
})  # fmt: skip
pages = {
    "The project": [PAGES["overview"], PAGES["data"], PAGES["method"]],
    "Results": [PAGES["results"], PAGES["production"], PAGES["try"]],
    "Behind the scenes": [PAGES["engineering"], PAGES["glossary"]],
}
page = st.navigation(pages)
with st.sidebar:
    st.caption(f"[Code on GitHub]({GITHUB})")
page.run()
