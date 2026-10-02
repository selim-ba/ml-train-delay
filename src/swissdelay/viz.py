"""Shared chart style for notebooks and reports (Plotly)."""

import plotly.graph_objects as go
import plotly.io as pio

# Fixed categorical order (validated palette, see the dataviz guidelines)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY, LIGHT_GRAY = "#8a8984", "#d4d3cd"

SCOPE_CATEGORIES = ("IC", "IR", "RE", "EC")
CAT_COLORS = dict(zip(SCOPE_CATEGORIES, SERIES, strict=False))
HORIZON_COLORS = {15: SERIES[0], 30: SERIES[1], 60: SERIES[2]}
SPLIT_COLORS = {"train": SERIES[0], "valid": SERIES[1], "test": SERIES[2], "production": GRAY}
STATUS_COLORS = {
    "ok": SERIES[0],
    "terminates": LIGHT_GRAY,
    "cancelled": SERIES[7],
    "not_measured": SERIES[3],
}


def use_style() -> None:
    """Register and activate the project's Plotly template."""
    pio.templates["swissdelay"] = go.layout.Template(
        layout=dict(
            colorway=SERIES,
            font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0b0b0b"),
            paper_bgcolor="#fcfcfb",
            plot_bgcolor="#fcfcfb",
            xaxis=dict(gridcolor="#e8e7e2", zeroline=False, linecolor="#c3c2b7"),
            yaxis=dict(gridcolor="#e8e7e2", zeroline=False, linecolor="#c3c2b7"),
            bargap=0.25,
            margin=dict(l=60, r=30, t=60, b=50),
        )
    )
    pio.templates.default = "plotly_white+swissdelay"
