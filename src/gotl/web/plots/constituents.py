"""Tidal constituent amplitude/phase charts."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gotl import CONSTITUENTS
from gotl.io.cache import StationCache


def constituent_chart(results_dir: Path, stnm: str) -> dict | None:
    """Amplitude bar chart with error bars + phase chart for all components."""
    cache = StationCache(Path(results_dir) / f"{stnm}.h5")
    if not cache.has("otl_coeff"):
        return None

    df = cache.read_otl_coeff()  # index=CONSTITUENTS

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Amplitude (mm)", "Phase (deg)"),
        vertical_spacing=0.15,
    )

    colors = {"U": "#3498db", "S": "#2ecc71", "W": "#e74c3c"}
    labels = {"U": "Up", "S": "South", "W": "West"}

    for comp in ("U", "S", "W"):
        amp_col = f"d{comp}"
        phase_col = f"g{comp}"
        sd_col = f"sd{comp}"

        amps_mm = df[amp_col].values * 1e3
        sds_mm = df[sd_col].values * 1e3 if sd_col in df.columns else None

        fig.add_trace(
            go.Bar(
                x=CONSTITUENTS, y=amps_mm, name=labels[comp],
                marker_color=colors[comp],
                error_y=dict(type="data", array=sds_mm, visible=True) if sds_mm is not None else None,
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(
                x=CONSTITUENTS, y=df[phase_col].values, name=labels[comp],
                marker_color=colors[comp], showlegend=False,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        barmode="group",
        height=600,
        margin=dict(l=60, r=20, t=40, b=40),
        title_text=f"{stnm.upper()} — Tidal Constituents (GPS)",
    )
    return fig.to_dict()
