"""Time series Plotly figure builders."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gotl.io.cache import StationCache

# Downsample threshold
_MAX_POINTS = 50_000


def _downsample(df, max_pts=_MAX_POINTS):
    if len(df) <= max_pts:
        return df
    step = len(df) // max_pts
    return df.iloc[::step]


def enu_timeseries(results_dir: Path, stnm: str) -> dict | None:
    """ENU displacements from the combined cache step.

    Plots u_nop, s_nop, w_nop (SLTM residuals) as E/N/U.
    """
    cache = StationCache(Path(results_dir) / f"{stnm}.h5")
    if not cache.has("combined"):
        return None

    df = cache.read_combined()
    df = _downsample(df)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Up (m)", "North (m)", "East (m)"),
        vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Scattergl(x=df["t"], y=df["u_nop"], mode="markers",
                      marker=dict(size=2, color="#3498db"), name="Up"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=df["t"], y=-df["s_nop"], mode="markers",
                      marker=dict(size=2, color="#2ecc71"), name="North"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=df["t"], y=-df["w_nop"], mode="markers",
                      marker=dict(size=2, color="#e74c3c"), name="East"),
        row=3, col=1,
    )

    fig.update_layout(
        height=700, showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        title_text=f"{stnm.upper()} — SLTM Residuals (ENU)",
    )
    return fig.to_dict()


def hardisp_timeseries(results_dir: Path, stnm: str) -> dict | None:
    """Hardisp forward model time series (dU, dS, dW)."""
    cache = StationCache(Path(results_dir) / f"{stnm}.h5")
    if not cache.has("hardisp"):
        return None

    df = cache.read_hardisp()
    df = _downsample(df)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("dU — Up (m)", "dS — South (m)", "dW — West (m)"),
        vertical_spacing=0.06,
    )
    for i, (col, color) in enumerate(
        [("dU", "#3498db"), ("dS", "#2ecc71"), ("dW", "#e74c3c")], 1
    ):
        fig.add_trace(
            go.Scattergl(x=df["t"], y=df[col], mode="lines",
                          line=dict(width=1, color=color), name=col),
            row=i, col=1,
        )

    fig.update_layout(
        height=700, showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        title_text=f"{stnm.upper()} — Hardisp Forward Model",
    )
    return fig.to_dict()
