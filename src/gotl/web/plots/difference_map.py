"""Geographic map of GPS vs ocean-tide-model differences.

Per-station scatter_geo colored by a chosen disagreement metric between
the VTide-derived OTL coefficient and a selected ocean tide model, for
a given constituent and component.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from gotl.compare.residuals import compare_station

if TYPE_CHECKING:
    from gotl.registry import StationMeta


METRICS = ("damp", "abs_damp", "vmfit")


def _metric_label(metric: str) -> str:
    return {
        "damp": "obs − model amp (mm)",
        "abs_damp": "|obs − model| amp (mm)",
        "vmfit": "vector misfit (mm)",
    }[metric]


def difference_map(
    registry: dict[str, StationMeta],
    results_dir: Path,
    otl_params_dir: Path,
    constituent: str,
    model: str,
    component: str,
    metric: str,
) -> dict:
    """Return Plotly figure dict for the GPS-vs-model difference map."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    lats: list[float] = []
    lons: list[float] = []
    values: list[float] = []
    hovers: list[str] = []

    for stnm, meta in sorted(registry.items()):
        try:
            df = compare_station(stnm, results_dir, otl_params_dir, [model])
        except (FileNotFoundError, KeyError):
            continue

        try:
            row = df.loc[(constituent, component)]
        except KeyError:
            continue

        mod_amp = row.get(f"{model}_amp", np.nan)
        if not np.isfinite(mod_amp):
            continue

        obs_amp = row["obs_amp"]
        obs_pha = row["obs_pha"]
        mod_pha = row[f"{model}_pha"]
        damp_mm = (obs_amp - mod_amp) * 1e3
        vmfit_mm = row[f"{model}_vmfit"] * 1e3

        if metric == "damp":
            value = damp_mm
        elif metric == "abs_damp":
            value = abs(damp_mm)
        else:
            value = vmfit_mm

        lats.append(meta.lat)
        lons.append(meta.lon)
        values.append(value)
        hovers.append(
            f"{stnm.upper()} ({meta.network})<br>"
            f"obs: {obs_amp * 1e3:.2f} mm @ {obs_pha:.1f}°<br>"
            f"{model}: {mod_amp * 1e3:.2f} mm @ {mod_pha:.1f}°<br>"
            f"Δamp: {damp_mm:+.2f} mm · vmfit: {vmfit_mm:.2f} mm"
        )

    title = (
        f"{constituent} {component}-component — {_metric_label(metric)} "
        f"(GPS vs {model})"
    )

    if not values:
        fig = go.Figure()
        fig.update_layout(
            geo=dict(projection_type="natural earth", showland=True,
                     landcolor="rgb(243, 243, 243)"),
            annotations=[dict(
                text=f"No stations available for {constituent} / {model} / {component}.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )],
            margin=dict(l=0, r=0, t=40, b=0),
            height=600,
            title=title,
        )
        return fig.to_dict()

    arr = np.array(values, dtype=float)
    if metric == "damp":
        extent = float(np.nanpercentile(np.abs(arr), 95))
        extent = max(extent, 0.5)
        cmin, cmax = -extent, extent
        colorscale = "RdBu_r"
    else:
        cmax = float(np.nanpercentile(arr, 95))
        cmax = max(cmax, 0.5)
        cmin = 0.0
        colorscale = "Viridis"

    fig = go.Figure(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=hovers,
            hoverinfo="text",
            marker=dict(
                size=10,
                color=values,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title=dict(text=_metric_label(metric))),
                line=dict(width=0.5, color="white"),
            ),
        )
    )
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
            coastlinecolor="rgb(150, 150, 150)",
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title=title,
    )
    return fig.to_dict()
