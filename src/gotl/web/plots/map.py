"""Station map using Plotly scatter_geo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from gotl.registry import StationMeta


def station_map(
    registry: dict[str, StationMeta],
    results_dir: Path,
) -> dict:
    """Return Plotly JSON for a station map colored by completion status."""
    from gotl.io.cache import StationCache

    lats, lons, texts, colors = [], [], [], []

    for stnm, meta in sorted(registry.items()):
        cache = StationCache(Path(results_dir) / f"{stnm}.h5")
        steps_done = sum(
            cache.has(step) for step in ("tdp", "hardisp", "combined", "otl_coeff")
        )

        lats.append(meta.lat)
        lons.append(meta.lon)
        texts.append(f"{stnm.upper()} ({meta.network})<br>{steps_done}/4 steps")

        if steps_done == 4:
            colors.append("#2ecc71")  # green
        elif steps_done > 0:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    fig = go.Figure(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=texts,
            hoverinfo="text",
            marker=dict(size=8, color=colors, line=dict(width=0.5, color="white")),
            customdata=[stnm for stnm in sorted(registry.keys())],
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
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )
    return fig.to_dict()
