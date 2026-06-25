"""Plot API endpoints — return HTML partials with embedded Plotly."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from gotl.web.deps import get_config, get_registry

router = APIRouter(tags=["plots"])


@router.get("/plots/map")
async def plot_map():
    """Return Plotly JSON for the station map."""
    from gotl.web.plots.map import station_map

    cfg = get_config()
    registry = get_registry()
    fig = station_map(registry, cfg.gotl.results_dir)
    return JSONResponse(fig)


@router.get("/plots/timeseries/{stnm}", response_class=HTMLResponse)
async def plot_timeseries(stnm: str):
    """Return HTML partial with ENU + hardisp time series plots."""
    from gotl.web.plots.timeseries import enu_timeseries, hardisp_timeseries

    cfg = get_config()
    parts = []

    enu = enu_timeseries(cfg.gotl.results_dir, stnm)
    if enu:
        parts.append(_plot_div(f"enu-{stnm}", enu))
    else:
        parts.append("<p>No SLTM residual data. Run the pipeline first.</p>")

    hardisp = hardisp_timeseries(cfg.gotl.results_dir, stnm)
    if hardisp:
        parts.append(_plot_div(f"hardisp-{stnm}", hardisp))
    else:
        parts.append("<p>No hardisp data available.</p>")

    return "\n".join(parts)


@router.get("/plots/constituents/{stnm}", response_class=HTMLResponse)
async def plot_constituents(stnm: str):
    """Return HTML partial with constituent bar charts."""
    from gotl.web.plots.constituents import constituent_chart

    cfg = get_config()
    fig = constituent_chart(cfg.gotl.results_dir, stnm)
    if fig:
        return _plot_div(f"const-{stnm}", fig)
    return "<p>No OTL coefficients available. Run the pipeline first.</p>"


@router.get("/plots/comparison/{stnm}", response_class=HTMLResponse)
async def plot_comparison(stnm: str):
    """Return HTML partial with model comparison charts."""
    from gotl.web.plots.comparison import comparison_chart

    cfg = get_config()
    fig = comparison_chart(
        cfg.gotl.results_dir, cfg.gotl.otl_params_dir,
        stnm, cfg.gotl.tide_models,
    )
    if fig:
        return _plot_div(f"cmp-{stnm}", fig)
    return "<p>No comparison data. Ensure OTL coefficients and model .otl files exist.</p>"


def _plot_div(div_id: str, fig: dict) -> str:
    """Wrap a Plotly figure dict in an HTML div + script."""
    fig_json = json.dumps(fig)
    return (
        f'<div id="{div_id}"></div>\n'
        f"<script>(function(){{ var f={fig_json}; Plotly.newPlot('{div_id}', f.data, f.layout, {{responsive: true}}); }})()</script>"
    )
