"""Compare API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from gotl import CONSTITUENTS
from gotl.web.app import templates
from gotl.web.deps import get_config, get_registry

router = APIRouter(tags=["compare"])


@router.get("/compare/map")
async def compare_map(
    constituent: str,
    model: str,
    component: str = "U",
    metric: str = "vmfit",
):
    """Plotly JSON for a per-station GPS-vs-model difference map."""
    from gotl.web.plots.difference_map import METRICS, difference_map

    cfg = get_config()
    registry = get_registry()

    if constituent not in CONSTITUENTS:
        raise HTTPException(400, f"unknown constituent {constituent!r}")
    if model not in cfg.gotl.tide_models:
        raise HTTPException(400, f"unknown model {model!r}")
    if component not in ("U", "S", "W"):
        raise HTTPException(400, f"component must be U, S, or W")
    if metric not in METRICS:
        raise HTTPException(400, f"metric must be one of {METRICS}")

    fig_dict = difference_map(
        registry=registry,
        results_dir=cfg.gotl.results_dir,
        otl_params_dir=cfg.gotl.otl_params_dir,
        constituent=constituent,
        model=model,
        component=component,
        metric=metric,
    )
    return JSONResponse(fig_dict)


@router.get("/compare/summary", response_class=HTMLResponse)
async def compare_summary(request: Request, component: str = "U"):
    """Aggregate RMS vector misfit across all processed stations."""
    from gotl.compare.residuals import compare_station
    from gotl.compare.stats import constituent_summary, rms_by_model

    cfg = get_config()
    registry = get_registry()
    models = cfg.gotl.tide_models

    all_residuals = []
    for stnm in sorted(registry):
        try:
            df = compare_station(stnm, cfg.gotl.results_dir, cfg.gotl.otl_params_dir, models)
            df = df.reset_index()
            df["stnm"] = stnm
            all_residuals.append(df)
        except (FileNotFoundError, KeyError):
            continue

    if not all_residuals:
        return templates.TemplateResponse(request, "_partials/compare_summary.html", {
            "component": component,
            "stations_used": 0,
            "rms_table": None,
            "const_fig": None,
            "const_fig_json": "null",
        })

    combined = pd.concat(all_residuals, ignore_index=True)
    n_stations = combined["stnm"].nunique()

    rms = rms_by_model(combined.set_index(["constituent", "component"]),
                       models, component=component)
    rms_table = []
    for model in models:
        if model in rms.index:
            rms_table.append({
                "model": model,
                "rms_mm": float(rms.loc[model, "rms_vmfit"]) * 1e3,
                "n_stations": int(rms.loc[model, "n_stations"]),
            })

    const_df = constituent_summary(
        combined.set_index(["constituent", "component"]),
        models, component=component,
    )

    # Build Plotly bar chart
    import plotly.graph_objects as go
    fig = go.Figure()
    for model in models:
        if model in const_df.columns:
            vals = const_df[model].values * 1e3
            fig.add_trace(go.Bar(
                x=const_df.index.tolist(),
                y=[float(v) if not np.isnan(v) else 0 for v in vals],
                name=model,
            ))
    fig.update_layout(
        barmode="group",
        title=f"RMS Vector Misfit by Constituent — {component} Component",
        yaxis_title="RMS Vmfit (mm)",
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    const_fig_json = json.dumps(fig.to_dict())

    return templates.TemplateResponse(request, "_partials/compare_summary.html", {
        "component": component,
        "stations_used": n_stations,
        "rms_table": rms_table,
        "const_fig": True,
        "const_fig_json": const_fig_json,
    })
