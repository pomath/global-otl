"""Model comparison charts."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gotl import CONSTITUENTS
from gotl.compare.residuals import compare_station


def comparison_chart(
    results_dir: Path, otl_params_dir: Path,
    stnm: str, models: list[str],
) -> dict | None:
    """GPS vs model amplitude comparison for all components."""
    try:
        df = compare_station(stnm, results_dir, otl_params_dir, models)
    except FileNotFoundError:
        return None

    df = df.reset_index()

    fig = make_subplots(
        rows=len(models), cols=1,
        subplot_titles=[f"Vector Misfit — {m} (mm)" for m in models],
        vertical_spacing=0.12,
    )

    colors = {"U": "#3498db", "S": "#2ecc71", "W": "#e74c3c"}

    for i, model in enumerate(models, 1):
        vmfit_col = f"{model}_vmfit"
        if vmfit_col not in df.columns:
            continue
        for comp in ("U", "S", "W"):
            sub = df[df["component"] == comp]
            fig.add_trace(
                go.Bar(
                    x=sub["constituent"],
                    y=sub[vmfit_col] * 1e3,
                    name=comp if i == 1 else None,
                    marker_color=colors[comp],
                    showlegend=(i == 1),
                ),
                row=i, col=1,
            )

    fig.update_layout(
        barmode="group",
        height=350 * max(len(models), 1),
        margin=dict(l=60, r=20, t=40, b=40),
        title_text=f"{stnm.upper()} — Model Comparison",
    )
    return fig.to_dict()
