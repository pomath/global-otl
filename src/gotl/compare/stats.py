"""Aggregate statistics across stations for model evaluation.

Computes per-model, per-constituent, and per-region summaries from the
residuals table produced by compare_station().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gotl import CONSTITUENTS


def rms_by_model(
    residuals: pd.DataFrame,
    models: list[str],
    constituent: str | None = None,
    component: str | None = None,
) -> pd.DataFrame:
    """Compute RMS vector misfit per model, optionally filtered.

    Args:
        residuals: Concatenated output of compare_station() across stations,
                   with an additional 'stnm' column.
        models: List of model names.
        constituent: Filter to one constituent (e.g. "M2"); None = all.
        component: Filter to one component ("U", "S", or "W"); None = all.

    Returns:
        DataFrame with index=models, columns=['rms_vmfit', 'n_stations'].
    """
    df = residuals.reset_index()
    if constituent:
        df = df[df["constituent"] == constituent]
    if component:
        df = df[df["component"] == component]

    rows = []
    for model in models:
        col = f"{model}_vmfit"
        if col not in df.columns:
            rows.append({"model": model, "rms_vmfit": np.nan, "n_stations": 0})
            continue
        vals = df[col].dropna()
        rows.append({
            "model": model,
            "rms_vmfit": float(np.sqrt(np.mean(vals**2))) if len(vals) > 0 else np.nan,
            "n_stations": df["stnm"].nunique(),
        })

    return pd.DataFrame(rows).set_index("model")


def constituent_summary(
    residuals: pd.DataFrame,
    models: list[str],
    component: str = "U",
) -> pd.DataFrame:
    """RMS vector misfit per constituent per model for one component.

    Args:
        residuals: Concatenated residuals table with 'stnm' column.
        models: List of model names.
        component: "U", "S", or "W".

    Returns:
        DataFrame with index=CONSTITUENTS, columns=models (RMS vmfit in metres).
    """
    df = residuals.reset_index()
    df = df[df["component"] == component]

    result = pd.DataFrame(index=CONSTITUENTS, columns=models, dtype=float)
    for const in CONSTITUENTS:
        sub = df[df["constituent"] == const]
        for model in models:
            col = f"{model}_vmfit"
            if col in sub.columns:
                vals = sub[col].dropna()
                result.loc[const, model] = float(np.sqrt(np.mean(vals**2))) if len(vals) else np.nan
    return result
