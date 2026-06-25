"""Compute GPS-observed vs tide-model OTL residuals, supporting multiple models.

For each station, constituent, and model the key outputs are:
    - amplitude residual (dA = obs - mod), metres
    - phase residual (dg = obs - mod), degrees
    - vector misfit: combined amp+phase scalar (metres)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gotl import CONSTITUENTS
from gotl.io.cache import StationCache
from gotl.io.otl_params import read_otl_params


def compare_station(
    stnm: str,
    results_dir: Path,
    otl_params_dir: Path,
    models: list[str],
) -> pd.DataFrame:
    """Compute residuals between GPS OTL estimates and one or more tide models.

    Args:
        stnm: 4-letter station code.
        results_dir: Directory containing {stnm}.h5 cache files.
        otl_params_dir: Directory containing {stnm}_{model}.otl files.
        models: List of model names (e.g. ["TPXO9", "FES2014b"]).

    Returns:
        DataFrame with MultiIndex (constituent, component) and columns:
            obs_amp, obs_pha, obs_amp_sd, obs_pha_sd  (GPS estimates + uncertainties)
            {model}_amp, {model}_pha                  (model prediction, per model)
            {model}_damp, {model}_dpha, {model}_vmfit (residuals, per model)
        Amplitudes in metres, phases in degrees, vmfit in metres.
    """
    cache = StationCache(Path(results_dir) / f"{stnm}.h5")
    if not cache.has("otl_coeff"):
        raise FileNotFoundError(
            f"OTL coefficients not computed for {stnm}. Run the pipeline first."
        )
    df_obs = cache.read_otl_coeff()

    rows = []
    for const in CONSTITUENTS:
        for comp_amp, comp_pha, comp_sd_amp, comp_sd_pha in [
            ("dU", "gU", "sdU", "sgU"),
            ("dS", "gS", "sdS", "sgS"),
            ("dW", "gW", "sdW", "sgW"),
        ]:
            row: dict = {
                "constituent": const,
                "component": comp_amp[1],  # U, S, or W
                "obs_amp": float(df_obs.loc[const, comp_amp]),
                "obs_pha": float(df_obs.loc[const, comp_pha]),
                "obs_amp_sd": float(df_obs.loc[const, comp_sd_amp]) if comp_sd_amp in df_obs.columns else np.nan,
                "obs_pha_sd": float(df_obs.loc[const, comp_sd_pha]) if comp_sd_pha in df_obs.columns else np.nan,
            }

            for model in models:
                # Check for .otl or .db file
                otl_otl = Path(otl_params_dir) / f"{stnm}_{model}.otl"
                otl_db = Path(otl_params_dir) / f"{stnm}_{model}.db"
                if not otl_otl.exists() and not otl_db.exists():
                    row[f"{model}_amp"] = np.nan
                    row[f"{model}_pha"] = np.nan
                    row[f"{model}_damp"] = np.nan
                    row[f"{model}_dpha"] = np.nan
                    row[f"{model}_vmfit"] = np.nan
                    continue

                df_mod = read_otl_params(stnm, otl_params_dir, suffix=f"_{model}")
                mod_amp = float(df_mod.loc[const, comp_amp])
                mod_pha = float(df_mod.loc[const, comp_pha])

                damp = row["obs_amp"] - mod_amp
                dpha = row["obs_pha"] - mod_pha
                vmfit = _vector_misfit(row["obs_amp"], row["obs_pha"], mod_amp, mod_pha)

                row[f"{model}_amp"] = mod_amp
                row[f"{model}_pha"] = mod_pha
                row[f"{model}_damp"] = damp
                row[f"{model}_dpha"] = dpha
                row[f"{model}_vmfit"] = vmfit

            rows.append(row)

    df = pd.DataFrame(rows).set_index(["constituent", "component"])
    return df


def _vector_misfit(obs_amp: float, obs_pha: float, mod_amp: float, mod_pha: float) -> float:
    """Scalar vector misfit combining amplitude and phase differences.

    Computes |z_obs - z_mod| where z = amp * exp(i * pha_rad).
    Result is in the same units as the amplitudes (metres).
    """
    z_obs = obs_amp * np.exp(1j * np.deg2rad(obs_pha))
    z_mod = mod_amp * np.exp(1j * np.deg2rad(mod_pha))
    return float(np.abs(z_obs - z_mod))
