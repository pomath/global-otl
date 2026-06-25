"""Tidal harmonic analysis via VTide (variational Bayesian).

Replaces the UTide-based solve_tides.m approach with VTide, which provides
per-constituent amplitude and phase uncertainty estimates in addition to
point estimates.

Phase convention:
    VTide's basis functions are [sin(φ), cos(φ)] where φ = 2π(U+V).
    arctan2(cos_coeff, sin_coeff) yields θ = 90° - G, NOT Greenwich phase lag.
    We convert to standard Greenwich phase lag: G = (90 - θ) % 360.
    Phase uncertainties are returned by VTide in radians; converted to degrees here.

Reference:
    Monahan et al. (2025), JGR Oceans, doi:10.1029/2024JC021533
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from vtide import VTide

from gotl import CONSTITUENTS


def solve_tides(
    df: pd.DataFrame,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Solve for 11 tidal constituents via VTide harmonic analysis.

    Sign conventions and coordinate reordering follow solve_tides.m:
        - dW = positive West  → E component = -dW
        - dS = positive South → N component = -dS
        - dU = positive Up    → U component = +dU
        - UTide/VTide ordering: [U, S, W] = [u_raw, -n_raw, -e_raw]

    Args:
        df: Combined SLTM-processed DataFrame with columns:
            't' (datetime), 'u_nop', 's_nop', 'w_nop' (SLTM residuals, m),
            'dU', 'dS', 'dW' (hardisp forward model, m).
        lat: Station latitude (degrees N).
        lon: Station longitude (degrees E).

    Returns:
        DataFrame with index=CONSTITUENTS, columns:
            dU, dS, dW  — amplitude (m)
            gU, gS, gW  — phase (degrees)
            sdU, sdS, sdW — amplitude uncertainty (m)
            sgU, sgS, sgW — phase uncertainty (degrees)
    """
    # --- Reconstruct full tidal signal: SLTM residuals + hardisp model ---
    # GipsyX applied OTL during PPP, removing tidal loading from positions.
    # SLTM residuals contain the leftover (model error + noise). Adding
    # the hardisp model back recovers the full observed tidal displacement.
    # Sign conventions: dW=+West → East=-dW; dS=+South → North=-dS
    e_raw = df["w_nop"].values + (-df["dW"].values)   # East = w_nop - dW
    n_raw = df["s_nop"].values + (-df["dS"].values)   # North = s_nop - dS
    u_raw = df["u_nop"].values + df["dU"].values       # Up    = u_nop + dU

    # Reorder to [U, S, W] convention (matches MATLAB solve_tides.m)
    u_usw = u_raw
    s_usw = -n_raw
    w_usw = -e_raw

    t_index = pd.DatetimeIndex(df["t"])

    amp_u, pha_u, sdu, sgu = _vtide_solve(u_usw, t_index, lat, lon)
    amp_s, pha_s, sds, sgs = _vtide_solve(s_usw, t_index, lat, lon)
    amp_w, pha_w, sdw, sgw = _vtide_solve(w_usw, t_index, lat, lon)

    return pd.DataFrame(
        {
            "dU": amp_u,
            "dS": amp_s,
            "dW": amp_w,
            "gU": pha_u,
            "gS": pha_s,
            "gW": pha_w,
            "sdU": sdu,
            "sdS": sds,
            "sdW": sdw,
            "sgU": sgu,
            "sgS": sgs,
            "sgW": sgw,
        },
        index=CONSTITUENTS,
    )


def _vtide_solve(
    signal: np.ndarray,
    t_index: pd.DatetimeIndex,
    lat: float,
    lon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run VTide on a single 1D signal.

    Args:
        signal: 1D displacement time series (m), length n.
        t_index: DatetimeIndex of length n (timezone-aware or naive).
        lat: Latitude (degrees N).
        lon: Longitude (degrees E).

    Returns:
        (amplitudes, phases, amp_uncertainties, phase_uncertainties)
        each shape (len(CONSTITUENTS),), in constituent order.
    """
    df_vtide = pd.DataFrame({"observations": signal}, index=t_index)
    model = VTide(df_vtide, lat=lat, lon=lon)
    model.Solve()

    harmonics = model.harmonics
    amps = np.array([harmonics[c]["amp"] if c in harmonics else np.nan for c in CONSTITUENTS])
    raw_phases = np.array([harmonics[c]["phase"] if c in harmonics else np.nan for c in CONSTITUENTS])
    amp_uncert = np.array([harmonics[c]["amp_uncert"] if c in harmonics else np.nan for c in CONSTITUENTS])
    raw_phase_uncert = np.array([harmonics[c]["phase_uncert"] if c in harmonics else np.nan for c in CONSTITUENTS])

    # VTide phase convention: raw phase = 90° - G (Greenwich phase lag).
    # Its basis functions are [sin(φ), cos(φ)] and arctan2(cos_coeff, sin_coeff)
    # yields 90° - G.  Convert to standard Greenwich phase lag (TPXO/FES/GOT convention).
    phases = (90.0 - raw_phases) % 360.0

    # VTide phase_uncert is in radians; convert to degrees.
    phase_uncert = np.rad2deg(raw_phase_uncert)

    return amps, phases, amp_uncert, phase_uncert
