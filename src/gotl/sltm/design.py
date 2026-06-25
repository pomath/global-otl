"""Design matrix construction for the Standard Linear Trajectory Model (SLTM).

Replaces: design_sltm.m, design_trend.m, design_jmp.m, design_osc.m

The SLTM has three additive components:
  position(t) = polynomial_trend + heaviside_jumps + fourier_oscillations
"""

from __future__ import annotations

import numpy as np


def design_trend(t: np.ndarray, tref: float, n: int) -> np.ndarray:
    """Design matrix for polynomial trend: columns [1, dt, dt², …, dtⁿ].

    Args:
        t: time vector, length m
        tref: reference epoch (same units as t)
        n: polynomial order (0=constant, 1=velocity, 2=acceleration …)

    Returns:
        Array of shape (m, n+1).
    """
    dt = np.asarray(t, dtype=float).ravel() - tref
    return np.column_stack([dt**j for j in range(n + 1)])


def design_jmp(t: np.ndarray, tjmp: np.ndarray) -> np.ndarray:
    """Design matrix for Heaviside jump terms.

    Args:
        t: time vector, length m
        tjmp: jump times, length nj

    Returns:
        Array of shape (m, nj); H(t - tjmp[j]) = 1 for t >= tjmp[j].
    """
    t = np.asarray(t, dtype=float).ravel()
    tjmp = np.asarray(tjmp, dtype=float).ravel()
    return np.column_stack([(t >= tj).astype(float) for tj in tjmp])


def design_osc(t: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Design matrix for truncated Fourier series oscillations.

    Column order matches MATLAB design_osc.m: S1, C1, S2, C2, …

    Args:
        t: time vector, length m
        periods: oscillation periods (same units as t), length nf

    Returns:
        Array of shape (m, 2*nf).
    """
    t = np.asarray(t, dtype=float).ravel()
    periods = np.asarray(periods, dtype=float).ravel()
    cols = []
    for p in periods:
        f = 2.0 * np.pi / p
        cols.append(np.sin(f * t))
        cols.append(np.cos(f * t))
    return np.column_stack(cols)


def design_sltm(
    t: np.ndarray,
    tref: float,
    n: int,
    tjmp: np.ndarray | None = None,
    fperiods: np.ndarray | None = None,
) -> np.ndarray:
    """Composite SLTM design matrix A = [A_trend | A_jumps | A_osc].

    Args:
        t: time vector, length m
        tref: reference epoch
        n: polynomial order for trend
        tjmp: jump times (None or empty → no jumps)
        fperiods: oscillation periods (None or empty → no oscillations)

    Returns:
        Array of shape (m, n+1 + nj + 2*nf).
    """
    parts = [design_trend(t, tref, n)]
    if tjmp is not None and len(tjmp) > 0:
        parts.append(design_jmp(t, tjmp))
    if fperiods is not None and len(fperiods) > 0:
        parts.append(design_osc(t, fperiods))
    return np.hstack(parts)
