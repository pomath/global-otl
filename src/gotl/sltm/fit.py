"""Fit a Standard Linear Trajectory Model (SLTM) to 3-component position data.

Replaces: fit_sltm.m (version 2.0)

The SLTM decomposes position as:
    pos(t) = polynomial_trend(t) + heaviside_jumps(t) + fourier_oscillations(t)

Fitting uses IRWLS (iterative re-weighted least squares) for robustness against outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gotl.sltm.design import design_sltm
from gotl.sltm.solver import IRWLSResult, irwls, wls_with_cov


@dataclass
class FitResult:
    """Output from fit_sltm.

    Attributes:
        cpvs: fitted trajectory values, shape (3, nt)
        swts: square-root weights, shape (3, nt); <1 where outliers downweighted
        pout: percent-outlier statistic per component, shape (3,)
        m: concatenated model parameters [m_E | m_N | m_U], length 3*nm
        Cm: covariance matrix of m, shape (3*nm, 3*nm); None if not requested
    """
    cpvs: np.ndarray
    swts: np.ndarray
    pout: np.ndarray
    m: np.ndarray
    Cm: np.ndarray | None = None


def fit_sltm(
    t: np.ndarray,
    tref: float,
    pvs: np.ndarray,
    sig: float | np.ndarray,
    n: int,
    tjmp: np.ndarray | None = None,
    fperiods: np.ndarray | None = None,
    compute_cov: bool = False,
) -> FitResult:
    """Fit SLTM to 3-component position time series using IRWLS.

    Mirrors fit_sltm.m with output order: cpvs, swts, pout, m [, Cm].

    Args:
        t: time vector, length nt (decimal years)
        tref: reference epoch (same units as t)
        pvs: position matrix, shape (3, nt); rows are [E, N, U] or [X, Y, Z]
        sig: noise sigma; scalar, length-3 vector, or (3, nt) matrix
        n: polynomial order for trend (0=constant, 1=velocity, ...)
        tjmp: jump times, or None for no jumps
        fperiods: oscillation periods, or None for no oscillations
        compute_cov: if True, also compute and return Cm (expensive)

    Returns:
        FitResult with cpvs, swts, pout, m, and optionally Cm
    """
    t = np.asarray(t, dtype=float).ravel()
    pvs = np.asarray(pvs, dtype=float)
    if pvs.shape[0] != 3:
        raise ValueError("pvs must have shape (3, nt)")

    nt = pvs.shape[1]
    nj = len(tjmp) if tjmp is not None and len(tjmp) > 0 else 0
    nf = len(fperiods) if fperiods is not None and len(fperiods) > 0 else 0
    nm = n + 1 + nj + 2 * nf

    if nt < nm:
        raise ValueError(
            f"Underdetermined system: nt={nt} observations < nm={nm} model params"
        )

    # Normalize sig to (3, nt) array for uniform per-component indexing
    sig = np.asarray(sig, dtype=float)
    if sig.ndim == 0 or sig.size == 1:
        sig_mat = np.full((3, nt), float(sig))
    elif sig.shape == (3,):
        sig_mat = np.repeat(sig[:, None], nt, axis=1)
    elif sig.shape == (3, nt):
        sig_mat = sig
    else:
        raise ValueError(f"sig has unexpected shape {sig.shape}")

    A = design_sltm(t, tref, n, tjmp, fperiods)

    m_all = np.zeros(3 * nm)
    swts = np.zeros((3, nt))
    cpvs = np.zeros((3, nt))
    pout = np.zeros(3)

    for i in range(3):
        result: IRWLSResult = irwls(A, pvs[i, :], sig=sig_mat[i, :])
        m_all[nm * i : nm * (i + 1)] = result.x
        swts[i, :] = result.swts
        cpvs[i, :] = result.bc
        pout[i] = result.pout

    Cm = None
    if compute_cov and nt > nm:
        Cm = np.zeros((3 * nm, 3 * nm))
        m_cov = np.zeros(3 * nm)
        cpvs_cov = np.zeros((3, nt))
        for i in range(3):
            sgm = sig_mat[i, :] / swts[i, :]
            x1, Cm1, _ = wls_with_cov(A, pvs[i, :], sgm)
            cpvs_cov[i, :] = A @ x1
            m_cov[nm * i : nm * (i + 1)] = x1
            Cm[nm * i : nm * (i + 1), nm * i : nm * (i + 1)] = Cm1
        m_all = m_cov
        cpvs = cpvs_cov

    return FitResult(cpvs=cpvs, swts=swts, pout=pout, m=m_all, Cm=Cm)
