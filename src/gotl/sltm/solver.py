"""Robust LS solvers: IRWLS and WLS with covariance output.

Replaces: itrwtdls.m, lscov3d.m
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import iqr

from gotl.sltm.weights import robust_weights


@dataclass
class IRWLSResult:
    x: np.ndarray        # solution vector, shape (m,)
    bc: np.ndarray       # fitted values A @ x, shape (n,)
    resids: np.ndarray   # residuals b - bc, shape (n,)
    swts: np.ndarray     # square-root weights from last iteration, shape (n,)
    pout: float          # percent-outlier statistic from last converged iteration
    n_iter: int          # actual number of iterations completed


def irwls(
    A: np.ndarray,
    b: np.ndarray,
    sig: float | np.ndarray,
    mxiter: int = 7,
) -> IRWLSResult:
    """Robust iterative re-weighted least squares (IRWLS).

    Solves Ax ≈ b robustly by downweighting outliers (|residual| > 2.5 sigma)
    using an exponential weight function, with optional sigma annealing when
    the initial outlier fraction is large.

    Mirrors itrwtdls.m (version 1.6) exactly, including the annealing schedule
    and convergence criterion (|Δpout| < 0.5%).

    Args:
        A: design matrix, shape (n, m), must be overdetermined (n > m)
        b: data vector, shape (n,)
        sig: nominal std deviation for 'good' data; scalar or array of length n
        mxiter: maximum iterations (must be >= 4)

    Returns:
        IRWLSResult with solution, fitted values, residuals, weights, pout, n_iter
    """
    if mxiter < 4 or mxiter != int(mxiter):
        raise ValueError("mxiter must be an integer >= 4")

    b = np.asarray(b, dtype=float).ravel()
    sig = np.asarray(sig, dtype=float)
    if sig.ndim == 0:
        sig = sig.reshape(1)  # keep as 1-element array for uniform handling
    n, m = A.shape
    if len(b) != n or (len(sig) != n and len(sig) != 1):
        raise ValueError("A, b and sig have inconsistent sizes")

    # --- Initial weighted LS solve ---
    D = A / sig[:, None] if len(sig) > 1 else A / sig[0]
    d = b / sig if len(sig) > 1 else b / sig[0]
    x, _, _, _ = np.linalg.lstsq(D, d, rcond=None)
    bc = A @ x
    resids = b - bc

    iqrd = iqr(resids)
    swts = robust_weights(resids, sig if len(sig) > 1 else sig[0])
    pout = 100.0 * (n - swts.sum()) / n

    # --- Build sigma annealing schedule if many outliers ---
    sig_scalar = len(sig) == 1
    sig1 = iqrd  # IQR-derived sigma (conservative starting point)

    if pout > 15 or (np.any(iqrd / sig > 3)):
        nstep = 3
        if pout > 25:
            nstep = 4
        if pout > 50:
            nstep = 5
        if sig_scalar:
            # shape (nstep,): annealing from sig1 to sig[0]
            sigs = np.exp(np.linspace(np.log(sig1), np.log(sig[0]), nstep))
        else:
            # shape (n, nstep): per-observation annealing
            sigs = np.zeros((n, nstep))
            for i in range(n):
                sigs[i, :] = np.exp(np.linspace(np.log(sig1), np.log(sig[i]), nstep))
    else:
        nstep = 1
        sigs = None  # not used

    # --- Iterative reweighting loop ---
    pinc = 0.5
    lastpout = pout
    iter_count = 0

    for iter_1 in range(1, mxiter + 2):  # iter_1 is 1-based, matches MATLAB
        iter_count = iter_1

        # Choose sigma for robust_wt_fn (annealing schedule or original)
        if sigs is not None and iter_1 < nstep:
            if sig_scalar:
                sigma_wt = sigs[iter_1 - 1]   # scalar
            else:
                sigma_wt = sigs[:, iter_1 - 1]  # array (n,)
        else:
            sigma_wt = sig[0] if sig_scalar else sig

        swts = robust_weights(resids, sigma_wt)
        pout = 100.0 * (n - swts.sum()) / n

        # Convergence check (after first iteration)
        if iter_1 > 1:
            if abs(lastpout - pout) < pinc:
                iter_count = iter_1 - 1
                break

        # WLS solve with effective per-observation sigma = sig / swts
        sigma_eff = (sig[0] if sig_scalar else sig) / swts
        D = A / sigma_eff[:, None]
        d = b / sigma_eff
        x, _, _, _ = np.linalg.lstsq(D, d, rcond=None)
        bc = A @ x
        resids = b - bc
        lastpout = pout

    return IRWLSResult(x=x, bc=bc, resids=resids, swts=swts, pout=lastpout, n_iter=iter_count)


def wls_with_cov(
    A: np.ndarray,
    b: np.ndarray,
    sigb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted least squares with solution covariance matrix.

    Solves the WLS problem Ax = b with diagonal data covariance, estimating
    the variance of unit weight to scale the output covariance.

    Mirrors lscov3d.m (version 1.0). Reference: Hamilton (1964).

    Args:
        A: design matrix, shape (m, n), must be overdetermined (m > n)
        b: data vector, shape (m,)
        sigb: data standard deviations, shape (m,)

    Returns:
        (x, Cx, sduw):
            x: solution, shape (n,)
            Cx: solution covariance matrix, shape (n, n)
            sduw: standard deviation of unit weight (scalar)
    """
    m, n = A.shape
    if m <= n:
        raise ValueError("System must be overdetermined (m > n)")
    b = np.asarray(b, dtype=float).ravel()
    sigb = np.asarray(sigb, dtype=float).ravel()

    Ap = A / sigb[:, None]
    B = Ap.T @ Ap
    iB = np.linalg.inv(B)
    x = iB @ Ap.T @ (b / sigb)
    r = b - A @ x
    s2 = float(r @ (r / sigb**2)) / (m - n)
    Cx = s2 * iB
    sduw = float(np.sqrt(s2))
    return x, Cx, sduw
