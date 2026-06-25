"""Robust weighting function for IRWLS outlier rejection.

Replaces: robust_wt_fn.m

Weight formula: for |residual|/sigma > 2.5:
    w = 10 ** ((2.5 - dev) / 2)
where dev = |residual| / sigma.  Points within 2.5-sigma get weight 1.
All weights are floored at machine epsilon to prevent division by zero.
"""

import numpy as np


def robust_weights(resids: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    """Compute square-root weights for the next IRWLS iteration.

    Args:
        resids: residual vector, shape (n,)
        sigma: scalar or array of same shape as resids

    Returns:
        swts: square-root weights, same shape as resids.
              Values in [eps, 1]. Points with |resid|/sigma > 2.5 are downweighted.
    """
    resids = np.asarray(resids, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    dev = np.abs(resids) / sigma
    swts = np.ones_like(dev)
    mask = dev > 2.5
    swts[mask] = 10.0 ** ((2.5 - dev[mask]) / 2.0)
    swts = np.maximum(swts, np.finfo(float).eps)
    return swts
