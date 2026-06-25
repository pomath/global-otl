"""Per-station processing pipeline.

Phases:
    1. Load TDP GPS data                → cache group 'tdp'
    2. Load hardisp forward model       → cache group 'hardisp'
    3. Combine + SLTM fit               → cache groups 'combined', 'combined_clean'
    4. VTide harmonic analysis          → cache group 'otl_coeff'
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gotl.config import Config
from gotl.coords.transforms import ecef_to_enu, ecef_to_lla
from gotl.io.cache import StationCache
from gotl.io.hardisp import load_hardisp
from gotl.io.tdp import load_tdp
from gotl.registry import StationMeta
from gotl.sltm.fit import fit_sltm
from gotl.tides.solve import solve_tides

log = logging.getLogger(__name__)

# Outlier rejection before VTide. Drop epochs where any component's residual
# exceeds k·σ_robust, with σ_robust = 1.4826·MAD per component. k=4.5 is a
# loose cushion above 3σ that still catches GipsyX-divergence-class outliers
# (CHUR: 10+ km position spikes) without flagging ordinary heavy-tailed PPP
# noise (cant/abmf: 2-5 cm σ, p99 ~15 cm). The earlier swts-based threshold
# was tied to a hardcoded 1 cm IRWLS sigma and dropped 18-67% of data on
# stations whose real noise exceeded that floor.
ROBUST_SIGMA_K = 4.5

# Floor on σ_robust to avoid divide-by-near-zero on synthetic flat residuals.
# 1 cm matches the previous IRWLS sigma so existing tests with cm-noise +
# m-scale spikes still flag the spikes.
ROBUST_SIGMA_FLOOR = 0.01


def process_station(
    meta: StationMeta,
    cfg: Config,
    force: bool = False,
) -> None:
    """Run the full pipeline for one station.

    Args:
        meta: Station metadata from registry (provides lat/lon/alt/jumps).
        cfg: Config with path references.
        force: If True, re-run all steps even if cached.
    """
    stnm = meta.stnm
    cache = StationCache(cfg.results_dir / f"{stnm}.h5")

    # Step 1: Load TDP
    if force or not cache.has("tdp"):
        log.info("%s: loading TDP data", stnm)
        df_tdp = load_tdp(stnm, cfg.tdp_root)
        cache.write_tdp(df_tdp)
    else:
        log.info("%s: TDP cached", stnm)

    # Step 2: Load hardisp
    if force or not cache.has("hardisp"):
        log.info("%s: loading hardisp data", stnm)
        # Detect year range from TDP data to avoid hardcoded default
        df_tdp = cache.read_tdp() if cache.has("tdp") else load_tdp(stnm, cfg.tdp_root)
        yr_min = df_tdp["t"].min().year
        yr_max = df_tdp["t"].max().year
        years = [str(y) for y in range(yr_min, yr_max + 1)]
        df_hardisp = load_hardisp(stnm, cfg.hardisp_root, years=years)
        cache.write_hardisp(df_hardisp)
    else:
        log.info("%s: hardisp cached", stnm)

    # Step 3: Combine + SLTM fit (writes both /combined and /combined_clean)
    if force or not cache.has("combined") or not cache.has("combined_clean"):
        log.info("%s: combining and fitting SLTM", stnm)
        df_tdp = cache.read_tdp()
        df_hardisp = cache.read_hardisp()
        df_combined, _swts = _combine_and_fit(df_tdp, df_hardisp, meta)
        df_clean, stats = _mask_outliers(df_combined)
        cache.write_combined(df_combined, meta, stats=stats)
        cache.write_combined_clean(df_clean, meta)
        if stats["n_total"] > 0 and stats["n_dropped"] / stats["n_total"] > 0.001:
            log.warning(
                "%s: dropped %d/%d epochs (%.2f%%) as outliers; max |resid| "
                "u=%.2fm s=%.2fm w=%.2fm",
                stnm,
                stats["n_dropped"],
                stats["n_total"],
                100 * stats["n_dropped"] / stats["n_total"],
                stats["max_abs_u"],
                stats["max_abs_s"],
                stats["max_abs_w"],
            )
    else:
        log.info("%s: combined cached", stnm)

    # Step 4: VTide harmonic analysis (operates on the outlier-filtered series)
    if force or not cache.has("otl_coeff"):
        log.info("%s: solving tides with VTide", stnm)
        df_clean = cache.read_combined_clean()
        df_coeff = solve_tides(df_clean, lat=meta.lat, lon=meta.lon)
        cache.write_otl_coeff(df_coeff)
    else:
        log.info("%s: OTL coefficients cached", stnm)

    log.info("%s: complete", stnm)


def _mask_outliers(
    df: pd.DataFrame,
    k: float = ROBUST_SIGMA_K,
    sigma_floor: float = ROBUST_SIGMA_FLOOR,
) -> tuple[pd.DataFrame, dict]:
    """Drop rows where any component's residual exceeds k·σ_robust.

    σ_robust = 1.4826·MAD per component, floored at sigma_floor to keep the
    threshold sensible when the residual distribution is degenerate (e.g.,
    synthetic flat data + a few large spikes). VTide has no outlier resistance,
    so the goal is to suppress GipsyX-divergence-class spikes (m-to-km scale)
    without trimming ordinary fat-tailed PPP noise (cm scale).

    Args:
        df: combined DataFrame (columns t, u_nop, s_nop, w_nop, dU, dS, dW).
        k: threshold multiplier; default 4.5 (≈3σ + cushion).
        sigma_floor: minimum σ_robust applied per component (default 1 cm).

    Returns:
        (df_clean, stats) where stats has n_total, n_dropped, and the max
        absolute residual per component (across all rows, pre-mask).
    """
    keep = np.ones(len(df), dtype=bool)
    for col in ("u_nop", "s_nop", "w_nop"):
        x = df[col].values
        med = float(np.median(x))
        sig = 1.4826 * float(np.median(np.abs(x - med)))
        sig_eff = max(sig, sigma_floor)
        bad = np.abs(x - med) > k * sig_eff
        keep &= ~bad
    n_total = int(len(df))
    n_dropped = int((~keep).sum())
    stats = {
        "n_total": n_total,
        "n_dropped": n_dropped,
        "max_abs_u": float(np.max(np.abs(df["u_nop"].values))) if n_total else 0.0,
        "max_abs_s": float(np.max(np.abs(df["s_nop"].values))) if n_total else 0.0,
        "max_abs_w": float(np.max(np.abs(df["w_nop"].values))) if n_total else 0.0,
    }
    df_clean = df.loc[keep].reset_index(drop=True)
    return df_clean, stats


def _combine_and_fit(
    df_tdp: pd.DataFrame,
    df_hardisp: pd.DataFrame,
    meta: StationMeta,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Join TDP + hardisp, convert to ENU, fit SLTM, return residuals.

    GipsyX applies OTL during PPP, so TDP positions have tidal loading
    already removed. SLTM fits directly to ENU displacements (no hardisp
    subtraction needed). The hardisp columns are carried through so that
    solve_tides can add the known OTL correction back to SLTM residuals,
    recovering the full tidal signal for VTide analysis.

    Returns:
        (df, swts) where df has columns
            t, u_nop, s_nop, w_nop (SLTM residuals, m)
            dU, dS, dW (hardisp forward model, m)
        and swts is the (3, nt) per-component IRWLS sqrt-weight array from
        FitResult (rows are [E, N, U], matching the ENU fit ordering).
    """
    # Inner-join on timestamp, remove duplicates
    df = df_tdp.merge(df_hardisp[["t", "dU", "dS", "dW"]], on="t", how="inner")
    df = df.drop_duplicates(subset="t", keep="first").sort_values("t").reset_index(drop=True)

    # ECEF → ENU using station reference position (first epoch mean)
    x0 = df["X"].median()
    y0 = df["Y"].median()
    z0 = df["Z"].median()
    lat0, lon0, alt0 = ecef_to_lla(x0, y0, z0)
    e, n, u = ecef_to_enu(df["X"].values, df["Y"].values, df["Z"].values, lat0, lon0, alt0)

    # Decimal year for SLTM
    t_decyear = df["t"].apply(_decyear).values
    tref = _decyear(df["t"].mean())

    # Jump epochs from registry
    tjmp = [_decyear(j) for j in meta.jump_epochs] if meta.jump_epochs else None

    # SLTM fit: remove trend + jumps + oscillations from each component.
    # GipsyX applies OTL during PPP, so tidal loading is already removed
    # from the positions. Fit SLTM directly to the ENU displacements.
    # The hardisp values are carried through so solve_tides can add
    # the known correction back to recover the full tidal signal.
    pvs = np.array([e, n, u])

    # Estimate noise from coordinate sigmas if available, else use 1 cm default
    sig = 0.01  # 1 cm default noise

    # Polynomial order 1 (constant + velocity); annual + semi-annual oscillations
    fit = fit_sltm(
        t_decyear,
        tref,
        pvs,
        sig=sig,
        n=1,
        tjmp=np.array(tjmp) if tjmp else None,
        fperiods=np.array([1.0, 0.5]),  # annual, semi-annual (years)
    )

    # Residuals after SLTM: input - fitted curve
    # cpvs shape is (3, nt) = fitted trajectory values
    e_resid = e - fit.cpvs[0, :]
    n_resid = n - fit.cpvs[1, :]
    u_resid = u - fit.cpvs[2, :]

    # Convention: w_nop = -e_residual, s_nop = -n_residual, u_nop = u_residual
    df_out = pd.DataFrame(
        {
            "t": df["t"].values,
            "u_nop": u_resid,
            "s_nop": -n_resid,
            "w_nop": -e_resid,
            "dU": df["dU"].values,
            "dS": df["dS"].values,
            "dW": df["dW"].values,
        }
    )
    return df_out, fit.swts


def _decyear(t: pd.Timestamp) -> float:
    """Convert Timestamp to decimal year."""
    t = pd.Timestamp(t)
    year = t.year
    t0 = pd.Timestamp(year, 1, 1)
    t1 = pd.Timestamp(year + 1, 1, 1)
    return year + (t - t0).total_seconds() / (t1 - t0).total_seconds()
