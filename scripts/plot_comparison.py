#!/usr/bin/env python3
"""Compare global_otl results with previous otl_work pipeline.

Compares:
1. Hardisp time series (should be identical — same binary + same OTL params)
2. M2 amplitudes from three sources: PREM prediction, FES2014b model, GPS (synthetic)
3. FES2014b model coefficients (same input file) — sanity check
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from gotl.io.hardisp import load_hardisp
from gotl.io.otl_params import read_otl_params
from gotl.io.cache import StationCache
from gotl import CONSTITUENTS


OUTDIR = Path("plots")
STNM = "aboa"

# ── Paths ──
NEW_HARDISP_ROOT = Path("/tmp/gotl_demo_rjjczazg/hardisp")
OLD_HARDISP_ROOT = Path("/path/to/otl_work/otl_proc/forward_modeling/hardisp")
NEW_OTL_DIR = Path("/tmp/gotl_demo_rjjczazg/otl_params")
OLD_OTL_DIR = Path("/path/to/otl_work/otl_proc/otl_params")
RESULTS_DIR = Path("/tmp/gotl_demo_rjjczazg/results")
M2_CSV = Path("/path/to/otl_work/otl_proc/station_selection/m2ampphase.csv")


def load_old_hardisp(stnm: str, root: Path, year: int) -> pd.DataFrame:
    """Load old-format hardisp (5-col: year doy dU dS dW)."""
    path = root / f"{stnm}_hardisp" / f"{stnm}-hardisp_{year}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["Year", "DOY", "dU", "dS", "dW"])
    # Build timestamps
    base = pd.Timestamp(f"{year}-01-01")
    n = len(df)
    samples_per_day = 288
    seconds_per_sample = 300  # 5 min
    t = []
    for i in range(n):
        doy = int(df.iloc[i]["DOY"])
        sample_in_day = i % samples_per_day
        dt = pd.Timedelta(days=doy - 1, seconds=sample_in_day * seconds_per_sample)
        t.append(base + dt)
    df["t"] = t
    return df


def plot_hardisp_comparison():
    """Plot 6: Overlay old vs new hardisp for first 3 days."""
    print("  Generating hardisp comparison...")

    df_new = load_hardisp(STNM, NEW_HARDISP_ROOT)
    df_old = load_old_hardisp(STNM, OLD_HARDISP_ROOT, 2010)

    # First 3 days
    t_end = df_new["t"].min() + pd.Timedelta(days=3)
    new = df_new[df_new["t"] <= t_end].reset_index(drop=True)
    old = df_old[df_old["t"] <= t_end].reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    for ax, col, label, color_new, color_old in zip(
        axes,
        ["dU", "dS", "dW"],
        ["dU — Up (mm)", "dS — South+ (mm)", "dW — West+ (mm)"],
        ["#F44336", "#4CAF50", "#2196F3"],
        ["#B71C1C", "#1B5E20", "#0D47A1"],
    ):
        ax.plot(new["t"], new[col] * 1000, "-", color=color_new, linewidth=1.5,
                label="global_otl (new)", alpha=0.8)
        ax.plot(old["t"], old[col] * 1000, "--", color=color_old, linewidth=1.5,
                label="otl_work (old)", alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    # Compute max absolute difference
    n_compare = min(len(new), len(old))
    max_diff_dU = np.max(np.abs(new["dU"].values[:n_compare] - old["dU"].values[:n_compare])) * 1000
    max_diff_dS = np.max(np.abs(new["dS"].values[:n_compare] - old["dS"].values[:n_compare])) * 1000
    max_diff_dW = np.max(np.abs(new["dW"].values[:n_compare] - old["dW"].values[:n_compare])) * 1000

    axes[0].set_title(
        f"{STNM.upper()} — Hardisp Comparison (first 3 days)\n"
        f"Max diff: dU={max_diff_dU:.6f} mm, dS={max_diff_dS:.6f} mm, dW={max_diff_dW:.6f} mm"
    )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    axes[-1].set_xlabel("Date (2010)")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{STNM}_06_hardisp_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {STNM}_06_hardisp_comparison.png")


def plot_m2_multi_source():
    """Plot 7: M2 amplitude from 3 sources — PREM, FES2014b, GPS (VTide)."""
    print("  Generating M2 multi-source comparison...")

    # Source 1: PREM Green's function prediction (from m2ampphase.csv — radial only)
    df_prem = pd.read_csv(M2_CSV)
    row = df_prem[df_prem["StationName"] == STNM.upper()]
    prem_m2_amp = float(row["PREMM2Amplitude"].values[0])  # mm
    prem_m2_pha = float(row["PREMM2Phase"].values[0])  # degrees

    # Source 2: FES2014b model (from .db file)
    df_fes = read_otl_params(STNM, OLD_OTL_DIR, suffix="_otl")
    fes_m2_dU = float(df_fes.loc["M2", "dU"]) * 1000  # mm
    fes_m2_gU = float(df_fes.loc["M2", "gU"])
    fes_m2_dS = float(df_fes.loc["M2", "dS"]) * 1000
    fes_m2_dW = float(df_fes.loc["M2", "dW"]) * 1000

    # Source 3: GPS (VTide on synthetic data)
    cache = StationCache(RESULTS_DIR / f"{STNM}.h5")
    df_gps = cache.read_otl_coeff()
    gps_m2_dU = float(df_gps.loc["M2", "dU"]) * 1000
    gps_m2_gU = float(df_gps.loc["M2", "gU"])
    gps_m2_sdU = float(df_gps.loc["M2", "sdU"]) * 1000

    # ── Bar chart: M2 radial amplitude ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sources = ["PREM\n(LoadDef)", "FES2014b\n(model .db)", "GPS\n(VTide, synthetic)"]
    amps = [prem_m2_amp, fes_m2_dU, gps_m2_dU]
    errs = [0, 0, gps_m2_sdU]
    colors = ["#9C27B0", "#FF9800", "#2196F3"]

    bars = ax1.bar(sources, amps, color=colors, alpha=0.85, yerr=errs, capsize=5,
                   error_kw={"linewidth": 1.5})
    for bar, amp in zip(bars, amps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{amp:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("M2 Radial Amplitude (mm)")
    ax1.set_title(f"{STNM.upper()} — M2 Up Amplitude: Three Sources")
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.set_ylim(0, max(amps) * 1.3)

    # ── Phasor comparison ──
    ax2 = fig.add_subplot(122, projection="polar")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)

    # PREM
    ax2.annotate("", xy=(np.deg2rad(prem_m2_pha), prem_m2_amp),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#9C27B0", lw=2.5))
    # FES2014b
    ax2.annotate("", xy=(np.deg2rad(fes_m2_gU), fes_m2_dU),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#FF9800", lw=2.5))
    # GPS
    if not np.isnan(gps_m2_dU):
        ax2.annotate("", xy=(np.deg2rad(gps_m2_gU), gps_m2_dU),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color="#2196F3", lw=2.5))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#9C27B0", lw=2.5, label=f"PREM ({prem_m2_amp:.1f} mm, {prem_m2_pha:.1f}°)"),
        Line2D([0], [0], color="#FF9800", lw=2.5, label=f"FES2014b ({fes_m2_dU:.1f} mm, {fes_m2_gU:.1f}°)"),
        Line2D([0], [0], color="#2196F3", lw=2.5, label=f"GPS ({gps_m2_dU:.1f} mm, {gps_m2_gU:.1f}°)"),
    ]
    ax2.legend(handles=legend_elements, loc="lower left", fontsize=8,
               bbox_to_anchor=(-0.1, -0.15))
    ax2.set_title("M2 Up Phasor (mm, °)")
    rmax = max(prem_m2_amp, fes_m2_dU, gps_m2_dU) * 1.2
    ax2.set_rlim(0, rmax)

    fig.tight_layout()
    fig.savefig(OUTDIR / f"{STNM}_07_m2_sources.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {STNM}_07_m2_sources.png")


def plot_fes2014b_all_constituents():
    """Plot 8: Full FES2014b model comparison — old .db vs new .otl (should be identical)."""
    print("  Generating model coefficient comparison...")

    df_old = read_otl_params(STNM, OLD_OTL_DIR, suffix="_otl")
    try:
        df_new = read_otl_params(STNM, NEW_OTL_DIR, suffix="_FES2014b")
    except FileNotFoundError:
        # New pipeline copies from old, so use same file
        df_new = read_otl_params(STNM, NEW_OTL_DIR, suffix="_otl")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    cols = ["dU", "dS", "dW", "gU", "gS", "gW"]
    titles = ["Up Amp (mm)", "South Amp (mm)", "West Amp (mm)",
              "Up Phase (°)", "South Phase (°)", "West Phase (°)"]
    x = np.arange(len(CONSTITUENTS))

    for ax, col, title in zip(axes.flat, cols, titles):
        scale = 1000 if col.startswith("d") else 1
        old_vals = df_old[col].values * scale
        new_vals = df_new[col].values * scale

        w = 0.35
        ax.bar(x - w/2, old_vals, w, label="otl_work (.db)", color="#FF9800", alpha=0.8)
        ax.bar(x + w/2, new_vals, w, label="global_otl (.otl)", color="#2196F3", alpha=0.8)

        max_diff = np.max(np.abs(old_vals - new_vals))
        ax.set_title(f"{title}  (max Δ={max_diff:.4f})")
        ax.set_xticks(x)
        ax.set_xticklabels(CONSTITUENTS, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")
        ax.legend(fontsize=7)

    fig.suptitle(f"{STNM.upper()} — FES2014b Model Coefficients: otl_work vs global_otl", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{STNM}_08_model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {STNM}_08_model_comparison.png")


def print_summary_table():
    """Print a text comparison table."""
    print(f"\n{'='*70}")
    print(f"  ABOA — M2 Comparison Summary")
    print(f"{'='*70}")

    # PREM
    df_prem = pd.read_csv(M2_CSV)
    row = df_prem[df_prem["StationName"] == STNM.upper()]
    prem_amp = float(row["PREMM2Amplitude"].values[0])
    prem_pha = float(row["PREMM2Phase"].values[0])

    # FES2014b
    df_fes = read_otl_params(STNM, OLD_OTL_DIR, suffix="_otl")

    # GPS
    cache = StationCache(RESULTS_DIR / f"{STNM}.h5")
    df_gps = cache.read_otl_coeff()

    print(f"\n  {'Source':<25} {'dU amp (mm)':>12} {'dU pha (°)':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'PREM (LoadDef)':<25} {prem_amp:>12.2f} {prem_pha:>12.1f}")
    print(f"  {'FES2014b (model .db)':<25} {float(df_fes.loc['M2','dU'])*1000:>12.2f} {float(df_fes.loc['M2','gU']):>12.1f}")
    gps_amp = float(df_gps.loc['M2','dU'])*1000
    gps_pha = float(df_gps.loc['M2','gU'])
    gps_sd = float(df_gps.loc['M2','sdU'])*1000
    print(f"  {'GPS VTide (synthetic)':<25} {gps_amp:>12.2f} {gps_pha:>12.1f}  ±{gps_sd:.2f}")
    print(f"  {'Injected (truth)':<25} {'10.00':>12} {'45.5':>12}")

    print(f"\n  Notes:")
    print(f"  - PREM uses PREM Green's functions (12.06 mm) — differs from FES2014b (10.05 mm)")
    print(f"  - GPS VTide recovers the injected synthetic M2 signal, NOT real GPS data")
    print(f"  - Phase from VTide may differ from model due to t=0 reference convention")
    print(f"  - Old pipeline (.mat) used UTide on real GPS data — cannot read without MATLAB")
    print(f"{'='*70}\n")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating comparison plots for {STNM.upper()}...")

    plot_hardisp_comparison()
    plot_m2_multi_source()
    plot_fes2014b_all_constituents()
    print_summary_table()

    print(f"All comparison plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
