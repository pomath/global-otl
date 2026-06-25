#!/usr/bin/env python3
"""Diagnostic plots for verifying the gotl pipeline on a single station.

Usage:
    poetry run python scripts/plot_diagnostics.py --results-dir /tmp/gotl_demo_rjjczazg/results \
        --tdp-root /tmp/gotl_demo_rjjczazg/tdp \
        --hardisp-root /tmp/gotl_demo_rjjczazg/hardisp \
        --otl-params-dir /tmp/gotl_demo_rjjczazg/otl_params \
        --station aboa --model FES2014b --outdir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from gotl.io.tdp import load_tdp
from gotl.io.hardisp import load_hardisp
from gotl.io.cache import StationCache
from gotl.io.otl_params import read_otl_params
from gotl.coords.transforms import ecef_to_enu, ecef_to_lla
from gotl.pipeline import _combine_and_fit
from gotl.registry import StationMeta
from gotl import CONSTITUENTS


def parse_args():
    p = argparse.ArgumentParser(description="Generate diagnostic plots")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--tdp-root", required=True)
    p.add_argument("--hardisp-root", required=True)
    p.add_argument("--otl-params-dir", required=True)
    p.add_argument("--station", default="aboa")
    p.add_argument("--model", default="FES2014b")
    p.add_argument("--outdir", default="plots")
    return p.parse_args()


def plot_enu_timeseries(df_tdp, stnm, outdir):
    """Plot 1: Raw ENU displacements from TDP data."""
    x0, y0, z0 = df_tdp["X"].median(), df_tdp["Y"].median(), df_tdp["Z"].median()
    lat0, lon0, alt0 = ecef_to_lla(x0, y0, z0)
    e, n, u = ecef_to_enu(df_tdp["X"].values, df_tdp["Y"].values, df_tdp["Z"].values,
                          lat0, lon0, alt0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t = df_tdp["t"]

    for ax, data, label, color in zip(
        axes,
        [e * 1000, n * 1000, u * 1000],
        ["East (mm)", "North (mm)", "Up (mm)"],
        ["#2196F3", "#4CAF50", "#F44336"],
    ):
        ax.plot(t, data, ",", color=color, alpha=0.5, rasterized=True)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{stnm.upper()} — Raw ENU Displacements from TDP")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date (2010)")
    fig.tight_layout()
    fig.savefig(outdir / f"{stnm}_01_enu_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {stnm}_01_enu_timeseries.png")


def plot_hardisp(df_hd, stnm, outdir):
    """Plot 2: Hardisp tidal predictions (first 5 days)."""
    # Show first 5 days to see tidal oscillations clearly
    t_end = df_hd["t"].min() + pd.Timedelta(days=5)
    mask = df_hd["t"] <= t_end
    df = df_hd[mask]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    t = df["t"]

    for ax, col, label, color in zip(
        axes,
        ["dU", "dS", "dW"],
        ["dU — Up (mm)", "dS — South+ (mm)", "dW — West+ (mm)"],
        ["#F44336", "#4CAF50", "#2196F3"],
    ):
        ax.plot(t, df[col] * 1000, "-", color=color, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{stnm.upper()} — Hardisp Tidal Predictions (first 5 days)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(outdir / f"{stnm}_02_hardisp.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {stnm}_02_hardisp.png")


def plot_sltm_fit(df_tdp, df_hd, meta, stnm, outdir):
    """Plot 3: SLTM fit overlaid on data, plus residuals."""
    # Merge and transform (same logic as pipeline)
    df = df_tdp.merge(df_hd[["t", "dU", "dS", "dW"]], on="t", how="inner")
    df = df.drop_duplicates(subset="t", keep="first").sort_values("t").reset_index(drop=True)

    x0, y0, z0 = df["X"].median(), df["Y"].median(), df["Z"].median()
    lat0, lon0, alt0 = ecef_to_lla(x0, y0, z0)
    e, n, u = ecef_to_enu(df["X"].values, df["Y"].values, df["Z"].values, lat0, lon0, alt0)

    # Subtract hardisp
    e_mhd = e - (-df["dW"].values)
    n_mhd = n - (-df["dS"].values)
    u_mhd = u - df["dU"].values

    # SLTM fit
    from gotl.sltm.fit import fit_sltm

    def decyear(t):
        t = pd.Timestamp(t)
        y0 = pd.Timestamp(t.year, 1, 1)
        y1 = pd.Timestamp(t.year + 1, 1, 1)
        return t.year + (t - y0).total_seconds() / (y1 - y0).total_seconds()

    t_dy = np.array([decyear(t) for t in df["t"]])
    tref = decyear(df["t"].mean())
    tjmp = [decyear(j) for j in meta.jump_epochs] if meta.jump_epochs else None

    pvs = np.array([e_mhd, n_mhd, u_mhd])
    fit = fit_sltm(t_dy, tref, pvs, sig=0.01, n=1,
                   tjmp=np.array(tjmp) if tjmp else None,
                   fperiods=np.array([1.0, 0.5]))

    t = df["t"]
    labels = ["East (mm)", "North (mm)", "Up (mm)"]
    colors = ["#2196F3", "#4CAF50", "#F44336"]
    data = [e_mhd, n_mhd, u_mhd]

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex="col",
                             gridspec_kw={"width_ratios": [3, 1]})

    for i, (ax_ts, ax_hist) in enumerate(axes):
        d = data[i] * 1000
        f = fit.cpvs[i, :] * 1000
        resid = (data[i] - fit.cpvs[i, :]) * 1000

        ax_ts.plot(t, d, ",", color=colors[i], alpha=0.3, rasterized=True, label="Data")
        ax_ts.plot(t, f, "-", color="black", linewidth=1.2, label="SLTM fit")
        ax_ts.set_ylabel(labels[i])
        ax_ts.grid(True, alpha=0.3)
        if i == 0:
            ax_ts.legend(loc="upper right", fontsize=8)
            ax_ts.set_title(f"{stnm.upper()} — SLTM Fit (hardisp subtracted)")

        # Residual histogram
        ax_hist.hist(resid, bins=60, orientation="horizontal", color=colors[i], alpha=0.6)
        rms = np.sqrt(np.mean(resid**2))
        ax_hist.set_title(f"RMS={rms:.2f} mm", fontsize=9)
        ax_hist.set_xlabel("Count")

    axes[-1][0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1][0].set_xlabel("Date (2010)")
    fig.tight_layout()
    fig.savefig(outdir / f"{stnm}_03_sltm_fit.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {stnm}_03_sltm_fit.png")


def plot_amplitude_comparison(cache, stnm, model, otl_params_dir, outdir):
    """Plot 4: Bar chart comparing GPS vs model amplitudes per constituent."""
    df_obs = cache.read_otl_coeff()

    try:
        df_mod = read_otl_params(stnm, otl_params_dir, suffix=f"_{model}")
    except FileNotFoundError:
        print(f"  Skipping amplitude plot: {model} OTL params not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    components = [("dU", "gU", "sdU", "Up"), ("dS", "gS", "sdS", "South"), ("dW", "gW", "sdW", "West")]

    for ax, (amp_col, pha_col, sd_col, comp_name) in zip(axes, components):
        obs_amp = df_obs[amp_col].values * 1000
        mod_amp = df_mod[amp_col].values * 1000
        obs_sd = df_obs[sd_col].values * 1000 if sd_col in df_obs.columns else np.zeros_like(obs_amp)
        # Clip error bars to keep plot readable
        obs_sd = np.clip(obs_sd, 0, np.nanmax(mod_amp) * 2)

        x = np.arange(len(CONSTITUENTS))
        w = 0.35

        bars_mod = ax.bar(x - w/2, mod_amp, w, label=model, color="#FF9800", alpha=0.8)
        bars_obs = ax.bar(x + w/2, obs_amp, w, label="GPS", color="#2196F3", alpha=0.8,
                         yerr=obs_sd, capsize=2, error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(CONSTITUENTS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Amplitude (mm)")
        ax.set_title(comp_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        # Set y-axis based on model amplitudes, not blown-up uncertainties
        ymax = max(np.nanmax(mod_amp), np.nanmax(obs_amp)) * 1.5
        if ymax > 0:
            ax.set_ylim(-0.5, ymax)

    fig.suptitle(f"{stnm.upper()} — OTL Amplitudes: GPS vs {model}", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / f"{stnm}_04_amplitudes.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {stnm}_04_amplitudes.png")


def plot_phasor_m2(cache, stnm, model, otl_params_dir, outdir):
    """Plot 5: Phasor (vector) diagram for M2 constituent, all 3 components."""
    df_obs = cache.read_otl_coeff()

    try:
        df_mod = read_otl_params(stnm, otl_params_dir, suffix=f"_{model}")
    except FileNotFoundError:
        print(f"  Skipping phasor plot: {model} OTL params not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={"projection": "polar"})
    components = [("dU", "gU", "sdU", "Up"), ("dS", "gS", "sdS", "South"), ("dW", "gW", "sdW", "West")]

    for ax, (amp_col, pha_col, sd_col, comp_name) in zip(axes, components):
        obs_amp = float(df_obs.loc["M2", amp_col]) * 1000
        obs_pha = float(df_obs.loc["M2", pha_col])
        mod_amp = float(df_mod.loc["M2", amp_col]) * 1000
        mod_pha = float(df_mod.loc["M2", pha_col])

        # Model vector
        ax.annotate("", xy=(np.deg2rad(mod_pha), mod_amp), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#FF9800", lw=2.5))
        # GPS vector
        if not np.isnan(obs_amp):
            ax.annotate("", xy=(np.deg2rad(obs_pha), obs_amp), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#2196F3", lw=2.5))

        ax.set_title(f"{comp_name}\n", fontsize=11)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Set radial limit
        rmax = max(obs_amp if not np.isnan(obs_amp) else 0, mod_amp) * 1.3
        if rmax > 0:
            ax.set_rlim(0, rmax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2196F3", lw=2.5, label="GPS"),
        Line2D([0], [0], color="#FF9800", lw=2.5, label=model),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(f"{stnm.upper()} — M2 Phasor Diagram (mm, degrees)", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(outdir / f"{stnm}_05_m2_phasor.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {stnm}_05_m2_phasor.png")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stnm = args.station

    print(f"Generating diagnostic plots for {stnm.upper()}...")

    # Load data
    df_tdp = load_tdp(stnm, Path(args.tdp_root))
    df_hd = load_hardisp(stnm, Path(args.hardisp_root))
    cache = StationCache(Path(args.results_dir) / f"{stnm}.h5")

    meta = StationMeta(
        stnm=stnm, lat=-73.0435, lon=-13.4073, alt_m=468.7,
        network="POLENET", data_start=None, data_end=None, jump_epochs=[],
    )

    # Plot 1: ENU time series
    plot_enu_timeseries(df_tdp, stnm, outdir)

    # Plot 2: Hardisp predictions
    plot_hardisp(df_hd, stnm, outdir)

    # Plot 3: SLTM fit + residuals
    plot_sltm_fit(df_tdp, df_hd, meta, stnm, outdir)

    # Plot 4: Amplitude comparison
    plot_amplitude_comparison(cache, stnm, args.model, Path(args.otl_params_dir), outdir)

    # Plot 5: M2 phasor
    plot_phasor_m2(cache, stnm, args.model, Path(args.otl_params_dir), outdir)

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
