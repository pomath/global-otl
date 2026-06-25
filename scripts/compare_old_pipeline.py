#!/usr/bin/env python3
"""Compare old MATLAB pipeline (UTide) results with OTL model predictions.

Loads:
  - data/reference/otl_coeff_all.csv (old pipeline GPS estimates from MATLAB/UTide)
  - ~/otl_work/otl_proc/otl_params/{stnm}_otl.db (FES2014b model predictions)

Outputs comparison plots to plots/old_pipeline/.

Usage:
    poetry run python scripts/compare_old_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gotl.io.otl_params import read_otl_params
from gotl import CONSTITUENTS


OTL_PARAMS_DIR = Path.home() / "otl_work" / "otl_proc" / "otl_params"
OLD_CSV = Path("data/reference/otl_coeff_all.csv")
OUTDIR = Path("plots/old_pipeline")


def load_old_pipeline(csv_path: Path) -> pd.DataFrame:
    """Load old pipeline CSV. Amplitudes in metres, phases in degrees."""
    return pd.read_csv(csv_path)


def load_model_for_station(stnm: str) -> pd.DataFrame | None:
    """Load FES2014b model OTL params for a station."""
    try:
        return read_otl_params(stnm, OTL_PARAMS_DIR, suffix="_otl")
    except FileNotFoundError:
        return None


def vector_misfit(amp_obs, pha_obs, amp_mod, pha_mod):
    """Complex vector misfit between observed and model."""
    z_obs = amp_obs * np.exp(1j * np.deg2rad(pha_obs))
    z_mod = amp_mod * np.exp(1j * np.deg2rad(pha_mod))
    return np.abs(z_obs - z_mod)


def plot_station_comparison(stnm, df_old_stn, df_mod, outdir):
    """Single-station amplitude comparison: old GPS pipeline vs model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    components = [("dU", "gU", "Up"), ("dS", "gS", "South"), ("dW", "gW", "West")]

    for ax, (amp_col, pha_col, comp_name) in zip(axes, components):
        # Old pipeline values (from CSV, in metres → mm)
        obs_amps = []
        mod_amps = []
        for const in CONSTITUENTS:
            row = df_old_stn[df_old_stn["constituent"] == const]
            if not row.empty:
                obs_amps.append(float(row[amp_col].iloc[0]) * 1000)
            else:
                obs_amps.append(np.nan)
            mod_amps.append(float(df_mod.loc[const, amp_col]) * 1000)

        obs_amps = np.array(obs_amps)
        mod_amps = np.array(mod_amps)

        x = np.arange(len(CONSTITUENTS))
        w = 0.35

        ax.bar(x - w/2, mod_amps, w, label="FES2014b", color="#FF9800", alpha=0.8)
        ax.bar(x + w/2, obs_amps, w, label="GPS (old)", color="#9C27B0", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(CONSTITUENTS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Amplitude (mm)")
        ax.set_title(comp_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"{stnm.upper()} — Old Pipeline (UTide) vs FES2014b Model", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / f"{stnm}_amplitudes.png", dpi=150)
    plt.close(fig)


def plot_m2_all_stations(df_old, outdir):
    """M2 amplitude comparison across all stations: GPS vs model."""
    stations = sorted(df_old["stnm"].unique())
    components = [("dU", "gU", "Up (mm)"), ("dS", "gS", "South (mm)"), ("dW", "gW", "West (mm)")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (amp_col, pha_col, label) in zip(axes, components):
        obs_vals = []
        mod_vals = []
        labels = []
        misfits = []

        for stnm in stations:
            df_mod = load_model_for_station(stnm)
            if df_mod is None:
                continue

            row = df_old[(df_old["stnm"] == stnm) & (df_old["constituent"] == "M2")]
            if row.empty:
                continue

            obs_a = float(row[amp_col].iloc[0]) * 1000
            obs_p = float(row[pha_col].iloc[0])
            mod_a = float(df_mod.loc["M2", amp_col]) * 1000
            mod_p = float(df_mod.loc["M2", pha_col])

            obs_vals.append(obs_a)
            mod_vals.append(mod_a)
            labels.append(stnm)
            misfits.append(vector_misfit(obs_a, obs_p, mod_a, mod_p))

        obs_vals = np.array(obs_vals)
        mod_vals = np.array(mod_vals)

        # 1:1 line
        vmax = max(np.nanmax(obs_vals), np.nanmax(mod_vals)) * 1.1
        ax.plot([0, vmax], [0, vmax], "k--", alpha=0.3, linewidth=0.8)

        ax.scatter(mod_vals, obs_vals, c="#9C27B0", s=30, alpha=0.7, edgecolors="white", linewidth=0.5)

        # Label outliers (misfit > 2 mm)
        for i, stnm in enumerate(labels):
            if misfits[i] > 2.0:
                ax.annotate(stnm, (mod_vals[i], obs_vals[i]),
                           fontsize=6, alpha=0.7, xytext=(3, 3),
                           textcoords="offset points")

        ax.set_xlabel(f"Model {label}")
        ax.set_ylabel(f"GPS {label}")
        ax.set_title(f"M2 {label.split()[0]}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("M2 Amplitude: Old Pipeline GPS vs FES2014b (all stations)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "all_stations_m2_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved all_stations_m2_scatter.png")


def plot_m2_phasors_all(df_old, outdir):
    """M2 phasor diagram for all stations (radial component only)."""
    stations = sorted(df_old["stnm"].unique())

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    for stnm in stations:
        df_mod = load_model_for_station(stnm)
        if df_mod is None:
            continue

        row = df_old[(df_old["stnm"] == stnm) & (df_old["constituent"] == "M2")]
        if row.empty:
            continue

        obs_a = float(row["dU"].iloc[0]) * 1000
        obs_p = float(row["gU"].iloc[0])
        mod_a = float(df_mod.loc["M2", "dU"]) * 1000
        mod_p = float(df_mod.loc["M2", "gU"])

        # Draw arrow from model to GPS (residual vector)
        ax.plot([np.deg2rad(mod_p), np.deg2rad(obs_p)],
                [mod_a, obs_a],
                "-", color="#9C27B0", alpha=0.3, linewidth=0.8)
        ax.plot(np.deg2rad(mod_p), mod_a, "o", color="#FF9800", markersize=4, alpha=0.6)
        ax.plot(np.deg2rad(obs_p), obs_a, "o", color="#9C27B0", markersize=4, alpha=0.6)

    ax.set_title("M2 Up — All Stations\n(orange=model, purple=GPS)", pad=20)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    fig.tight_layout()
    fig.savefig(outdir / "all_stations_m2_phasors.png", dpi=150)
    plt.close(fig)
    print(f"  Saved all_stations_m2_phasors.png")


def plot_residual_summary(df_old, outdir):
    """RMS misfit per constituent across all stations."""
    stations = sorted(df_old["stnm"].unique())
    components = [("dU", "gU", "Up"), ("dS", "gS", "South"), ("dW", "gW", "West")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (amp_col, pha_col, comp_name) in zip(axes, components):
        rms_per_const = []
        for const in CONSTITUENTS:
            misfits = []
            for stnm in stations:
                df_mod = load_model_for_station(stnm)
                if df_mod is None:
                    continue
                row = df_old[(df_old["stnm"] == stnm) & (df_old["constituent"] == const)]
                if row.empty:
                    continue

                obs_a = float(row[amp_col].iloc[0]) * 1000
                obs_p = float(row[pha_col].iloc[0])
                mod_a = float(df_mod.loc[const, amp_col]) * 1000
                mod_p = float(df_mod.loc[const, pha_col])

                mf = vector_misfit(obs_a, obs_p, mod_a, mod_p)
                misfits.append(mf)

            if misfits:
                rms_per_const.append(np.sqrt(np.mean(np.array(misfits)**2)))
            else:
                rms_per_const.append(np.nan)

        x = np.arange(len(CONSTITUENTS))
        colors = ["#F44336" if v > 2 else "#4CAF50" for v in rms_per_const]
        ax.bar(x, rms_per_const, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(CONSTITUENTS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMS Misfit (mm)")
        ax.set_title(comp_name)
        ax.grid(True, alpha=0.2, axis="y")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.suptitle("RMS Vector Misfit: Old Pipeline vs FES2014b (all stations)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "rms_misfit_by_constituent.png", dpi=150)
    plt.close(fig)
    print(f"  Saved rms_misfit_by_constituent.png")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading old pipeline results...")
    df_old = load_old_pipeline(OLD_CSV)
    stations = sorted(df_old["stnm"].unique())
    print(f"  {len(stations)} stations, {len(df_old)} rows")

    # Exclude stations with obviously bad solutions (SSA > 100 mm means garbage)
    bad_stations = []
    for stnm in stations:
        ssa_row = df_old[(df_old["stnm"] == stnm) & (df_old["constituent"] == "SSA")]
        if not ssa_row.empty:
            ssa_dU = float(ssa_row["dU"].iloc[0]) * 1000
            if ssa_dU > 50:
                bad_stations.append(stnm)
                print(f"  WARNING: {stnm} SSA dU = {ssa_dU:.1f} mm — bad solution, excluding")
    if bad_stations:
        df_old = df_old[~df_old["stnm"].isin(bad_stations)]
        stations = sorted(df_old["stnm"].unique())

    # Count how many have model files
    n_model = sum(1 for s in stations if load_model_for_station(s) is not None)
    print(f"  {n_model}/{len(stations)} have matching FES2014b model files")

    # Per-station plots (just aboa and a few others for now)
    for stnm in ["aboa", "robn", "capf", "thur", "vnad"]:
        if stnm not in stations:
            continue
        df_mod = load_model_for_station(stnm)
        if df_mod is None:
            print(f"  Skipping {stnm}: no model file")
            continue
        plot_station_comparison(stnm, df_old[df_old["stnm"] == stnm], df_mod, OUTDIR)
        print(f"  Saved {stnm}_amplitudes.png")

    # Multi-station summary plots
    plot_m2_all_stations(df_old, OUTDIR)
    plot_m2_phasors_all(df_old, OUTDIR)
    plot_residual_summary(df_old, OUTDIR)

    # Print aboa M2 comparison table
    print("\n--- ABOA M2 Comparison ---")
    df_mod = load_model_for_station("aboa")
    row = df_old[(df_old["stnm"] == "aboa") & (df_old["constituent"] == "M2")]
    if not row.empty and df_mod is not None:
        for amp_col, pha_col, comp in [("dU","gU","Up"), ("dS","gS","South"), ("dW","gW","West")]:
            obs_a = float(row[amp_col].iloc[0]) * 1000
            obs_p = float(row[pha_col].iloc[0])
            mod_a = float(df_mod.loc["M2", amp_col]) * 1000
            mod_p = float(df_mod.loc["M2", pha_col])
            mf = vector_misfit(obs_a, obs_p, mod_a, mod_p)
            print(f"  {comp:6s}: GPS={obs_a:6.2f} mm / {obs_p:6.1f}°  "
                  f"Model={mod_a:6.2f} mm / {mod_p:6.1f}°  "
                  f"Misfit={mf:.2f} mm")

    print(f"\nAll plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
