"""Quality diagnostic plots for all year-long BRUX GipsyX solutions."""

import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

CONSTITUENTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"]

RUNS = {
    "cloud_tdp":     "/path/to/results/brux_cloud.h5",
    "cloud_tdp2":    "/path/to/results/brux_cloud2.h5",
    "tdp":           "/path/to/results/brux.h5",
    "rate30_year":   "/path/to/work/rate30_year/results/brux.h5",
    "rate30_year_v2": "/path/to/work/rate30_year_v2/results/brux.h5",
}

# FES2014 model values from brux_FES2014.otl (mm)
FES2014 = {
    "dU": [6.89, 2.41, 1.43, 0.68, 2.31, 1.19, 0.67, 0.13, 0.66, 0.35, 0.36],
    "dW": [2.50, 0.74, 0.54, 0.21, 0.39, 0.31, 0.11, 0.07, 0.06, 0.04, 0.03],
    "dS": [0.37, 0.23, 0.13, 0.07, 0.28, 0.27, 0.10, 0.08, 0.02, 0.00, 0.02],
    "gU": [278.2, 313.3, 263.4, 310.2, 296.7, 282.7, 294.8, 268.0, 11.9, 6.3, 1.5],
    "gW": [257.4, 295.4, 239.7, 293.6, 266.8, 213.4, 257.9, 173.6, 1.2, 4.6, 357.7],
    "gS": [200.9, 217.7, 217.9, 210.8, 252.5, 155.1, 253.6, 82.1, 208.5, 97.2, 184.4],
}

RUN_COLORS = {
    "cloud_tdp":     "#1f77b4",
    "cloud_tdp2":    "#ff7f0e",
    "tdp":           "#2ca02c",
    "rate30_year":   "#d62728",
    "rate30_year_v2": "#9467bd",
}

OUT = "brux_quality.png"


def load_run(path):
    """Load summary data from one HDF5 cache."""
    with h5py.File(path, "r") as f:
        tdp_t = f["tdp/t"][:]
        comb_t = f["combined/t"][:]
        dt = np.median(np.diff(tdp_t))
        info = {
            "n_tdp": len(tdp_t),
            "n_comb": len(comb_t),
            "dt_s": dt,
            "start": pd.Timestamp(tdp_t[0], unit="s"),
            "end": pd.Timestamp(tdp_t[-1], unit="s"),
        }
        # SLTM residuals
        combined = pd.DataFrame({
            "t": pd.to_datetime(comb_t, unit="s"),
            "u_nop": f["combined/u_nop"][:],
            "s_nop": f["combined/s_nop"][:],
            "w_nop": f["combined/w_nop"][:],
        })
        # OTL coefficients
        otl = pd.DataFrame({
            col: f[f"otl_coeff/{col}"][:] for col in
            ["dU", "dS", "dW", "gU", "gS", "gW", "sdU", "sdS", "sdW", "sgU", "sgS", "sgW"]
        }, index=CONSTITUENTS)
        # Convert amplitudes/uncertainties to mm
        for c in ["dU", "dS", "dW", "sdU", "sdS", "sdW"]:
            otl[c] *= 1e3
    return info, combined, otl


def phase_diff(gps, model):
    """Signed phase difference wrapped to [-180, 180]."""
    d = gps - model
    return (d + 180) % 360 - 180


def vector_misfit(amp_gps, pha_gps, amp_mod, pha_mod):
    """Vector difference magnitude between GPS and model phasors."""
    g, m = np.deg2rad(pha_gps), np.deg2rad(pha_mod)
    dx = amp_gps * np.cos(g) - amp_mod * np.cos(m)
    dy = amp_gps * np.sin(g) - amp_mod * np.sin(m)
    return np.sqrt(dx**2 + dy**2)


def main():
    # Load all runs
    data = {}
    for name, path in RUNS.items():
        data[name] = load_run(path)

    x = np.arange(len(CONSTITUENTS))
    fes = FES2014

    fig = plt.figure(figsize=(20, 26), facecolor="white")
    fig.suptitle(
        "BRUX — Year-Long GipsyX Solution Quality: 5 Runs Compared\n"
        "(2024-01-03 to 2024-12-30, FES2014 reference model)",
        fontsize=15, fontweight="bold", y=0.99,
    )
    gs = fig.add_gridspec(6, 2, hspace=0.38, wspace=0.28,
                          left=0.07, right=0.96, top=0.96, bottom=0.03)

    # ── Panel 1: SLTM residual time series for best 5-min run ──
    ax_ts = fig.add_subplot(gs[0, :])
    best_name = "cloud_tdp2"
    _, comb_best, _ = data[best_name]
    for comp, label, color in [
        ("u_nop", "Up", "#1f77b4"),
        ("w_nop", "East", "#2ca02c"),
        ("s_nop", "North", "#d62728"),
    ]:
        vals = comb_best[comp].values * 1e3
        ax_ts.plot(comb_best["t"], vals, lw=0.15, alpha=0.5, color=color, label=label)
    ax_ts.set_ylabel("Residual (mm)")
    ax_ts.set_title(f"SLTM Residuals — {best_name} (5-min, 104k epochs)", fontsize=11)
    ax_ts.legend(loc="upper right", fontsize=9)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    for comp, yoff, color in [("u_nop", 0.95, "#1f77b4"),
                               ("w_nop", 0.88, "#2ca02c"),
                               ("s_nop", 0.81, "#d62728")]:
        rms = np.sqrt(np.mean(comb_best[comp].values**2)) * 1e3
        lbl = {"u_nop": "U", "w_nop": "E", "s_nop": "N"}[comp]
        ax_ts.text(0.01, yoff, f"RMS {lbl}: {rms:.1f} mm",
                   transform=ax_ts.transAxes, fontsize=9, color=color,
                   fontweight="bold", family="monospace")
    ax_ts.grid(True, alpha=0.3)

    # ── Panel 2: RMS comparison bar chart ──
    ax_rms = fig.add_subplot(gs[1, 0])
    bar_w = 0.15
    for i, (name, (info, comb, _)) in enumerate(data.items()):
        rms_u = np.sqrt(np.mean(comb["u_nop"].values**2)) * 1e3
        rms_e = np.sqrt(np.mean(comb["w_nop"].values**2)) * 1e3
        rms_n = np.sqrt(np.mean(comb["s_nop"].values**2)) * 1e3
        dt_min = info["dt_s"] / 60
        lbl = f"{name}\n({dt_min:.0f}min, {info['n_comb']/1000:.0f}k)"
        positions = np.array([0, 1, 2]) + i * bar_w
        ax_rms.bar(positions, [rms_u, rms_e, rms_n], bar_w * 0.9,
                   color=RUN_COLORS[name], edgecolor="k", lw=0.5, label=lbl)
    ax_rms.set_xticks([0 + 2*bar_w, 1 + 2*bar_w, 2 + 2*bar_w])
    ax_rms.set_xticklabels(["Up", "East", "North"])
    ax_rms.set_ylabel("RMS (mm)")
    ax_rms.set_title("SLTM Residual RMS by Run", fontsize=11)
    ax_rms.legend(fontsize=6, loc="upper right")
    ax_rms.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Daily epoch count comparison ──
    ax_daily = fig.add_subplot(gs[1, 1])
    for name, (info, comb, _) in data.items():
        c2 = comb[["t"]].copy()
        c2["date"] = c2["t"].dt.date
        daily = c2.groupby("date").size()
        ax_daily.plot(pd.to_datetime(daily.index), daily.values,
                      lw=0.8, alpha=0.8, color=RUN_COLORS[name],
                      label=f"{name} ({info['dt_s']/60:.0f}min)")
    ax_daily.set_ylabel("Epochs / Day")
    ax_daily.set_title("Daily Epoch Count", fontsize=11)
    ax_daily.legend(fontsize=7)
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_daily.xaxis.set_major_locator(mdates.MonthLocator())
    ax_daily.grid(True, alpha=0.3)

    # ── Panel 4: Up amplitude — GPS vs FES2014 ──
    ax_amp_u = fig.add_subplot(gs[2, 0])
    bar_w = 0.13
    # FES2014 reference bars
    ax_amp_u.bar(x - 0.35, fes["dU"], 0.12, color="gray", alpha=0.5,
                 edgecolor="k", lw=0.5, label="FES2014", zorder=2)
    for i, (name, (_, _, otl)) in enumerate(data.items()):
        offset = -0.2 + i * bar_w
        ax_amp_u.bar(x + offset, otl["dU"].values, bar_w * 0.9,
                     color=RUN_COLORS[name], edgecolor="k", lw=0.3,
                     label=name, zorder=3)
    ax_amp_u.set_xticks(x)
    ax_amp_u.set_xticklabels(CONSTITUENTS, fontsize=8)
    ax_amp_u.set_ylabel("Amplitude (mm)")
    ax_amp_u.set_title("Up Amplitude: GPS runs vs FES2014", fontsize=11)
    ax_amp_u.legend(fontsize=6, ncol=3)
    ax_amp_u.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: Up phase difference (GPS - FES2014) ──
    ax_pha_u = fig.add_subplot(gs[2, 1])
    markers = ["o", "s", "^", "D", "v"]
    for i, (name, (_, _, otl)) in enumerate(data.items()):
        diff = phase_diff(otl["gU"].values, np.array(fes["gU"]))
        ax_pha_u.scatter(x, diff, s=35, c=RUN_COLORS[name], marker=markers[i],
                        label=name, edgecolors="k", lw=0.4, zorder=3)
    ax_pha_u.axhline(0, color="k", lw=0.8)
    ax_pha_u.axhspan(-10, 10, alpha=0.1, color="green", label="±10°")
    ax_pha_u.set_xticks(x)
    ax_pha_u.set_xticklabels(CONSTITUENTS, fontsize=8)
    ax_pha_u.set_ylabel("GPS - FES2014 (deg)")
    ax_pha_u.set_title("Up Phase Difference (wrapped ±180°)", fontsize=11)
    ax_pha_u.legend(fontsize=6, ncol=3)
    ax_pha_u.set_ylim(-180, 180)
    ax_pha_u.grid(True, alpha=0.3)

    # ── Panel 6: Vector misfit (Up) ──
    ax_vm = fig.add_subplot(gs[3, 0])
    bar_w = 0.14
    for i, (name, (_, _, otl)) in enumerate(data.items()):
        vm = vector_misfit(otl["dU"].values, otl["gU"].values,
                          np.array(fes["dU"]), np.array(fes["gU"]))
        offset = -0.28 + i * bar_w
        ax_vm.bar(x + offset, vm, bar_w * 0.9, color=RUN_COLORS[name],
                  edgecolor="k", lw=0.3, label=name)
    ax_vm.set_xticks(x)
    ax_vm.set_xticklabels(CONSTITUENTS, fontsize=8)
    ax_vm.set_ylabel("Vector Misfit (mm)")
    ax_vm.set_title("Up Vector Misfit: |GPS - FES2014|", fontsize=11)
    ax_vm.legend(fontsize=6, ncol=2)
    ax_vm.grid(True, alpha=0.3, axis="y")

    # ── Panel 7: Horizontal amplitudes (E, N) — best run only ──
    ax_hz = fig.add_subplot(gs[3, 1])
    best_otl = data[best_name][2]
    bar_w = 0.3
    ax_hz.bar(x - bar_w/2, best_otl["dW"].values, bar_w, color="#2ca02c",
              edgecolor="k", lw=0.5, label="GPS West", alpha=0.8)
    ax_hz.bar(x + bar_w/2, best_otl["dS"].values, bar_w, color="#d62728",
              edgecolor="k", lw=0.5, label="GPS North", alpha=0.8)
    ax_hz.bar(x - bar_w/2, np.array(fes["dW"]), bar_w, fill=False,
              edgecolor="#2ca02c", lw=1.5, linestyle="--", label="FES2014 West")
    ax_hz.bar(x + bar_w/2, np.array(fes["dS"]), bar_w, fill=False,
              edgecolor="#d62728", lw=1.5, linestyle="--", label="FES2014 North")
    ax_hz.set_xticks(x)
    ax_hz.set_xticklabels(CONSTITUENTS, fontsize=8)
    ax_hz.set_ylabel("Amplitude (mm)")
    ax_hz.set_title(f"Horizontal Amplitudes — {best_name}", fontsize=11)
    ax_hz.legend(fontsize=7, ncol=2)
    ax_hz.grid(True, alpha=0.3, axis="y")

    # ── Panel 8: Amplitude uncertainty comparison ──
    ax_unc = fig.add_subplot(gs[4, 0])
    bar_w = 0.14
    for i, (name, (_, _, otl)) in enumerate(data.items()):
        offset = -0.28 + i * bar_w
        ax_unc.bar(x + offset, otl["sdU"].values, bar_w * 0.9,
                   color=RUN_COLORS[name], edgecolor="k", lw=0.3, label=name)
    ax_unc.set_xticks(x)
    ax_unc.set_xticklabels(CONSTITUENTS, fontsize=8)
    ax_unc.set_ylabel("Uncertainty (mm)")
    ax_unc.set_title("Up Amplitude Uncertainty (1σ) per Run", fontsize=11)
    ax_unc.legend(fontsize=6, ncol=2)
    ax_unc.grid(True, alpha=0.3, axis="y")

    # ── Panel 9: SNR comparison ──
    ax_snr = fig.add_subplot(gs[4, 1])
    for i, (name, (_, _, otl)) in enumerate(data.items()):
        snr = otl["dU"].values / otl["sdU"].values
        offset = -0.28 + i * bar_w
        ax_snr.bar(x + offset, snr, bar_w * 0.9, color=RUN_COLORS[name],
                   edgecolor="k", lw=0.3, label=name)
    ax_snr.axhline(1, color="r", ls="--", lw=1, label="SNR = 1")
    ax_snr.set_xticks(x)
    ax_snr.set_xticklabels(CONSTITUENTS, fontsize=8)
    ax_snr.set_ylabel("SNR")
    ax_snr.set_title("Up Signal-to-Noise Ratio", fontsize=11)
    ax_snr.legend(fontsize=6, ncol=3)
    ax_snr.grid(True, alpha=0.3, axis="y")

    # ── Panel 10: Summary table ──
    ax_tab = fig.add_subplot(gs[5, :])
    ax_tab.axis("off")

    # Build table: one row per run, columns = run info + M2/S2/K1/O1 misfit
    key_consts = ["M2", "S2", "N2", "K2", "K1", "O1"]
    col_labels = (["Run", "dt", "Epochs", "RMS U\n(mm)", "RMS E\n(mm)", "RMS N\n(mm)"]
                  + [f"{c}\nmisfit" for c in key_consts]
                  + ["M2\nΔφ (°)"])
    rows = []
    for name, (info, comb, otl) in data.items():
        rms_u = np.sqrt(np.mean(comb["u_nop"].values**2)) * 1e3
        rms_e = np.sqrt(np.mean(comb["w_nop"].values**2)) * 1e3
        rms_n = np.sqrt(np.mean(comb["s_nop"].values**2)) * 1e3
        row = [
            name,
            f"{info['dt_s']/60:.0f} min",
            f"{info['n_comb']:,}",
            f"{rms_u:.1f}",
            f"{rms_e:.1f}",
            f"{rms_n:.1f}",
        ]
        for c in key_consts:
            idx = CONSTITUENTS.index(c)
            vm = vector_misfit(
                np.array([otl["dU"].iloc[idx]]), np.array([otl["gU"].iloc[idx]]),
                np.array([fes["dU"][idx]]), np.array([fes["gU"][idx]])
            )[0]
            row.append(f"{vm:.2f}")
        m2_pdiff = phase_diff(otl["gU"].iloc[0], fes["gU"][0])
        row.append(f"{m2_pdiff:.1f}")
        rows.append(row)
    # Add FES2014 row
    rows.append(["FES2014", "-", "-", "-", "-", "-"] + ["0.00"] * len(key_consts) + ["0.0"])

    tab = ax_tab.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.4)
    # Header styling
    for j in range(len(col_labels)):
        tab[0, j].set_facecolor("#d4e6f1")
        tab[0, j].set_text_props(fontweight="bold")
    # FES2014 row
    for j in range(len(col_labels)):
        tab[len(rows), j].set_facecolor("#e8f8e8")
    # Color the run name cells
    for i, name in enumerate(data.keys()):
        tab[i + 1, 0].set_facecolor(RUN_COLORS[name] + "30")
    ax_tab.set_title("Summary: Up Component Diagnostics (vector misfit in mm)", fontsize=11)

    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
