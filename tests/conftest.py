"""Shared test fixtures for gotl pipeline verification."""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Station under test
STNM = "aboa"
LAT = -73.0435
LON = -13.4073
ALT = 468.7

# ECEF reference position for ABOA (approximate)
# Computed from lat/lon/alt using pymap3d
from pymap3d import geodetic2ecef
X0, Y0, Z0 = geodetic2ecef(LAT, LON, ALT)

# J2000 epoch for TDP
J2000 = pd.Timestamp("2000-01-01 12:00:00")

# M2 tidal parameters for testing (period ~12.42 hours)
M2_PERIOD_HOURS = 12.4206
M2_AMP_U = 0.010   # 10 mm radial
M2_AMP_E = 0.004   # 4 mm east
M2_AMP_N = 0.001   # 1 mm north
M2_PHASE_U = 45.0   # degrees
M2_PHASE_E = -110.0
M2_PHASE_N = 80.0


@pytest.fixture(scope="session")
def tmp_root(tmp_path_factory):
    """Root temp directory for all test data."""
    return tmp_path_factory.mktemp("gotl_test")


@pytest.fixture(scope="session")
def tdp_dir(tmp_root):
    """Generate synthetic TDP files for aboa (1 year, 5-min sampling)."""
    stn_dir = tmp_root / "tdp" / f"{STNM}_results"
    stn_dir.mkdir(parents=True)

    # Generate 30 days of data (enough for SLTM + basic tidal analysis)
    start = pd.Timestamp("2010-01-01")
    t_5min = pd.date_range(start, periods=288 * 30, freq="5min")

    # GPS seconds since J2000
    gps_sec = (t_5min - J2000).total_seconds().values

    # GipsyX applies OTL during PPP, so tidal loading is already removed
    # from TDP positions. Synthetic TDP contains only reference position + noise.
    # The tidal signal is recovered in post-processing by adding hardisp back
    # to the SLTM residuals.
    rng = np.random.default_rng(42)
    noise = 0.002  # 2 mm noise
    X = X0 + rng.normal(0, noise, len(t_5min))
    Y = Y0 + rng.normal(0, noise, len(t_5min))
    Z = Z0 + rng.normal(0, noise, len(t_5min))

    # Write one TDP file per day
    for day in range(30):
        sl = slice(day * 288, (day + 1) * 288)
        day_gps = gps_sec[sl]
        day_X = X[sl]
        day_Y = Y[sl]
        day_Z = Z[sl]

        lines = []
        for i in range(len(day_gps)):
            sec = day_gps[i]
            lines.append(f"{sec:.6f}  {STNM.upper()}.Pos.X  {day_X[i]:.6f}  0.010")
            lines.append(f"{sec:.6f}  {STNM.upper()}.Pos.Y  {day_Y[i]:.6f}  0.010")
            lines.append(f"{sec:.6f}  {STNM.upper()}.Pos.Z  {day_Z[i]:.6f}  0.010")

        # Name the file with a numeric prefix
        fname = f"{int(day_gps[0])}.tdp"
        (stn_dir / fname).write_text("\n".join(lines) + "\n")

    return tmp_root / "tdp"


@pytest.fixture(scope="session")
def otl_params_dir(tmp_root):
    """Copy real aboa OTL params from ~/otl_work/ or create synthetic ones."""
    odir = tmp_root / "otl_params"
    odir.mkdir(parents=True)

    real_db = Path.home() / "otl_work/otl_proc/otl_params/aboa_otl.db"
    if real_db.exists():
        # Copy as aboa_FES2014b.otl (our test model)
        import shutil
        shutil.copy(real_db, odir / f"{STNM}_FES2014b.otl")
        shutil.copy(real_db, odir / f"{STNM}_otl.db")
    else:
        # Create synthetic OTL params
        header = _make_otl_header(STNM, LON, LAT, ALT)
        (odir / f"{STNM}_FES2014b.otl").write_text(header)
        (odir / f"{STNM}_otl.db").write_text(header)

    return odir


def _make_otl_header(stnm, lon, lat, alt):
    """Create a minimal Harpos .db file with synthetic OTL coefficients."""
    # 33 header lines + 6 data lines
    header_lines = []
    header_lines.append("$$ Ocean loading displacement")
    header_lines.append("$$")
    for i in range(29):
        header_lines.append(f"$$ header line {i+3}")
    header_lines.append(f"  {stnm}")
    header_lines.append(f"$$ Synthetic test data")
    # Line 33 (0-indexed 32): station info
    header_lines.append(f"$$ {stnm}  RADI TANG  lon/lat: {lon:.4f}  {lat:.4f}  {alt:.3f}")

    # 6 data lines (lines 34-39 in 1-indexed):
    # M2    S2     N2     K2     K1     O1     P1     Q1     MF     MM     SSA
    amp_u = "  .01005 .00712 .00150 .00206 .00746 .00945 .00262 .00238 .00160 .00091 .00084"
    amp_w = "  .00439 .00235 .00078 .00068 .00141 .00186 .00048 .00045 .00016 .00009 .00007"
    amp_s = "  .00114 .00102 .00019 .00031 .00146 .00163 .00050 .00041 .00014 .00010 .00010"
    pha_u = "    45.5   60.7   24.6   61.7 -168.2 -175.9 -168.4  173.3   16.8    9.4    1.1"
    pha_w = "  -111.0  -92.4 -125.9  -93.8   57.1   46.8   56.1   34.9 -170.2 -176.5  179.8"
    pha_s = "    79.4   64.1  100.4   64.3 -177.5  173.5 -178.1  164.1   35.1   21.6    1.6"

    data_lines = [amp_u, amp_w, amp_s, pha_u, pha_w, pha_s]

    lines = header_lines + data_lines + ["$$"]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def hardisp_dir(tmp_root, otl_params_dir):
    """Generate hardisp files using the actual binary."""
    hdir = tmp_root / "hardisp"
    hdir.mkdir(parents=True)

    hardisp_exe = Path(__file__).resolve().parent.parent / "hardisp" / "hardisp_exe"
    otl_file = otl_params_dir / f"{STNM}_FES2014b.otl"

    if hardisp_exe.exists() and otl_file.exists():
        from gotl.hardisp_gen.batch import generate_hardisp_year
        stn_out = hdir / STNM
        stn_out.mkdir(parents=True, exist_ok=True)
        generate_hardisp_year(hardisp_exe, otl_file, STNM, 2010, stn_out)

    return hdir


@pytest.fixture(scope="session")
def results_dir(tmp_root):
    """Directory for HDF5 cache files."""
    rdir = tmp_root / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    return rdir
