"""Load GPS TDP (Tropospheric Displacement Product) position files.

Replaces: load_bystn.m

File format: GipsyX TDP text output; each line has whitespace-delimited fields.
Relevant lines contain the station name (case-insensitive) and "Pos.X", "Pos.Y", "Pos.Z".
Column layout (0-indexed): 0=GPS_seconds, 1=something, 2=value, ...

Time epoch: GPS seconds since J2000 = 2000-01-01 12:00:00 UTC.

Files are matched by pattern: starts with digits, ends with '.tdp' (e.g., '2456789.tdp').
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

_J2000_EPOCH = pd.Timestamp("2000-01-01 12:00:00")
_TDP_PATTERN = re.compile(r"^\d+\.tdp$")
_COORDS = ("X", "Y", "Z")


def load_tdp(stnm: str, tdp_root: Path) -> pd.DataFrame:
    """Load all TDP files for a station into a single DataFrame.

    Args:
        stnm: 4-letter station code (case-insensitive matching against file content)
        tdp_root: root directory; TDP files live in {tdp_root}/{stnm}_results/*.tdp

    Returns:
        DataFrame with columns ['t', 'X', 'Y', 'Z'], sorted by 't',
        't' is a datetime column (UTC), dtype datetime64[ns].
    """
    stn_dir = Path(tdp_root) / f"{stnm}_results"
    tdp_files = sorted(
        f for f in stn_dir.iterdir() if _TDP_PATTERN.match(f.name)
    )
    if not tdp_files:
        raise FileNotFoundError(f"No TDP files found in {stn_dir}")

    frames = []
    for fpath in tdp_files:
        df = _load_one_tdp(fpath, stnm)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        raise ValueError(f"No usable data found for station '{stnm}' in {stn_dir}")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("t").reset_index(drop=True)
    return result


def _load_one_tdp(fpath: Path, stnm: str) -> pd.DataFrame | None:
    """Parse one TDP file and return one day of X, Y, Z data."""
    text = fpath.read_text(errors="replace")
    lines = text.splitlines()

    # Filter lines that belong to this station (case-insensitive)
    stnm_lower = stnm.lower()
    stn_lines = [ln for ln in lines if stnm_lower in ln.lower()]
    if not stn_lines:
        return None

    # Extract X, Y, Z by further filtering on "Pos.X" etc.
    coords: dict[str, np.ndarray] = {}
    gps_seconds: np.ndarray | None = None

    for coord in _COORDS:
        key = f"Pos.{coord}"
        matched = [ln for ln in stn_lines if key in ln]
        if not matched:
            return None
        tokens = [ln.split() for ln in matched]
        try:
            values = np.array([float(t[2]) for t in tokens])
            times = np.array([float(t[0]) for t in tokens])
        except (IndexError, ValueError):
            return None
        coords[coord] = values
        if gps_seconds is None:
            gps_seconds = times

    if gps_seconds is None or len(gps_seconds) == 0:
        return None

    # Convert GPS seconds (J2000 epoch) to datetime
    t = _J2000_EPOCH + pd.to_timedelta(gps_seconds, unit="s")

    df = pd.DataFrame({"t": t, "X": coords["X"], "Y": coords["Y"], "Z": coords["Z"]})

    # Keep only one day: filter to midnight–23:59:59 of the median day
    center = df["t"].mean()
    day_start = center.normalize()
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df["t"] >= day_start) & (df["t"] <= day_end)].copy()

    return df if len(df) > 0 else None
