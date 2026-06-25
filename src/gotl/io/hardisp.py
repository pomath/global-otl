"""Load pre-computed hardisp tidal displacement files.

Replaces: get_hardisp.m

File format: space/tab-delimited text, 8 columns (no header):
    Year  DOY  Hour  Minute  Second  dU  dS  dW
Also supports legacy 5-column format: Year  DOY  dU  dS  dW
288 rows per DOY (24 hours × 12 five-minute intervals).

Column conventions (Chalmers/CARGA output):
    dU = radial displacement, positive upward (m)
    dS = tangential NS displacement, positive South (m)
    dW = tangential EW displacement, positive West (m)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_YEARS = [str(y) for y in range(2000, 2023)]
_COLS_8 = ["Year", "DOY", "Hour", "Minute", "Second", "dU", "dS", "dW"]
_COLS_5 = ["Year", "DOY", "dU", "dS", "dW"]


def load_hardisp(
    stnm: str,
    hardisp_root: Path,
    years: list[str] | None = None,
) -> pd.DataFrame:
    """Load pre-computed hardisp forward model for one station.

    Args:
        stnm: 4-letter station code
        hardisp_root: parent directory; station files are in
                      {hardisp_root}/{stnm}/{stnm}-hardisp_{YYYY}.txt
        years: list of year strings to load; defaults to 2000-2022

    Returns:
        DataFrame with columns ['t', 'dU', 'dS', 'dW'],
        't' is a datetime column (UTC), dtype datetime64[ns].
    """
    if years is None:
        years = _DEFAULT_YEARS

    hardisp_dir = Path(hardisp_root) / stnm
    frames = []
    for yr in years:
        fpath = hardisp_dir / f"{stnm}-hardisp_{yr}.txt"
        if not fpath.exists():
            continue
        df_yr = _load_one_year(fpath)
        frames.append(df_yr)

    if not frames:
        raise FileNotFoundError(
            f"No hardisp files found for {stnm} in {hardisp_dir} "
            f"(tried years {years[0]}–{years[-1]})"
        )
    return pd.concat(frames, ignore_index=True)


def _load_one_year(fpath: Path) -> pd.DataFrame:
    """Parse one annual hardisp file and return a DataFrame with column 't'.

    Auto-detects 8-column (year doy h m s dU dS dW) or 5-column (year doy dU dS dW) format.
    """
    # Detect column count from first non-comment line
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ncols = len(line.split())
                break
        else:
            raise ValueError(f"{fpath}: empty or all-comment file")

    if ncols == 8:
        cols = _COLS_8
        dtypes = {c: int for c in ("Year", "DOY", "Hour", "Minute", "Second")}
        dtypes.update({"dU": float, "dS": float, "dW": float})
    elif ncols == 5:
        cols = _COLS_5
        dtypes = {"Year": int, "DOY": int, "dU": float, "dS": float, "dW": float}
    else:
        raise ValueError(f"{fpath}: expected 5 or 8 columns, got {ncols}")

    df = pd.read_csv(
        fpath,
        sep=r"\s+",
        header=None,
        names=cols,
        dtype=dtypes,
        comment="#",
    )
    n_rows = len(df)
    if n_rows % 288 != 0:
        raise ValueError(f"{fpath}: expected multiple of 288 rows, got {n_rows}")

    # Build timestamps
    year = int(df["Year"].iloc[0])
    base_dates = pd.to_datetime(year * 1000 + df["DOY"].values.astype(int), format="%Y%j")

    if ncols == 8:
        deltas = pd.to_timedelta(
            df["Hour"].values * 3600 + df["Minute"].values * 60 + df["Second"].values,
            unit="s",
        )
    else:
        # Infer 5-min grid within each day
        n_days = n_rows // 288
        hrs = np.tile(np.repeat(np.arange(24), 12), n_days)
        mins = np.tile(np.tile(np.arange(0, 60, 5), 24), n_days)
        deltas = pd.to_timedelta(hrs * 3600 + mins * 60, unit="s")

    t = base_dates + deltas

    return pd.DataFrame(
        {"t": t, "dU": df["dU"].values, "dS": df["dS"].values, "dW": df["dW"].values}
    )
