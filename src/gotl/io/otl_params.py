"""Parse OTL loading coefficients from Chalmers .db files.

Replaces: read_otl_params.m

File format (text):
  - 33 header lines beginning with '$$' or station name/metadata
  - 6 data lines (lines 34-39 in 1-indexed MATLAB convention):
      row 0: amplitude dU (radial, m)
      row 1: amplitude dW (EW tangential, m)
      row 2: amplitude dS (NS tangential, m)
      row 3: phase gU (radial, degrees)
      row 4: phase gW (EW tangential, degrees)
      row 5: phase gS (NS tangential, degrees)
  Each data line has 11 space-separated floats: M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gotl import CONSTITUENTS

_COLUMNS = ["dU", "dW", "dS", "gU", "gW", "gS"]


def read_otl_params(stnm: str, params_dir: Path, suffix: str = "_otl") -> pd.DataFrame:
    """Parse a Chalmers .db/.otl file and return a DataFrame of tidal loading coefficients.

    Args:
        stnm: 4-letter station code (e.g., 'aboa')
        params_dir: directory containing OTL coefficient files
        suffix: filename suffix before extension (default '_otl' → {stnm}_otl.db;
                for model-specific files use e.g. '_FES2014b' → {stnm}_FES2014b.otl)

    Returns:
        DataFrame with index=CONSTITUENTS, columns=['dU','dW','dS','gU','gW','gS']
        Amplitudes in metres, phases in degrees.
    """
    # Try .otl first, fall back to .db
    path = Path(params_dir) / f"{stnm}{suffix}.otl"
    if not path.exists():
        path = Path(params_dir) / f"{stnm}{suffix}.db"
    lines = path.read_text().splitlines()
    # Lines 34-39 in 1-indexed MATLAB = Python indices 33:39
    data_lines = lines[33:39]
    rows = [list(map(float, ln.split())) for ln in data_lines]
    # rows shape: [6][11] → transpose to [11][6]
    arr = np.array(rows).T
    return pd.DataFrame(arr, index=CONSTITUENTS, columns=_COLUMNS)
