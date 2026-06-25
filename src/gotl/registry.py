"""Station registry: load and query the global IGS station metadata CSV.

The registry CSV (data/station_registry.csv) is committed to the repo and
contains one row per station. Fields:

    stnm        4-character station code (lowercase)
    lat         Latitude, degrees N
    lon         Longitude, degrees E
    alt_m       Ellipsoidal height, metres
    network     Network code (e.g. IGS, POLENET)
    data_start  First date with TDP data (YYYY-MM-DD)
    data_end    Last date with TDP data (YYYY-MM-DD)
    jump_epochs Semicolon-separated list of jump dates (YYYY-MM-DD); may be empty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class StationMeta:
    stnm: str
    lat: float
    lon: float
    alt_m: float
    network: str
    data_start: pd.Timestamp | None
    data_end: pd.Timestamp | None
    jump_epochs: list[pd.Timestamp] = field(default_factory=list)


def _parse_ts(val) -> pd.Timestamp | None:
    """Parse a timestamp string, returning None for empty/NaN values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return pd.Timestamp(s) if s else None


def load_registry(registry_path: Path | str) -> dict[str, StationMeta]:
    """Load station registry CSV into a dict keyed by station code.

    Args:
        registry_path: Path to station_registry.csv.

    Returns:
        Dict mapping lowercase station code → StationMeta.
    """
    df = pd.read_csv(registry_path, dtype=str)
    df.columns = df.columns.str.strip()

    stations: dict[str, StationMeta] = {}
    for _, row in df.iterrows():
        stnm = row["stnm"].strip().lower()
        jumps_raw = row.get("jump_epochs", "")
        if pd.isna(jumps_raw) or str(jumps_raw).strip() == "":
            jumps = []
        else:
            jumps = [pd.Timestamp(d.strip()) for d in str(jumps_raw).split(";") if d.strip()]

        stations[stnm] = StationMeta(
            stnm=stnm,
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            alt_m=float(row["alt_m"]),
            network=row["network"].strip(),
            data_start=_parse_ts(row.get("data_start")),
            data_end=_parse_ts(row.get("data_end")),
            jump_epochs=jumps,
        )
    return stations


def list_stations(registry_path: Path | str) -> list[str]:
    """Return sorted list of station codes from registry."""
    df = pd.read_csv(registry_path, dtype=str, usecols=["stnm"])
    return sorted(df["stnm"].str.strip().str.lower().tolist())
