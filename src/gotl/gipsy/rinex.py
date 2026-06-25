"""RINEX observation file discovery and filename parsing.

Supports RINEX 2 and RINEX 3 naming conventions, including compressed
variants (.Z, .gz).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# RINEX2: {STNM}{DOY}0.{YY}o  optionally followed by .Z or .gz
# Example: ABOA0010.10o.Z
_RNX2_RE = re.compile(
    r"^([A-Za-z0-9]{4})(\d{3})0\.(\d{2})[oOdD](\.Z|\.gz)?$"
)

# RINEX3: {STNM}00{CC}_R_{YYYY}{DOY}0000_01D_30S_MO.(rnx|crx)(.gz)?
# Example: ABOA00ATA_R_20100010000_01D_30S_MO.crx.gz
_RNX3_RE = re.compile(
    r"^([A-Za-z0-9]{4})\d{2}[A-Z]{3}_R_(\d{4})(\d{3})\d{4}_\d{2}D_\d{2,3}S_MO"
    r"\.(rnx|crx)(\.gz)?$"
)


@dataclass
class RinexFile:
    """Metadata for one RINEX observation file."""

    path: Path
    stnm: str  # 4-char lowercase
    date: date
    compressed: bool  # .Z or .gz


def _parse_rinex2_name(name: str) -> tuple[str, date, bool] | None:
    """Parse a RINEX2 filename into (stnm, date, compressed)."""
    m = _RNX2_RE.match(name)
    if not m:
        return None
    stnm = m.group(1).lower()
    doy = int(m.group(2))
    yy = int(m.group(3))
    year = 1900 + yy if yy >= 80 else 2000 + yy
    d = date(year, 1, 1) + timedelta(days=doy - 1)
    compressed = m.group(4) is not None
    return stnm, d, compressed


def _parse_rinex3_name(name: str) -> tuple[str, date, bool] | None:
    """Parse a RINEX3 long filename into (stnm, date, compressed)."""
    m = _RNX3_RE.match(name)
    if not m:
        return None
    stnm = m.group(1).lower()
    year = int(m.group(2))
    doy = int(m.group(3))
    d = date(year, 1, 1) + timedelta(days=doy - 1)
    compressed = m.group(5) is not None
    return stnm, d, compressed


def parse_rinex_name(name: str) -> tuple[str, date, bool] | None:
    """Parse a RINEX filename (v2 or v3) into (stnm, date, compressed).

    Returns None if the filename does not match any known pattern.
    """
    return _parse_rinex2_name(name) or _parse_rinex3_name(name)


def discover_rinex(
    rinex_dir: Path,
    stnm: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[RinexFile]:
    """Find all RINEX observation files for a station.

    Searches in ``{rinex_dir}/{stnm}/`` first; falls back to
    ``{rinex_dir}/`` with station-code filtering.

    Args:
        rinex_dir: Root directory containing RINEX files.
        stnm: 4-letter station code (case-insensitive).
        start_date: Include files on or after this date.
        end_date: Include files on or before this date.

    Returns:
        List of RinexFile sorted by date.

    Raises:
        FileNotFoundError: if no RINEX files are found.
    """
    stnm = stnm.lower()
    stn_dir = Path(rinex_dir) / stnm
    if stn_dir.is_dir():
        search_dir = stn_dir
    else:
        search_dir = Path(rinex_dir)

    if not search_dir.is_dir():
        raise FileNotFoundError(f"RINEX directory not found: {search_dir}")

    results: list[RinexFile] = []
    for fpath in search_dir.iterdir():
        if not fpath.is_file():
            continue
        parsed = parse_rinex_name(fpath.name)
        if parsed is None:
            continue
        file_stnm, file_date, compressed = parsed
        if file_stnm != stnm:
            continue
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue
        results.append(RinexFile(
            path=fpath,
            stnm=file_stnm,
            date=file_date,
            compressed=compressed,
        ))

    if not results:
        raise FileNotFoundError(
            f"No RINEX files found for station '{stnm}' in {search_dir}"
        )

    results.sort(key=lambda r: r.date)
    return results
