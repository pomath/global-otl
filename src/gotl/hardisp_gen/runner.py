"""Run the hardisp binary to generate 5-min tidal displacement time series.

The hardisp binary reads OTL coefficients (Harpos format) from stdin and
writes a tidal time series to stdout:

    cat {stn}.db_stripped | hardisp_exe <year> <doy> 0 0 0 288 300

Arguments:
    year:    4-digit year
    doy:     day of year (1-365/366)
    0 0 0:   start time offset (hours, minutes, seconds from midnight)
    288:     number of samples
    300:     sample interval in seconds (5 min)

Output columns: year doy hour minute second dU dS dW  (displacements in metres)

The OTL input file is the stripped version of the Harpos .db file:
    sed '34,39!d' {stn}_otl.db  →  6 lines of amplitude/phase data
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_NSAMPLES = 288       # 288 × 5 min = 1 day
_INTERVAL_SEC = 300   # 5 minutes


def run_hardisp_day(
    hardisp_exe: Path,
    otl_db_path: Path,
    year: int,
    doy: int,
) -> str:
    """Run hardisp for one station-day; return stdout as string.

    Args:
        hardisp_exe: Path to compiled hardisp binary.
        otl_db_path: Path to Harpos-format .db or .otl file.
                     Lines 34-39 (1-indexed) are the amplitude/phase data.
        year: 4-digit year.
        doy:  Day of year (1–366).

    Returns:
        Raw hardisp stdout (one row per sample).

    Raises:
        RuntimeError: if hardisp exits non-zero.
    """
    stripped = _strip_otl(otl_db_path)
    cmd = [str(hardisp_exe), str(year), str(doy), "0", "0", "0",
           str(_NSAMPLES), str(_INTERVAL_SEC)]
    result = subprocess.run(
        cmd,
        input=stripped,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"hardisp failed for {year} DOY {doy}: {result.stderr.strip()}"
        )
    return result.stdout


def _strip_otl(otl_db_path: Path) -> str:
    """Extract lines 34-39 (1-indexed) from a Harpos .db file."""
    lines = Path(otl_db_path).read_text().splitlines()
    data_lines = lines[33:39]  # 0-indexed: lines 34-39
    return "\n".join(data_lines) + "\n"
