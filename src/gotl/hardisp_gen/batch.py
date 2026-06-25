"""Batch hardisp generation for all stations and years.

Generates 5-min tidal displacement time series for each station and year
by calling the hardisp binary (one call per station-day, parallelized via
GNU parallel in the shell scripts, or sequentially here).

Output files: {hardisp_root}/{stnm}/{stnm}-hardisp_{YYYY}.txt
Format: one row per 5-min sample
    year doy hour minute second dU(m) dS(m) dW(m)
"""

from __future__ import annotations

import calendar
import logging
from pathlib import Path

import pandas as pd

from gotl.hardisp_gen.runner import run_hardisp_day

log = logging.getLogger(__name__)

_OUTPUT_HEADER = "# year doy hour minute second dU(m) dS(m) dW(m)"


def generate_hardisp_year(
    hardisp_exe: Path,
    otl_db_path: Path,
    stnm: str,
    year: int,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Generate hardisp time series for one station and one year.

    Args:
        hardisp_exe: Path to hardisp binary.
        otl_db_path: Harpos .db file for this station.
        stnm: Station code (4-letter lowercase).
        year: Year to process.
        output_dir: Directory to write output file into.
        force: Re-run even if output file already exists.

    Returns:
        Path to output file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{stnm}-hardisp_{year}.txt"

    if outfile.exists() and not force:
        log.debug("hardisp %s %d: cached, skipping", stnm, year)
        return outfile

    n_days = 366 if calendar.isleap(year) else 365
    rows: list[str] = []

    for doy in range(1, n_days + 1):
        stdout = run_hardisp_day(hardisp_exe, otl_db_path, year, doy)
        # hardisp outputs 3 columns: dU dS dW (no time info)
        # Prepend year doy hour minute second for each 5-min sample
        raw_lines = [ln for ln in stdout.strip().split("\n") if ln.strip()]
        for i, ln in enumerate(raw_lines):
            h = (i * 5) // 60
            m = (i * 5) % 60
            rows.append(f"{year} {doy:03d} {h:02d} {m:02d} 00 {ln.strip()}")

    outfile.write_text("\n".join(rows) + "\n")
    log.info("hardisp %s %d: wrote %s", stnm, year, outfile)
    return outfile


def generate_hardisp_station(
    hardisp_exe: Path,
    otl_params_dir: Path,
    stnm: str,
    years: list[int],
    hardisp_root: Path,
    model: str = "TPXO9",
    force: bool = False,
) -> dict[int, Path]:
    """Generate hardisp for all requested years for one station.

    Args:
        hardisp_exe: Path to hardisp binary.
        otl_params_dir: Directory containing {stnm}_{model}.otl files.
        stnm: Station code.
        years: List of years to process.
        hardisp_root: Root directory for hardisp output.
        model: Tide model name (used to select the .otl file).
        force: Re-run even if cached.

    Returns:
        Dict mapping year → output file path.
    """
    otl_file = otl_params_dir / f"{stnm}_{model}.otl"
    if not otl_file.exists():
        raise FileNotFoundError(f"OTL file not found: {otl_file}")

    output_dir = hardisp_root / stnm
    results: dict[int, Path] = {}
    for year in years:
        results[year] = generate_hardisp_year(
            hardisp_exe, otl_file, stnm, year, output_dir, force=force
        )
    return results
