"""JPL GNSS orbit and clock product path resolution.

GipsyX PPP requires final JPL products (orbits + clocks) for each
processing day. Products are organized by year:
    {jpl_root}/{YYYY}/

Reference: otl_proc/cmd_dir/dl_orbits.sh
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_products_dir(jpl_root: Path, obs_date: date) -> Path:
    """Return the JPL products directory for a given date.

    Expected layout::

        {jpl_root}/{YYYY}/

    Args:
        jpl_root: Root directory of JPL GNSS products.
        obs_date: Observation date.

    Returns:
        Path to the year directory containing products.

    Raises:
        FileNotFoundError: if the year directory does not exist.
    """
    year_dir = Path(jpl_root) / str(obs_date.year)
    if not year_dir.is_dir():
        raise FileNotFoundError(
            f"JPL products not found for {obs_date.year}: {year_dir}"
        )
    return year_dir


def check_products_available(
    jpl_root: Path,
    dates: list[date],
) -> tuple[list[date], list[date]]:
    """Check which dates have JPL products available.

    Args:
        jpl_root: Root directory of JPL GNSS products.
        dates: List of dates to check.

    Returns:
        Tuple of (available_dates, missing_dates).
    """
    jpl_root = Path(jpl_root)
    available: list[date] = []
    missing: list[date] = []

    for d in dates:
        year_dir = jpl_root / str(d.year)
        if year_dir.is_dir():
            available.append(d)
        else:
            missing.append(d)

    return available, missing
