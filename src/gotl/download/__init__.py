"""Download RINEX observation files from GNSS data archives."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Summary of a RINEX download run."""

    stnm: str
    dates_requested: int = 0
    dates_downloaded: int = 0
    dates_skipped: int = 0
    dates_failed: int = 0
    failures: dict[date, str] = field(default_factory=dict)


def download_rinex_range(
    stnm: str,
    start_date: date,
    end_date: date,
    rinex_root: Path,
    archives: list[str] | None = None,
    skip_existing: bool = True,
    max_retries: int = 3,
    dry_run: bool = False,
) -> DownloadResult:
    """Download daily RINEX files for a station over a date range.

    Tries each archive in order until one succeeds for each date.
    Files are saved to ``{rinex_root}/{stnm}/{filename}``.

    Args:
        stnm: 4-letter station code.
        start_date: First date to download (inclusive).
        end_date: Last date to download (inclusive).
        rinex_root: Root directory for RINEX files.
        archives: Archive names to try, in priority order.
            Default: ["cddis", "bkg"].
        skip_existing: Skip dates where a local file already exists.
        max_retries: Max HTTP retries per archive per date.
        dry_run: If True, log URLs without downloading.

    Returns:
        DownloadResult with counts and per-date failure details.
    """
    from gotl.download.archives import get_archives

    archive_list = get_archives(archives or ["cddis", "bkg"])
    dest_dir = Path(rinex_root) / stnm.lower()
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = DownloadResult(stnm=stnm)
    current = start_date
    while current <= end_date:
        result.dates_requested += 1
        _download_one_day(
            stnm, current, dest_dir, archive_list,
            skip_existing, max_retries, dry_run, result,
        )
        current += timedelta(days=1)

    return result


def _download_one_day(stnm, obs_date, dest_dir, archives, skip_existing,
                      max_retries, dry_run, result):
    """Try to download one day's RINEX file from the archive list."""
    # Check if any file for this station/date already exists locally
    if skip_existing and _local_file_exists(stnm, obs_date, dest_dir):
        log.debug("Skipping %s (exists locally)", obs_date)
        result.dates_skipped += 1
        return

    if dry_run:
        urls = [a.url_for(stnm, obs_date) for a in archives]
        log.info("[dry-run] %s → %s", obs_date, " | ".join(urls))
        result.dates_skipped += 1
        return

    for archive in archives:
        try:
            fname = archive.download_day(stnm, obs_date, dest_dir,
                                         max_retries=max_retries)
            log.info("Downloaded %s from %s: %s", obs_date, archive.name, fname)
            result.dates_downloaded += 1
            return
        except Exception as e:
            log.debug("Archive %s failed for %s %s: %s",
                      archive.name, stnm, obs_date, e)
            continue

    # All archives failed
    err = f"No archive had data for {stnm} on {obs_date}"
    log.warning(err)
    result.dates_failed += 1
    result.failures[obs_date] = err


def _local_file_exists(stnm: str, obs_date: date, dest_dir: Path) -> bool:
    """Check if a RINEX file for this station/date already exists."""
    from gotl.download.archives import _date_parts

    yyyy, doy, yy = _date_parts(obs_date)
    stnm_upper = stnm.upper()

    # Check RINEX 2 patterns
    for ext in ("d.Z", "d.gz", "o.Z", "o.gz"):
        if (dest_dir / f"{stnm_upper}{doy:03d}0.{yy:02d}{ext}").exists():
            return True

    # Check RINEX 3 pattern (any country code)
    import re
    pattern = re.compile(
        rf"^{stnm_upper}\d{{2}}[A-Z]{{3}}_R_{yyyy}{doy:03d}\d{{4}}_",
        re.IGNORECASE,
    )
    if dest_dir.exists():
        for f in dest_dir.iterdir():
            if pattern.match(f.name):
                return True
    return False
