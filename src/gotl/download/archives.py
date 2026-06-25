"""GNSS data archive backends for RINEX downloads."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

# HTTP status codes that warrant a retry
_RETRYABLE = {429, 500, 502, 503, 504}


class Archive(ABC):
    """Base class for a GNSS data archive."""

    name: str

    @abstractmethod
    def download_day(self, stnm: str, obs_date: date, dest_dir: Path,
                     max_retries: int = 3) -> str:
        """Download one day's RINEX file for a station.

        Args:
            stnm: 4-letter station code.
            obs_date: Date to download.
            dest_dir: Directory to save the file in.
            max_retries: Max retries per URL.

        Returns:
            The filename that was saved.

        Raises:
            FileNotFoundError: If the file is not available.
            IOError: On download failure.
        """

    def url_for(self, stnm: str, obs_date: date) -> str:
        """Return a representative URL for dry-run display."""
        yyyy, doy, yy = _date_parts(obs_date)
        return f"<{self.name}:{stnm.upper()}/{yyyy}/{doy:03d}>"

    def filename(self, stnm: str, obs_date: date) -> str:
        """Return the expected RINEX 2 filename."""
        yyyy, doy, yy = _date_parts(obs_date)
        return f"{stnm.upper()}{doy:03d}0.{yy:02d}d.Z"


def _date_parts(obs_date: date) -> tuple[int, int, int]:
    """Return (YYYY, DOY, YY) for a date."""
    doy = obs_date.timetuple().tm_yday
    yy = obs_date.year % 100
    return obs_date.year, doy, yy


class CDDISArchive(Archive):
    """NASA CDDIS archive — requires Earthdata auth.

    Downloads Hatanaka-compressed RINEX 2 daily files (.{YY}d.Z).
    """

    name = "cddis"
    _BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"

    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            from gotl.download.auth import get_earthdata_session
            self._session = get_earthdata_session()
        return self._session

    def url_for(self, stnm: str, obs_date: date) -> str:
        yyyy, doy, yy = _date_parts(obs_date)
        fname = self.filename(stnm, obs_date)
        return f"{self._BASE}/{yyyy}/{doy:03d}/{yy:02d}d/{fname}"

    def download_day(self, stnm: str, obs_date: date, dest_dir: Path,
                     max_retries: int = 3) -> str:
        session = self._get_session()
        yyyy, doy, yy = _date_parts(obs_date)

        # CDDIS serves lowercase + .gz today; older files used uppercase + .Z.
        # Try Hatanaka (.d) then observation (.o), each in both conventions.
        lo, hi = stnm.lower(), stnm.upper()
        candidates = [
            f"{lo}{doy:03d}0.{yy:02d}d.gz",
            f"{lo}{doy:03d}0.{yy:02d}o.gz",
            f"{hi}{doy:03d}0.{yy:02d}d.Z",
            f"{hi}{doy:03d}0.{yy:02d}o.Z",
        ]
        for fname in candidates:
            url = f"{self._BASE}/{yyyy}/{doy:03d}/{yy:02d}d/{fname}"
            dest = dest_dir / fname
            try:
                _http_download(session, url, dest, max_retries)
                return fname
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"CDDIS: no file found for {stnm} on {obs_date}"
        )


class BKGArchive(Archive):
    """BKG (Germany) archive — HTTPS, no auth.

    Serves RINEX 3 long filenames. Falls back to directory listing
    to find the correct country code for RINEX 3 filenames.
    """

    name = "bkg"
    _BASE = "https://igs.bkg.bund.de/root_ftp/IGS/obs"

    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def url_for(self, stnm: str, obs_date: date) -> str:
        yyyy, doy, yy = _date_parts(obs_date)
        return f"{self._BASE}/{yyyy}/{doy:03d}/ [{stnm.upper()}*.crx.gz]"

    def download_day(self, stnm: str, obs_date: date, dest_dir: Path,
                     max_retries: int = 3) -> str:
        session = self._get_session()
        yyyy, doy, yy = _date_parts(obs_date)
        dir_url = f"{self._BASE}/{yyyy}/{doy:03d}/"

        # Find matching RINEX 3 filename from directory listing
        stnm_upper = stnm.upper()
        pattern = re.compile(
            rf"{stnm_upper}\d{{2}}[A-Z]{{3}}_R_{yyyy}{doy:03d}\d{{4}}_\d{{2}}D_\d{{2,3}}S_MO"
            rf"\.(crx|rnx)(\.gz)?",
            re.IGNORECASE,
        )

        try:
            resp = session.get(dir_url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            raise FileNotFoundError(f"BKG: cannot list {dir_url}: {e}")

        matches = pattern.findall(resp.text)
        if not matches:
            raise FileNotFoundError(
                f"BKG: no RINEX file for {stnm} on {obs_date}"
            )

        # Reconstruct the full filename from the first regex match
        # pattern captures (ext, gz_ext), so re-search for full match
        full_match = pattern.search(resp.text)
        fname = full_match.group(0)
        file_url = dir_url + fname
        dest = dest_dir / fname

        _http_download(session, file_url, dest, max_retries)
        return fname


def _http_download(session, url: str, dest: Path, max_retries: int) -> None:
    """Download a URL to dest with retries and atomic write."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, stream=True, timeout=60,
                               allow_redirects=True)
            if resp.status_code == 404:
                raise FileNotFoundError(f"404 Not Found: {url}")
            if resp.status_code in _RETRYABLE:
                raise IOError(f"HTTP {resp.status_code} for {url}")
            resp.raise_for_status()

            # Atomic write: download to temp, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=dest.parent, suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                os.rename(tmp_path, dest)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return

        except FileNotFoundError:
            raise  # Don't retry 404s
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.debug("Retry %d/%d for %s in %ds: %s",
                          attempt + 1, max_retries, url, wait, e)
                time.sleep(wait)

    raise last_err or IOError(f"Download failed: {url}")


def get_archives(names: list[str]) -> list[Archive]:
    """Instantiate Archive objects by name."""
    registry = {
        "cddis": CDDISArchive,
        "bkg": BKGArchive,
    }
    result = []
    for name in names:
        name = name.strip().lower()
        if name not in registry:
            raise ValueError(
                f"Unknown archive '{name}'. Available: {', '.join(registry)}"
            )
        result.append(registry[name]())
    return result
