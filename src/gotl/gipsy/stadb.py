"""Station database creation for GipsyX.

Wraps the GipsyX ``rinex2StaDb.py`` utility to create a station database
from RINEX observation file headers. The station database contains antenna
type, receiver info, and approximate coordinates needed by gd2e.py.

Reference: otl_proc/cmd_dir/old/mk_drfile.sh line:
    rinex2StaDb.py -outFile {stn}.stadb {stn}_edited/*.Z
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def create_stadb(
    rinex_files: list[Path],
    stnm: str,
    output_path: Path,
    gipsyx_env: dict[str, str],
    timeout_sec: int = 120,
) -> Path:
    """Create a GipsyX station database from RINEX file headers.

    Args:
        rinex_files: Paths to RINEX observation files (can be compressed).
        stnm: 4-letter station code.
        output_path: Where to write the .stadb file.
        gipsyx_env: Environment dict with GipsyX on PATH.
        timeout_sec: Subprocess timeout.

    Returns:
        Path to the created station database file.

    Raises:
        RuntimeError: if rinex2StaDb.py fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rinex2StaDb.py",
        "-outFile", str(output_path),
    ] + [str(f) for f in rinex_files]

    log.info("Creating station DB for %s from %d RINEX files", stnm, len(rinex_files))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=gipsyx_env,
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rinex2StaDb.py failed for {stnm}: {result.stderr.strip()}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"rinex2StaDb.py returned 0 but output not found: {output_path}"
        )

    _normalize_station_id(output_path, stnm)

    log.info("Station DB created: %s", output_path)
    return output_path


def _normalize_station_id(stadb_path: Path, stnm: str) -> None:
    """Rewrite the stadb so every station ID equals ``stnm.upper()``.

    rinex2StaDb.py keys entries off the RINEX ``MARKER NAME`` header. RINEX 3
    files often use the 9-char site identifier (e.g. ``MAC100AUS``), which
    won't match the 4-char code we pass to ``gd2e.py -recList``. Without this
    rewrite, ``antennaCalFiles`` can't find the station and gd2e.py dies in
    ``prepareRunDirSubDir``.

    Year-long RINEX archives can also mix marker formats (e.g. some files
    tagged ``ARHT`` and others ``ARHT00ATA``), producing a stadb with
    multiple distinct IDs that map to the same station. Coalesce all of
    them onto the target. After rewriting, drop duplicate ID records — when
    two different markers each had their own ID line, post-rewrite they
    collapse to two lines with the same station name, and StationDataBase.py
    refuses to load that with "duplicate ID records". State records (ANT,
    RX, POS, etc.) keep all entries; only the ID line is deduped.
    """
    text = stadb_path.read_text()
    target = stnm.upper()

    ids: set[str] = set()
    for line in text.splitlines():
        if not line or line.startswith(("#", "KEYWORDS:")):
            continue
        first = line.split(None, 1)[0]
        ids.add(first)

    sources = sorted(ids - {target})
    needs_rewrite = bool(sources)
    if needs_rewrite:
        for source in sources:
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)

    new_lines: list[str] = []
    seen_id = False
    dropped_ids = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "KEYWORDS:")):
            parts = stripped.split()
            if len(parts) >= 2 and parts[0] == target and parts[1] == "ID":
                if seen_id:
                    dropped_ids += 1
                    continue
                seen_id = True
        new_lines.append(line)

    if not (needs_rewrite or dropped_ids):
        return

    trailing_newline = "\n" if text.endswith("\n") else ""
    stadb_path.write_text("\n".join(new_lines) + trailing_newline)
    if needs_rewrite:
        log.info("Normalized stadb station IDs %s → %s", sources, target)
    if dropped_ids:
        log.info("Dropped %d duplicate ID record(s) for %s", dropped_ids, target)
