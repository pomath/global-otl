"""Single-window GipsyX PPP processing.

Runs gd2e.py for one overlapping time window of a data record.
Each window spans 30 hours: 3 hours before the day boundary to
27 hours after the next day boundary.

Reference: otl_proc/cmd_dir/par_gipsy.sh command.sh (lines 52-72)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result of processing one time window."""

    index: int
    start_sec: float
    end_sec: float
    tdp_path: Path | None  # None if failed
    error: str | None  # Error message if failed
    elapsed_sec: float


def process_window(
    dr_path: Path,
    stnm: str,
    index: int,
    start_sec: float,
    end_sec: float,
    jpl_products_dir: Path,
    tree_dir: Path,
    stadb_path: Path,
    otl_stadb_path: Path | None,
    work_dir: Path,
    output_tdp: Path,
    gipsyx_env: dict[str, str],
    timeout_sec: int = 3600,
) -> WindowResult:
    """Run gd2e.py for one processing window.

    Replicates par_gipsy.sh command.sh::

        gd2e.py -drEditedFile {dr_path}
                -treeSequence Trees/
                -recList {STNM}
                -staDb stadb.db
                -startTime {start_sec}
                -endTime {end_sec}
                -GNSSproducts {jpl_dir}

    Args:
        dr_path: Path to the merged data record (.dr.gz).
        stnm: 4-letter station code.
        index: Window index (for output naming).
        start_sec: Start time in GPS seconds.
        end_sec: End time in GPS seconds.
        jpl_products_dir: Directory with JPL orbit/clock products.
        tree_dir: Path to Trees/ directory containing ppp0_0.tree.
        stadb_path: Station database file.
        otl_stadb_path: OTL station database (.otl file). Copied into
                        the working directory as otlstadb.db for GipsyX.
        work_dir: Base working directory (a subdirectory will be created).
        output_tdp: Where to write the output TDP file.
        gipsyx_env: Environment dict with GipsyX on PATH.
        timeout_sec: Subprocess timeout.

    Returns:
        WindowResult with status and output path.
    """
    t0 = time.monotonic()
    win_dir = Path(work_dir) / f"win_{index:04d}"
    win_dir.mkdir(parents=True, exist_ok=True)

    # Copy tree and station DB into window working directory
    win_trees = win_dir / "Trees"
    shutil.copytree(tree_dir, win_trees, dirs_exist_ok=True)
    shutil.copy2(stadb_path, win_dir / "stadb.db")

    if otl_stadb_path is not None:
        # GipsyX looks up station by uppercase name in the OTL file.
        # Our .otl files use lowercase; uppercase the station name line.
        otl_text = Path(otl_stadb_path).read_text()
        otl_text = otl_text.replace(
            f"  {stnm.lower()}\n", f"  {stnm.upper()}\n"
        )
        (win_dir / "otlstadb.db").write_text(otl_text)

    cmd = [
        "gd2e.py",
        "-drEditedFile", str(Path(dr_path).resolve()),
        "-treeSequenceDir", str(win_trees.resolve()) + "/",
        "-recList", stnm.upper(),
        "-staDb", str((win_dir / "stadb.db").resolve()),
        "-startTime", str(int(start_sec)),
        "-endTime", str(int(end_sec)),
        "-selectGnss", "GPS",
    ]

    # Auto-detect product format:
    #   GCORE format (from fetchGNSSproducts.py): flat dir with GNSS.pos → use -orbClkDir
    #   GOA format (year subdirs): YYYY/YYYY-MM-DD.pos.gz → use -GNSSproducts
    products_path = Path(jpl_products_dir).resolve()
    if (products_path / "GNSS.pos").exists() or (products_path / "GNSS.meta").exists():
        cmd += ["-orbClkDir", str(products_path)]
    else:
        cmd += ["-GNSSproducts", str(products_path)]

    log.debug("gd2e.py cmd: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(win_dir),
            capture_output=True,
            text=True,
            env=gipsyx_env,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return WindowResult(
            index=index,
            start_sec=start_sec,
            end_sec=end_sec,
            tdp_path=None,
            error=f"Timeout after {timeout_sec}s",
            elapsed_sec=elapsed,
        )

    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        error = _diagnose_failure(
            result, win_dir, dr_path, jpl_products_dir,
            start_sec, end_sec, gipsyx_env,
        )
        return WindowResult(
            index=index,
            start_sec=start_sec,
            end_sec=end_sec,
            tdp_path=None,
            error=error,
            elapsed_sec=elapsed,
        )

    # gd2e.py writes use_me.tdp in the working directory
    tdp_file = win_dir / "use_me.tdp"
    if not tdp_file.exists():
        error = _diagnose_missing_output(
            result, win_dir, dr_path, start_sec, end_sec,
        )
        return WindowResult(
            index=index,
            start_sec=start_sec,
            end_sec=end_sec,
            tdp_path=None,
            error=error,
            elapsed_sec=elapsed,
        )

    # Move to output location
    output_tdp = Path(output_tdp)
    output_tdp.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tdp_file), str(output_tdp))

    return WindowResult(
        index=index,
        start_sec=start_sec,
        end_sec=end_sec,
        tdp_path=output_tdp,
        error=None,
        elapsed_sec=elapsed,
    )


def _diagnose_failure(
    result: subprocess.CompletedProcess,
    win_dir: Path,
    dr_path: Path,
    jpl_products_dir: Path,
    start_sec: float,
    end_sec: float,
    gipsyx_env: dict[str, str],
) -> str:
    """Build a useful error message when gd2e.py exits non-zero.

    gd2e.py often exits with no stderr at all. This function checks
    the most likely causes and returns a human-readable diagnosis.
    """
    stderr = result.stderr.strip() if result.stderr else ""
    stdout = result.stdout.strip() if result.stdout else ""
    hints: list[str] = []

    # ── Check data record ────────────────────────────────────────
    if not Path(dr_path).exists():
        hints.append(f"Data record not found: {dr_path}")
    elif Path(dr_path).stat().st_size == 0:
        hints.append(f"Data record is empty (0 bytes): {dr_path}")

    # ── Check JPL products ───────────────────────────────────────
    _check_products(jpl_products_dir, start_sec, end_sec, hints)

    # ── Check station DB ─────────────────────────────────────────
    stadb = win_dir / "stadb.db"
    if not stadb.exists():
        hints.append("Station database missing from working directory")
    elif stadb.stat().st_size < 50:
        hints.append(f"Station database suspiciously small ({stadb.stat().st_size} bytes)")

    # ── Check tree configuration ─────────────────────────────────
    tree_file = win_dir / "Trees" / "ppp0_0.tree"
    if not tree_file.exists():
        # Try alternate tree names
        trees = list((win_dir / "Trees").glob("*.tree")) if (win_dir / "Trees").exists() else []
        if not trees:
            hints.append("No .tree files found in Trees/ directory")
        else:
            hints.append(f"ppp0_0.tree not found; found: {[t.name for t in trees[:5]]}")

    # ── Check for common stderr patterns ─────────────────────────
    combined = stderr + "\n" + stdout
    if "No such file or directory" in combined:
        hints.append("A required file or directory was not found (check paths in log)")
    if "products" in combined.lower() and ("not found" in combined.lower() or "missing" in combined.lower()):
        hints.append("JPL orbit/clock products appear to be missing for this time window")
    if "singular" in combined.lower() or "non-invertible" in combined.lower():
        hints.append("Normal equation matrix is singular — too few valid observations")
    if "antenna" in combined.lower():
        hints.append("Possible antenna model issue — check station database")
    if "ImportError" in combined or "ModuleNotFoundError" in combined:
        hints.append("Python import error — GipsyX Python environment may be misconfigured")
    if "MemoryError" in combined or "killed" in combined.lower():
        hints.append("Process ran out of memory or was killed by the OS")
    if "VMF1" in combined or "vmf" in combined.lower():
        hints.append("VMF1 troposphere data issue — check vmf1_dir path or switch to GMF")

    # ── Check for smoothed output (partial success) ──────────────
    smth = list(win_dir.glob("smoothFinal*.tdp"))
    if smth:
        hints.append(f"Smoother ran ({len(smth)} smoothed files) but final output failed")

    # ── Build message ────────────────────────────────────────────
    msg = f"gd2e.py exit {result.returncode}"
    if stderr:
        msg += f": {stderr[:300]}"
    if hints:
        msg += "\n  Likely cause(s):\n    - " + "\n    - ".join(hints)
    elif not stderr:
        msg += " (no stderr output)"
        msg += "\n  No specific cause identified. Check the gd2e.py log in: " + str(win_dir)

    return msg


def _diagnose_missing_output(
    result: subprocess.CompletedProcess,
    win_dir: Path,
    dr_path: Path,
    start_sec: float,
    end_sec: float,
) -> str:
    """Build a useful error message when gd2e.py exits 0 but use_me.tdp is missing."""
    hints: list[str] = []

    # Check what files gd2e.py did produce
    tdp_files = list(win_dir.glob("*.tdp"))
    if tdp_files:
        hints.append(f"Other .tdp files exist ({[f.name for f in tdp_files[:5]]}) but use_me.tdp was not generated")
        hints.append("The smoother may have run but the final combination step failed")
    else:
        hints.append("No .tdp files produced at all")
        hints.append("Likely too few valid observations in this time window for the filter to converge")

    # Check window duration
    duration_hours = (end_sec - start_sec) / 3600
    if duration_hours < 6:
        hints.append(f"Short window ({duration_hours:.1f}h) — may not have enough data")

    # Check for debug/residual files that indicate partial processing
    if (win_dir / "debug").exists():
        hints.append("A debug/ directory exists — check it for filter convergence details")
    resid = list(win_dir.glob("*.resid*"))
    if resid:
        hints.append(f"Residual files found ({len(resid)}) — processing started but did not complete")

    msg = "gd2e.py exited successfully but use_me.tdp not found"
    if hints:
        msg += "\n  Likely cause(s):\n    - " + "\n    - ".join(hints)

    return msg


def _check_products(jpl_products_dir: Path, start_sec: float, end_sec: float,
                    hints: list[str]) -> None:
    """Check if JPL products exist for the window's date range."""
    from datetime import date, timedelta

    # GPS epoch: 2000-01-01 12:00:00 UTC
    _GPS_EPOCH_UNIX = 946728000.0
    # Approximate conversion from GPS seconds to dates
    try:
        start_unix = start_sec + _GPS_EPOCH_UNIX
        end_unix = end_sec + _GPS_EPOCH_UNIX
        from datetime import datetime, timezone
        start_dt = datetime.fromtimestamp(start_unix, tz=timezone.utc).date()
        end_dt = datetime.fromtimestamp(end_unix, tz=timezone.utc).date()
    except (ValueError, OSError, OverflowError):
        return  # Can't convert; skip product checks

    jpl = Path(jpl_products_dir)
    if not jpl.is_dir():
        hints.append(f"JPL products directory does not exist: {jpl}")
        return

    current = start_dt
    missing_years = set()
    while current <= end_dt:
        year_dir = jpl / str(current.year)
        if not year_dir.is_dir():
            missing_years.add(current.year)
        current += timedelta(days=1)

    if missing_years:
        hints.append(
            f"JPL products missing for year(s): {sorted(missing_years)} "
            f"(expected at {jpl}/YYYY/)"
        )


def compute_processing_windows(
    start_sec: float,
    end_sec: float,
) -> list[tuple[int, float, float]]:
    """Compute overlapping 30-hour processing windows.

    Replicates par_gipsy.sh window logic (lines 44-50)::

        all_dates = seq(start_sec, end_sec, 86400)
        for i in 0..len-2:
            window_start = all_dates[i] - 10800    # 3 hours before
            window_end   = all_dates[i+1] + 97200  # 27 hours after

    Args:
        start_sec: Start time in GPS seconds (midnight of first day).
        end_sec: End time in GPS seconds (midnight of last day).

    Returns:
        List of (index, window_start_sec, window_end_sec) tuples.
    """
    day_secs = 86400
    pre_buffer = 10800   # 3 hours
    post_buffer = 97200  # 27 hours

    # Generate midnight boundaries
    all_dates = []
    t = start_sec
    while t <= end_sec:
        all_dates.append(t)
        t += day_secs

    windows = []
    for i in range(len(all_dates) - 1):
        w_start = all_dates[i] - pre_buffer
        w_end = all_dates[i + 1] + post_buffer
        windows.append((i, w_start, w_end))

    return windows
