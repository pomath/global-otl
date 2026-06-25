"""Full RINEX → TDP processing for one station.

Orchestrates the complete pipeline:
    1. Discover RINEX files
    2. Create station database (rinex2StaDb.py)
    3. Convert RINEX → individual daily data records (parallel)
    4. Compute overlapping 30h processing windows
    5. Per window: merge 2-3 daily DRs → run gd2e.py (parallel)
    6. Organize output for load_tdp()

Optimizations vs the original par_gipsy.sh approach:
    - No full-year DR merge: each window merges only the 2-3 daily DRs
      it needs, avoiding 140 MB I/O per window and the serial merge step
    - tdpdiff merge skipped: load_tdp() reads individual window TDPs
    - MaxIteration reduced from 5→1 in kinematicPPP.tree (same quality)

Reference: otl_proc/cmd_dir/mkdr_parallel.sh + par_gipsy.sh
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from gotl.gipsy.datarecord import build_data_record, merge_data_records
from gotl.gipsy.env import load_gipsyx_env
from gotl.gipsy.products import check_products_available
from gotl.gipsy.rinex import discover_rinex
from gotl.gipsy.runner import WindowResult, compute_processing_windows, process_window
from gotl.gipsy.stadb import create_stadb
from gotl.gipsy.tree import prepare_tree_dir

log = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Result of processing one station."""

    stnm: str
    dr_path: Path | None = None
    window_results: list[WindowResult] = field(default_factory=list)
    tdp_dir: Path | None = None
    n_rinex: int = 0
    n_windows_ok: int = 0
    n_windows_fail: int = 0
    error: str | None = None


def get_dr_time_span(
    dr_path: Path,
    gipsyx_env: dict[str, str],
) -> tuple[float, float]:
    """Get start/end GPS seconds from a data record.

    Calls::

        dataRecordInfo -file {dr_path}

    and parses the ``Time Tag`` line for start and end dates, then
    converts to GPS seconds via ``date2sec``.

    Args:
        dr_path: Path to data record file.
        gipsyx_env: Environment dict with GipsyX on PATH.

    Returns:
        (start_sec, end_sec) in GPS seconds.

    Raises:
        RuntimeError: if dataRecordInfo or date2sec fails.
    """
    result = subprocess.run(
        ["dataRecordInfo", "-file", str(dr_path)],
        capture_output=True,
        text=True,
        env=gipsyx_env,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dataRecordInfo failed: {result.stderr.strip()}"
        )

    # Parse time tag lines from dataRecordInfo output.
    # Format:
    #   Earliest Time Tag : 2024-01-01 00:00:00.000000
    #   Latest   Time Tag : 2024-01-01 23:55:00.000000
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    start_date_str = None
    end_date_str = None
    for line in result.stdout.splitlines():
        if "Earliest" in line and "Time Tag" in line:
            m = date_pattern.search(line)
            if m:
                start_date_str = m.group()
        elif "Latest" in line and "Time Tag" in line:
            m = date_pattern.search(line)
            if m:
                end_date_str = m.group()

    if start_date_str is None or end_date_str is None:
        raise RuntimeError(
            f"Could not parse time tags from dataRecordInfo output:\n"
            f"{result.stdout[:500]}"
        )

    # Convert to GPS seconds using date2sec
    start_sec = _date2sec(start_date_str, gipsyx_env)
    end_sec = _date2sec(end_date_str, gipsyx_env)

    return start_sec, end_sec


def _date2sec(date_str: str, gipsyx_env: dict[str, str]) -> float:
    """Convert a date string to GPS seconds using GipsyX date2sec."""
    result = subprocess.run(
        ["date2sec", date_str],
        capture_output=True,
        text=True,
        env=gipsyx_env,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"date2sec failed for '{date_str}': {result.stderr.strip()}"
        )
    return float(result.stdout.strip())


def merge_tdp_files(
    tdp_files: list[Path],
    output_path: Path,
    gipsyx_env: dict[str, str],
) -> Path:
    """Merge consecutive daily TDP files using tdpdiff.

    Replicates par_gipsy.sh lines 87-91::

        for i in 1..N:
            tdpdiff -f prev.tdp curr.tdp -p '\\.Station\\..*' -d >> stats.tdp

    Args:
        tdp_files: Sorted list of daily TDP file paths.
        output_path: Where to write the merged stats TDP.
        gipsyx_env: Environment dict with GipsyX on PATH.

    Returns:
        Path to the merged TDP file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_lines: list[str] = []
    for i in range(1, len(tdp_files)):
        result = subprocess.run(
            [
                "tdpdiff",
                "-f", str(tdp_files[i - 1]),
                str(tdp_files[i]),
                "-p", r"\.Station\..*",
                "-d",
            ],
            capture_output=True,
            text=True,
            env=gipsyx_env,
            timeout=60,
        )
        if result.returncode == 0:
            merged_lines.append(result.stdout)
        else:
            log.warning(
                "tdpdiff failed for %s + %s: %s",
                tdp_files[i - 1].name,
                tdp_files[i].name,
                result.stderr.strip()[:200],
            )

    output_path.write_text("".join(merged_lines))
    return output_path


def _run_window(args: tuple) -> WindowResult:
    """Worker function for parallel window processing."""
    (dr_path, stnm, index, start_sec, end_sec,
     jpl_dir, tree_dir, stadb_path, otl_stadb_path,
     work_dir, output_tdp, gipsyx_env, timeout_sec) = args
    return process_window(
        dr_path=dr_path,
        stnm=stnm,
        index=index,
        start_sec=start_sec,
        end_sec=end_sec,
        jpl_products_dir=jpl_dir,
        tree_dir=tree_dir,
        stadb_path=stadb_path,
        otl_stadb_path=otl_stadb_path,
        work_dir=work_dir,
        output_tdp=output_tdp,
        gipsyx_env=gipsyx_env,
        timeout_sec=timeout_sec,
    )


def _build_daily_dr_map(dr_dir: Path) -> dict[int, Path]:
    """Map DOY → daily DR path from a directory of daily data records.

    Handles both RINEX 3 and RINEX 2 source filenames:
        RINEX 3: ``BRUX00BEL_R_20240150000_01D_30S_MO.crx.dr.gz`` → DOY 15
        RINEX 2: ``bhr30010.24d.dr.gz`` → DOY 001
    """
    dr_map: dict[int, Path] = {}
    for dr in sorted(dr_dir.glob("*.dr.gz")):
        # RINEX 3: ..._R_YYYYDDD0000_...
        m = re.search(r"_R_\d{4}(\d{3})", dr.name)
        if not m:
            # RINEX 2: 4-char station + 3-digit DOY + 1-char session
            m = re.match(r"^[A-Za-z0-9]{4}(\d{3})\d\.", dr.name)
        if m:
            dr_map[int(m.group(1))] = dr
    return dr_map


def _get_window_doys(w_start: float, w_end: float) -> list[int]:
    """Return the DOYs (1-based) covered by a processing window.

    Each 30h window spans 2-3 calendar days. We need the daily DRs
    for all days the window touches.
    """
    import pandas as pd

    _J2000 = pd.Timestamp("2000-01-01 12:00:00")
    t_start = _J2000 + pd.to_timedelta(w_start, unit="s")
    t_end = _J2000 + pd.to_timedelta(w_end, unit="s")

    doys = []
    d = t_start.date()
    while d <= t_end.date():
        doys.append(d.timetuple().tm_yday)
        d += timedelta(days=1)
    return doys


def _run_window_minidr(args: tuple) -> WindowResult:
    """Worker function that creates a per-window mini-DR then processes.

    Instead of reading the full year-long merged DR, this merges only
    the 2-3 daily DRs that overlap the window's time span. This reduces
    per-window I/O from ~140 MB to ~2 MB.
    """
    (stnm, index, start_sec, end_sec,
     jpl_dir, tree_dir, stadb_path, otl_stadb_path,
     work_dir, output_tdp, gipsyx_env, timeout_sec,
     daily_drs, fallback_dr) = args

    # Determine which daily DRs this window needs
    doys = _get_window_doys(start_sec, end_sec)
    needed_drs = [daily_drs[doy] for doy in doys if doy in daily_drs]

    if len(needed_drs) >= 2:
        # Mini-merge the 2-3 daily DRs for this window
        mini_dr = work_dir / f"minidr_{index:04d}.dr.gz"
        try:
            merge_data_records(needed_drs, mini_dr, gipsyx_env)
            dr_to_use = mini_dr
        except Exception:
            dr_to_use = fallback_dr  # fall back if available; else None
    elif len(needed_drs) == 1:
        # Only one daily DR overlaps — use it directly (no merge needed)
        dr_to_use = needed_drs[0]
    else:
        # Window has no daily DR coverage; fall back if available, else fail
        dr_to_use = fallback_dr

    if dr_to_use is None:
        return WindowResult(
            index=index,
            start_sec=start_sec,
            end_sec=end_sec,
            tdp_path=None,
            error=f"No daily DR covers DOYs {doys} (skip_merge mode)",
            elapsed_sec=0.0,
        )

    wr = process_window(
        dr_path=dr_to_use,
        stnm=stnm,
        index=index,
        start_sec=start_sec,
        end_sec=end_sec,
        jpl_products_dir=jpl_dir,
        tree_dir=tree_dir,
        stadb_path=stadb_path,
        otl_stadb_path=otl_stadb_path,
        work_dir=work_dir,
        output_tdp=output_tdp,
        gipsyx_env=gipsyx_env,
        timeout_sec=timeout_sec,
    )

    # Clean up mini-DR
    mini_dr = work_dir / f"minidr_{index:04d}.dr.gz"
    if mini_dr.exists():
        mini_dr.unlink(missing_ok=True)

    return wr


def process_station_rinex(
    stnm: str,
    cfg,  # gotl.config.Config — avoid circular import
    start_date: date | None = None,
    end_date: date | None = None,
    max_workers: int | None = None,
    force: bool = False,
) -> ProcessResult:
    """Full RINEX → TDP pipeline for one station.

    Steps:
        1. Discover RINEX files
        2. Create station database (rinex2StaDb.py)
        3. Convert RINEX → individual daily data records (parallel)
        4. Compute processing windows from daily DR time spans
        5. Per window: mini-merge 2-3 daily DRs → gd2e.py (parallel)
        6. Organize output as {tdp_root}/{stnm}_results/{NNNN}.tdp

    Args:
        stnm: 4-letter station code.
        cfg: Config object with rinex_root, tdp_root, jpl_products_dir, etc.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        max_workers: Max parallel workers (default: cfg.gipsy_max_workers).
        force: Re-process even if output exists.

    Returns:
        ProcessResult with status and output paths.
    """
    result = ProcessResult(stnm=stnm)
    workers = max_workers or cfg.gipsy_max_workers

    # Output directory
    out_dir = Path(cfg.tdp_root) / f"{stnm}_results"
    if out_dir.exists() and not force:
        existing = list(out_dir.glob("*.tdp"))
        if existing:
            log.info("TDP files already exist for %s (%d files), skipping. Use --force to reprocess.",
                     stnm, len(existing))
            result.tdp_dir = out_dir
            return result

    # Load GipsyX environment
    gipsyx_env = load_gipsyx_env(cfg.gipsyx_rc)

    # Step 1: Discover RINEX
    log.info("Step 1: Discovering RINEX files for %s", stnm)
    rinex_files = discover_rinex(cfg.rinex_root, stnm, start_date, end_date)
    result.n_rinex = len(rinex_files)
    log.info("Found %d RINEX files (%s to %s)",
             len(rinex_files), rinex_files[0].date, rinex_files[-1].date)

    # Check JPL products
    all_dates = [r.date for r in rinex_files]
    available, missing = check_products_available(cfg.jpl_products_dir, all_dates)
    if missing:
        log.warning("JPL products missing for %d dates (e.g. %s)",
                     len(missing), missing[0])

    with tempfile.TemporaryDirectory(prefix=f"gotl_{stnm}_") as tmpdir:
        work_dir = Path(tmpdir)

        # Step 2: Create station database
        log.info("Step 2: Creating station database")
        stadb_path = work_dir / f"{stnm}.stadb"
        rinex_paths = [r.path for r in rinex_files]
        create_stadb(rinex_paths, stnm, stadb_path, gipsyx_env)

        # Step 3: Convert RINEX → individual daily data records.
        # Optionally skip the final drMerge: per-window mini-DRs cover all
        # interior windows, and we use a single DR for edge windows where
        # only one daily DR overlaps. Saves ~5–10 min per station.
        rate_sec = getattr(cfg, "gipsy_rate_sec", 300)
        skip_merge = getattr(cfg, "gipsy_skip_full_merge", False)
        log.info("Step 3: Converting %d RINEX files to daily data records "
                 "(workers=%d, rate=%ds, skip_merge=%s)",
                 len(rinex_files), workers, rate_sec, skip_merge)
        dr_dir = work_dir / "dr"
        dr_dir.mkdir(parents=True, exist_ok=True)
        dr_path_placeholder = work_dir / f"{stnm}_final.dr.gz"
        merged_dr = build_data_record(
            rinex_paths, stadb_path, work_dir, dr_path_placeholder,
            gipsyx_env, max_workers=workers, rate_sec=rate_sec,
            merge=not skip_merge,
        )
        result.dr_path = merged_dr  # None when skip_merge=True

        # Step 4: Get time span. If we skipped the merge, derive it from the
        # first and last daily DR rather than calling dataRecordInfo on the
        # (non-existent) merged file.
        log.info("Step 4: Getting time span from data record")
        daily_drs = _build_daily_dr_map(dr_dir)
        if skip_merge:
            sorted_drs = [daily_drs[doy] for doy in sorted(daily_drs)]
            first_start, _ = get_dr_time_span(sorted_drs[0], gipsyx_env)
            _, last_end = get_dr_time_span(sorted_drs[-1], gipsyx_env)
            start_sec, end_sec = first_start, last_end
        else:
            start_sec, end_sec = get_dr_time_span(dr_path_placeholder, gipsyx_env)
        log.info("Time span: %.0f to %.0f GPS seconds", start_sec, end_sec)

        # Step 5: Create processing windows
        windows = compute_processing_windows(start_sec, end_sec)
        log.info("Step 5: %d processing windows", len(windows))

        # Step 6: Prepare tree directory
        log.info("Step 6: Preparing kinematicPPP.tree (rate=%ds)", rate_sec)
        tree_dir = prepare_tree_dir(work_dir, vmf1_dir=cfg.vmf1_dir, rate_sec=rate_sec)

        # Copy stadb for window processing
        win_stadb = work_dir / "stadb.db"
        shutil.copy2(stadb_path, win_stadb)

        # Resolve OTL file for GipsyX ocean loading correction
        otl_model = getattr(cfg, "gipsy_otl_model", "FES2014")
        otl_file = Path(cfg.otl_params_dir) / f"{stnm}_{otl_model}.otl"
        if not otl_file.exists():
            raise FileNotFoundError(
                f"OTL file not found: {otl_file}. "
                f"Generate it first: gotl gen-otl-params {stnm} --tide-model {otl_model}"
            )

        # Step 7: Process windows in parallel with per-window mini-DRs
        log.info("Step 7: Processing %d windows (workers=%d)", len(windows), workers)
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for idx, w_start, w_end in windows:
            tdp_out = out_dir / f"{idx:04d}.tdp"
            jpl_dir = cfg.jpl_products_dir
            tasks.append((
                stnm, idx, w_start, w_end,
                jpl_dir, tree_dir, win_stadb, otl_file,
                work_dir, tdp_out, gipsyx_env, 3600,
                daily_drs, merged_dr,
            ))

        window_results: list[WindowResult] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_window_minidr, t): t[1] for t in tasks}
            for future in as_completed(futures):
                wr = future.result()
                window_results.append(wr)
                if wr.error:
                    log.warning("Window %04d failed: %s", wr.index, wr.error)
                else:
                    log.info("Window %04d complete (%.1fs)", wr.index, wr.elapsed_sec)

        window_results.sort(key=lambda w: w.index)
        result.window_results = window_results
        result.n_windows_ok = sum(1 for w in window_results if w.error is None)
        result.n_windows_fail = sum(1 for w in window_results if w.error is not None)

        if result.n_windows_ok == 0:
            result.error = "No windows produced TDP output"
            log.error("All %d windows failed for %s", len(windows), stnm)

    result.tdp_dir = out_dir
    log.info(
        "Station %s: %d/%d windows succeeded, output in %s",
        stnm, result.n_windows_ok, len(windows), out_dir,
    )
    return result
