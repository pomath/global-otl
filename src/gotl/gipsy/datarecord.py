"""RINEX to data record conversion and merging.

Wraps GipsyX utilities:
    - ``rnxEditGde.py``: converts a single RINEX file to a data record (.dr.gz)
    - ``drMerge.py``: merges multiple data records into one

Reference: otl_proc/cmd_dir/mkdr_parallel.sh

Note: ``rnxEditGde.py`` writes temp files (``gde.tree``, ``{STNM}.gde.stats``,
``{STNM}.gde.debug.tree``) in its working directory. When running multiple
conversions in parallel, each must use a **separate working directory** to
avoid race conditions on these files.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)


def convert_rinex_to_dr(
    rinex_path: Path,
    stadb_path: Path,
    output_path: Path,
    gipsyx_env: dict[str, str],
    timeout_sec: int = 300,
    work_dir: Path | None = None,
    rate_sec: int | None = None,
) -> Path:
    """Convert one RINEX file to a GipsyX data record.

    Calls::

        rnxEditGde.py -staDb {stadb} -d {rinex} -out {output} [-rate RATE]

    Args:
        rinex_path: Path to RINEX observation file (can be compressed).
        stadb_path: Path to station database file.
        output_path: Where to write the .dr.gz file.
        gipsyx_env: Environment dict with GipsyX on PATH.
        timeout_sec: Subprocess timeout.
        work_dir: Working directory for rnxEditGde.py temp files.
            If None, a temporary directory is created and cleaned up.
            **Must be unique per concurrent call** to avoid race conditions
            on gde.tree and other temp files.
        rate_sec: Decimate the data record to this sampling interval (seconds).
            E.g. 300 for 5-min, 1800 for 30-min. If None, uses the RINEX
            file's native rate (typically 30s).

    Returns:
        Path to the created data record.

    Raises:
        RuntimeError: if rnxEditGde.py fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rnxEditGde.py",
        "-staDb", str(Path(stadb_path).resolve()),
        "-d", str(Path(rinex_path).resolve()),
        "-out", str(output_path.resolve()),
    ]
    if rate_sec is not None:
        cmd += ["-rate", str(rate_sec)]

    # Use a unique CWD to avoid race conditions when running in parallel.
    # rnxEditGde.py writes gde.tree, {STNM}.gde.stats, {STNM}.gde.debug.tree
    # in its CWD — concurrent workers sharing a CWD corrupt each other.
    cleanup_cwd = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="gotl_gde_"))
        cleanup_cwd = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=gipsyx_env,
            cwd=str(work_dir),
            timeout=timeout_sec,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout_tail = result.stdout.strip()[-500:] if result.stdout.strip() else ""
            raise RuntimeError(
                f"rnxEditGde.py failed for {rinex_path.name} (rc={result.returncode}): "
                f"{stderr[-500:]}"
                f"{f' | stdout: {stdout_tail}' if stdout_tail else ''}"
            )

        if not output_path.exists():
            stderr = result.stderr.strip()
            stdout_tail = result.stdout.strip()[-500:] if result.stdout.strip() else ""
            raise RuntimeError(
                f"rnxEditGde.py returned 0 but output not found: {output_path}"
                f"{f' | stderr: {stderr[-300:]}' if stderr else ''}"
                f"{f' | stdout: {stdout_tail}' if stdout_tail else ''}"
            )

        return output_path

    finally:
        if cleanup_cwd:
            shutil.rmtree(work_dir, ignore_errors=True)


def merge_data_records(
    dr_files: list[Path],
    output_path: Path,
    gipsyx_env: dict[str, str],
    batch_size: int = 199,
    timeout_sec: int = 600,
) -> Path:
    """Merge multiple data records into one.

    Calls::

        drMerge.py -inFiles {files...} -outFile {output}

    Handles more than ``batch_size`` files via a two-pass merge
    (matches mkdr_parallel.sh approach).

    Args:
        dr_files: List of .dr.gz file paths to merge.
        output_path: Where to write the merged data record.
        gipsyx_env: Environment dict with GipsyX on PATH.
        batch_size: Max files per drMerge.py call.
        timeout_sec: Subprocess timeout per merge call.

    Returns:
        Path to the merged data record.

    Raises:
        RuntimeError: if drMerge.py fails.
    """
    if len(dr_files) == 1:
        shutil.copy2(dr_files[0], output_path)
        return output_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(dr_files) <= batch_size:
        _run_drmerge(dr_files, output_path, gipsyx_env, timeout_sec)
        return output_path

    # Two-pass merge for large file sets
    inter_dir = output_path.parent / "interdr"
    inter_dir.mkdir(parents=True, exist_ok=True)
    intermediate: list[Path] = []

    for batch_idx in range(0, len(dr_files), batch_size):
        batch = dr_files[batch_idx : batch_idx + batch_size]
        inter_path = inter_dir / f"batch_{batch_idx:04d}.dr.gz"
        _run_drmerge(batch, inter_path, gipsyx_env, timeout_sec)
        intermediate.append(inter_path)

    _run_drmerge(intermediate, output_path, gipsyx_env, timeout_sec)
    return output_path


def _run_drmerge(
    in_files: list[Path],
    out_file: Path,
    gipsyx_env: dict[str, str],
    timeout_sec: int,
) -> None:
    """Run a single drMerge.py call."""
    cmd = [
        "drMerge.py",
        "-inFiles",
    ] + [str(f) for f in in_files] + [
        "-outFile", str(out_file),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=gipsyx_env,
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"drMerge.py failed: {result.stderr.strip()}"
        )


def _convert_one(args: tuple) -> tuple[Path, Path | None, str | None]:
    """Worker function for parallel RINEX→DR conversion.

    Each worker creates its own temporary working directory to avoid
    race conditions on gde.tree and other temp files.

    Args:
        args: (rinex_path, stadb_path, output_path, gipsyx_env, timeout_sec, rate_sec)

    Returns:
        (rinex_path, dr_path or None, error or None)
    """
    rinex_path, stadb_path, output_path, gipsyx_env, timeout_sec, rate_sec = args
    try:
        dr = convert_rinex_to_dr(
            rinex_path, stadb_path, output_path, gipsyx_env, timeout_sec,
            rate_sec=rate_sec,
        )
        return rinex_path, dr, None
    except Exception as e:
        return rinex_path, None, str(e)


def build_data_record(
    rinex_paths: list[Path],
    stadb_path: Path,
    work_dir: Path,
    output_path: Path,
    gipsyx_env: dict[str, str],
    max_workers: int = 4,
    rate_sec: int | None = None,
    merge: bool = True,
) -> Path | None:
    """Full RINEX → data record pipeline: convert each file, then merge.

    Replicates mkdr_parallel.sh: parallel rnxEditGde.py + drMerge.py.

    Args:
        rinex_paths: List of RINEX file paths (sorted by date).
        stadb_path: Station database file.
        work_dir: Temporary working directory for intermediate files.
        output_path: Where to write the final merged .dr.gz.
        gipsyx_env: Environment dict with GipsyX on PATH.
        max_workers: Max parallel rnxEditGde.py processes.
        rate_sec: Decimate data records to this sampling rate (seconds).
            E.g. 300 for 5-min, 1800 for 30-min. If None, native rate.
        merge: If False, skip the final drMerge step. Daily DRs are still
            written to ``work_dir/dr/``. Use when downstream callers only
            need per-window mini-merges (no full-year fallback DR).

    Returns:
        Path to the final merged data record, or None if ``merge=False``.

    Raises:
        RuntimeError: if all conversions fail.
    """
    dr_dir = Path(work_dir) / "dr"
    dr_dir.mkdir(parents=True, exist_ok=True)

    # Prepare conversion tasks
    tasks = []
    for i, rnx in enumerate(rinex_paths):
        dr_out = dr_dir / f"{rnx.stem}.dr.gz"
        tasks.append((rnx, stadb_path, dr_out, gipsyx_env, 300, rate_sec))

    # Convert in parallel
    dr_files: list[Path] = []
    errors: list[str] = []

    log.info("Converting %d RINEX files to data records (workers=%d)",
             len(tasks), max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_convert_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            rnx_path, dr_path, error = future.result()
            if dr_path is not None:
                dr_files.append(dr_path)
            else:
                errors.append(f"{rnx_path.name}: {error}")
                log.warning("Failed to convert %s: %s", rnx_path.name, error)

    if not dr_files:
        raise RuntimeError(
            f"All {len(tasks)} RINEX conversions failed. "
            f"First error: {errors[0] if errors else 'unknown'}"
        )

    if errors:
        log.warning("%d/%d RINEX conversions failed", len(errors), len(tasks))

    # Sort by name to maintain chronological order
    dr_files.sort(key=lambda p: p.name)

    if not merge:
        log.info("Skipping full drMerge (merge=False); %d daily DRs ready",
                 len(dr_files))
        return None

    # Merge
    log.info("Merging %d data records into %s", len(dr_files), output_path)
    return merge_data_records(dr_files, output_path, gipsyx_env)
