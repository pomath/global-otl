#!/usr/bin/env python
"""Batch RINEX download for a station fleet.

Reads `data/rinex_availability_2024.csv` (or any file with the same
`stnm,archive,coverage,...` schema), selects stations classified as
full-coverage on a public archive, and runs parallel downloads via
``gotl.download.download_rinex_range``.

Each station's primary archive (from the CSV) is tried first, with the
other as fallback — same ordering discipline that ``download_rinex_range``
expects. Files land under ``{rinex_root}/{stnm}/`` and the existing
skip-existing check inside ``download_rinex_range`` makes this idempotent.

Usage::

    scripts/download/fleet_download.py                              # 2024 full
    scripts/download/fleet_download.py --jobs 4                     # less parallel
    scripts/download/fleet_download.py --stations brux,areq         # override
    scripts/download/fleet_download.py --include-partial            # broader net
    scripts/download/fleet_download.py --start 2023-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gotl.download import download_rinex_range  # noqa: E402

DEFAULT_CSV = _REPO_ROOT / "data" / "rinex_availability_2024.csv"
DEFAULT_ROOT = Path("/path/to/rinex")


def load_targets(csv_path: Path, include_partial: bool) -> list[tuple[str, str]]:
    """Return [(stnm, primary_archive)] for usable rows."""
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["archive"] == "none":
                continue
            if row["coverage"] != "full" and not include_partial:
                continue
            out.append((row["stnm"], row["archive"]))
    return out


def download_one(stnm: str, primary: str, start: date, end: date,
                 rinex_root: Path, dry_run: bool, log_dir: Path):
    """Run download_rinex_range for one station with primary-first archive order.

    Writes a per-station summary to ``log_dir/{stnm}.log`` so a fleet run
    leaves a triage trail without trying to capture every download's INFO
    chatter from the shared ``gotl.download`` logger.
    """
    order = [primary] + (["bkg"] if primary == "cddis" else ["cddis"])
    log_path = log_dir / f"{stnm}.log"

    t0 = time.monotonic()
    try:
        res = download_rinex_range(
            stnm=stnm, start_date=start, end_date=end,
            rinex_root=rinex_root, archives=order,
            skip_existing=True, max_retries=3, dry_run=dry_run,
        )
        elapsed = time.monotonic() - t0
        _write_station_log(log_path, stnm, order, res, elapsed, error=None)
        return stnm, res, elapsed, None
    except Exception as e:
        elapsed = time.monotonic() - t0
        _write_station_log(log_path, stnm, order, None, elapsed, error=str(e))
        return stnm, None, elapsed, str(e)


def _write_station_log(path: Path, stnm: str, archives: list[str],
                       res, elapsed: float, error: str | None) -> None:
    """Dump a one-screen summary per station: counts + per-date failures."""
    with open(path, "w") as f:
        f.write(f"station = {stnm}\n")
        f.write(f"archives = {','.join(archives)}\n")
        f.write(f"elapsed_s = {elapsed:.1f}\n")
        if error:
            f.write(f"error = {error}\n")
            return
        f.write(f"requested = {res.dates_requested}\n")
        f.write(f"downloaded = {res.dates_downloaded}\n")
        f.write(f"skipped = {res.dates_skipped}\n")
        f.write(f"failed = {res.dates_failed}\n")
        if res.failures:
            f.write("\nfailures:\n")
            for d, why in sorted(res.failures.items()):
                f.write(f"  {d}: {why}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--rinex-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--jobs", type=int, default=8,
                    help="Parallel stations in flight (default 8).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-partial", action="store_true",
                    help="Also pull stations classified as 'partial_*/4'.")
    ap.add_argument("--stations", default=None,
                    help="Comma-separated override list (filters the CSV rows).")
    ap.add_argument("--log-dir", type=Path, default=Path("/tmp/rinex_dl_logs"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("fleet_download")

    targets = load_targets(args.csv, args.include_partial)
    if args.stations:
        keep = {s.strip() for s in args.stations.split(",") if s.strip()}
        targets = [t for t in targets if t[0] in keep]
    if not targets:
        log.error("No target stations; nothing to do.")
        return 1

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    args.rinex_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    n_days = (end - start).days + 1
    log.info("Fleet: %d stations, %s..%s (%d days), jobs=%d, root=%s",
             len(targets), start, end, n_days, args.jobs, args.rinex_root)
    if args.dry_run:
        log.info("DRY RUN — no files will be written")

    totals = {"downloaded": 0, "skipped": 0, "failed": 0, "errors": 0}
    t0 = time.monotonic()
    done = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {
            ex.submit(download_one, stnm, arch, start, end,
                      args.rinex_root, args.dry_run, args.log_dir): (stnm, arch)
            for stnm, arch in targets
        }
        for fut in as_completed(futures):
            stnm, arch = futures[fut]
            done += 1
            stnm_, res, elapsed, err = fut.result()
            if err:
                totals["errors"] += 1
                log.warning("[%3d/%3d] %s  ERROR (%.1fs): %s",
                            done, len(targets), stnm_, elapsed, err)
                continue
            totals["downloaded"] += res.dates_downloaded
            totals["skipped"]    += res.dates_skipped
            totals["failed"]     += res.dates_failed
            log.info("[%3d/%3d] %s  dl=%d skip=%d fail=%d  (%.1fs, %s-first)",
                     done, len(targets), stnm_,
                     res.dates_downloaded, res.dates_skipped,
                     res.dates_failed, elapsed, arch)

    total_elapsed = time.monotonic() - t0
    log.info("")
    log.info("=== FLEET SUMMARY (%.1f min wall) ===", total_elapsed / 60)
    log.info("Stations: %d processed, %d errored",
             len(targets) - totals["errors"], totals["errors"])
    log.info("Files   : %d downloaded, %d skipped-existing, %d failed (per-day)",
             totals["downloaded"], totals["skipped"], totals["failed"])
    log.info("Per-station logs: %s", args.log_dir)
    return 0 if totals["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
