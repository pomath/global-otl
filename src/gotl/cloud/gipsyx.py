"""GipsyX cloud packaging and remote execution.

Packages a minimal GipsyX installation for cloud workers and provides
the remote worker function for RINEX → TDP processing on Coiled.

GipsyX cloud archive layout (S3):
    s3://{bucket}/gipsyx/gipsyx-cloud.tar.gz
        GipsyX-2.0/bin/          — scripts + rtgx binary
        GipsyX-2.0/lib/          — shared libraries
        GipsyX-2.0/rc_GipsyX.sh  — environment setup
        GipsyX-2.0/share/        — misc support files
        venv39/                  — Python 3.9 venv (gcore module)
        goa-var/                 — trimmed support data (~600 MB)

    s3://{bucket}/gipsyx/jpl_gcore/{YYYY}/
        GNSS.pos, GNSS.eo, GNSS.tdp, ...  — GCORE-format products
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _configure_worker_logging() -> None:
    """Make gotl log messages visible on a Coiled worker.

    cli.py calls ``logging.basicConfig`` at import time, but cloud workers
    only import the worker functions (not the CLI), so they start with no
    handlers configured and ``log.info(...)`` writes nowhere. Coiled
    captures the worker's stderr into ``coiled cluster logs``, so wiring a
    StreamHandler to ``sys.stderr`` here surfaces per-station progress
    (Step 7, ``stnm: M/N windows in Xs``, etc.) in the cluster logs.

    Safe to call repeatedly — does nothing if a handler is already attached.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)

# Directories/files in goa-var that are required for GPS kinematic PPP.
# Everything else (antenna_cals_xmit, doris, slr, vlbi) is skipped.
_GOA_VAR_KEEP = [
    "etc/",  # all support files (ConstellationInfo, antenna cals, receiver types, etc.)
    "eph/",  # planetary ephemeris (de421.bin)
    "sta_info/",  # station reference frames (IGb14, IGS20)
    "time-pole/",  # LEAPSECS, IERS Bulletin A (iersa_all.final)
    "gds-products/",  # GDS product support files
]

# Subdirs of etc/ to SKIP (saves ~4.4 GB)
_GOA_VAR_ETC_SKIP = {"antenna_cals_xmit"}

# Directories in GipsyX-2.0 to skip (test data, verification)
_GIPSYX_SKIP = {"testData", "verify"}


def create_gipsyx_archive(
    gipsyx_root: Path,
    venv39_path: Path,
    goa_var_path: Path,
    output_path: Path,
    python39_root: Path | None = None,
) -> Path:
    """Create a minimal GipsyX tar.gz archive for cloud deployment.

    Args:
        gipsyx_root: Path to GipsyX-2.0/ directory.
        venv39_path: Path to Python 3.9 venv directory.
        goa_var_path: Path to goa-var/ directory.
        output_path: Where to write the .tar.gz file.
        python39_root: Path to standalone Python 3.9 installation (e.g. UV
            cpython-3.9). If provided, included so the venv works on cloud
            workers that don't have Python 3.9 installed.

    Returns:
        Path to the created archive.
    """
    log.info("Creating GipsyX cloud archive...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_path, "w:gz") as tar:
        # GipsyX-2.0 (bin, lib, share, rc_GipsyX.sh)
        gipsyx_root = Path(gipsyx_root)
        for item in sorted(gipsyx_root.iterdir()):
            if item.name in _GIPSYX_SKIP:
                continue
            arcname = f"GipsyX-2.0/{item.name}"
            log.debug("Adding %s", arcname)
            tar.add(str(item), arcname=arcname)

        # Python 3.9 venv
        log.debug("Adding venv39/")
        tar.add(str(venv39_path), arcname="venv39")

        # Standalone Python 3.9 (stdlib + binary)
        if python39_root:
            python39_root = Path(python39_root)
            if python39_root.is_dir():
                log.debug("Adding python39/ (standalone interpreter)")
                tar.add(str(python39_root), arcname="python39")

        # goa-var (trimmed — skip antenna_cals_xmit to save 4.4 GB)
        goa_var_path = Path(goa_var_path)
        for keep in _GOA_VAR_KEEP:
            src = goa_var_path / keep
            if not src.exists():
                log.warning("goa-var path not found, skipping: %s", src)
                continue
            arcname = f"goa-var/{keep}"
            if src.is_dir() and keep == "etc/":
                # Add etc/ but skip large subdirs
                for item in sorted(src.iterdir()):
                    if item.name in _GOA_VAR_ETC_SKIP:
                        log.debug("Skipping %s", item.name)
                        continue
                    tar.add(str(item), arcname=f"goa-var/etc/{item.name}")
            else:
                log.debug("Adding %s", arcname)
                tar.add(str(src), arcname=arcname)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Archive created: %s (%.0f MB)", output_path, size_mb)
    return output_path


def upload_gipsyx_archive(
    archive_path: Path,
    s3_bucket: str,
    s3_key: str = "gipsyx/gipsyx-cloud.tar.gz",
    aws_region: str = "us-east-1",
) -> str:
    """Upload the GipsyX archive to S3.

    Returns the full S3 URI.
    """
    import boto3

    s3_client = boto3.client("s3", region_name=aws_region)
    log.info("Uploading %s to s3://%s/%s ...", archive_path.name, s3_bucket, s3_key)
    s3_client.upload_file(str(archive_path), s3_bucket, s3_key)
    uri = f"s3://{s3_bucket}/{s3_key}"
    log.info("Upload complete: %s", uri)
    return uri


def upload_jpl_gcore(
    gcore_dir: Path,
    s3_bucket: str,
    year: int,
    s3_prefix: str = "gipsyx/jpl_gcore",
    aws_region: str = "us-east-1",
) -> int:
    """Upload GCORE-format JPL products to S3.

    Args:
        gcore_dir: Local directory with GNSS.pos, GNSS.eo, etc.
        s3_bucket: S3 bucket.
        year: Year label for the S3 path.
        s3_prefix: S3 key prefix.
        aws_region: AWS region.

    Returns:
        Number of files uploaded.
    """
    import boto3

    s3_client = boto3.client("s3", region_name=aws_region)
    gcore_dir = Path(gcore_dir)
    count = 0
    for f in sorted(gcore_dir.iterdir()):
        if f.is_file():
            key = f"{s3_prefix}/{year}/{f.name}"
            s3_client.upload_file(str(f), s3_bucket, key)
            count += 1
    log.info("Uploaded %d GCORE product files to s3://%s/%s/%d/",
             count, s3_bucket, s3_prefix, year)
    return count


def setup_gipsyx_on_worker(
    s3_bucket: str,
    s3_gipsyx_key: str = "gipsyx/gipsyx-cloud.tar.gz",
    install_dir: str = "/tmp/gipsyx",
) -> Path:
    """Download and unpack GipsyX on a cloud worker.

    Called at the start of each GipsyX worker task.
    Caches the install — only downloads if not already present.

    Returns:
        Path to the rc_GipsyX.sh file.
    """
    import boto3

    install = Path(install_dir)
    rc_path = install / "GipsyX-2.0" / "rc_GipsyX.sh"

    # Always clean install to ensure latest patches are applied
    if install.exists():
        shutil.rmtree(install)

    install.mkdir(parents=True, exist_ok=True)
    archive = install / "gipsyx-cloud.tar.gz"

    log.info("Downloading GipsyX archive from S3...")
    s3_client = boto3.client("s3")
    s3_client.download_file(s3_bucket, s3_gipsyx_key, str(archive))

    log.info("Unpacking GipsyX archive...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=str(install))

    archive.unlink()

    # Fix rc_GipsyX.sh to point at the install directory
    _patch_rc_gipsyx(rc_path, install)

    # Fix venv39 python symlinks (shebangs point to original build path)
    _fix_venv39(install)

    log.info("GipsyX installed at %s", install)
    return rc_path


def _patch_rc_gipsyx(rc_path: Path, install_dir: Path) -> None:
    """Rewrite rc_GipsyX.sh to use the cloud install paths."""
    text = rc_path.read_text()

    # Replace the original GIPSYX_HOME and GOA_VAR paths
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        # Replace GIPSYX_HOME assignment
        if "GIPSYX_HOME=" in line and not line.strip().startswith("#"):
            new_lines.append(f'export GIPSYX_HOME="{install_dir}/GipsyX-2.0"')
        elif "GOA_VAR=" in line and not line.strip().startswith("#"):
            new_lines.append(f'export GOA_VAR="{install_dir}/goa-var"')
        else:
            new_lines.append(line)

    # If a bundled python39 exists, add PYTHONHOME to the rc script
    # so it's always set when the script is sourced
    python39_dir = install_dir / "python39"
    if python39_dir.is_dir():
        new_lines.append(f'export PYTHONHOME="{python39_dir}"')

    rc_path.write_text("\n".join(new_lines) + "\n")


def _ensure_python39_on_worker(install_dir: Path) -> None:
    """Ensure Python 3.9 is available for GipsyX scripts on a cloud worker.

    Downloads a standalone relocatable Python 3.9 build from
    python-build-standalone (Gregory Szorc's project, used by UV/Rye).
    These builds are fully self-contained and work without PYTHONHOME.
    """
    venv_bin = install_dir / "venv39" / "bin"

    # Check if python3.9 is already usable
    py39_test = subprocess.run(
        [str(venv_bin / "python3"), "-c", "import sys; print(sys.version)"],
        capture_output=True, text=True, timeout=10,
    )
    if py39_test.returncode == 0 and "3.9" in py39_test.stdout:
        log.info("Python 3.9 already working in venv39")
        return

    log.info("Downloading standalone Python 3.9...")
    py39_dir = install_dir / "python39_standalone"
    py39_dir.mkdir(parents=True, exist_ok=True)

    # Download python-build-standalone release (relocatable build)
    url = "https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.9.20+20241016-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    archive = py39_dir / "python.tar.gz"

    import urllib.request
    urllib.request.urlretrieve(url, str(archive))
    log.info("Downloaded Python 3.9 standalone")

    # Extract — the archive contains python/ with bin/, lib/, etc.
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=str(py39_dir))
    archive.unlink()

    # The extracted dir is py39_dir/python/bin/python3.9
    py39_bin = py39_dir / "python" / "bin" / "python3.9"
    if not py39_bin.exists():
        log.error("Python 3.9 binary not found at %s", py39_bin)
        return

    # Recreate venv with this python, preserving site-packages (gcore)
    venv_dir = install_dir / "venv39"
    old_sp = venv_dir / "lib" / "python3.9" / "site-packages"

    sp_backup = install_dir / "_sp_backup"
    if old_sp.is_dir():
        shutil.copytree(str(old_sp), str(sp_backup))

    shutil.rmtree(str(venv_dir))
    r = subprocess.run(
        [str(py39_bin), "-m", "venv", str(venv_dir)],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        log.error("Failed to create venv: %s", r.stderr[:200])
        return

    # Restore site-packages
    new_sp = venv_dir / "lib" / "python3.9" / "site-packages"
    if sp_backup.is_dir():
        for item in sp_backup.iterdir():
            dst = new_sp / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dst), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dst))
        shutil.rmtree(str(sp_backup))

    log.info("Python 3.9 standalone installed and venv39 recreated")


def _fix_venv39(install_dir: Path) -> None:
    """Fix Python 3.9 venv to work at the cloud install path.

    The venv's python binary may be a symlink to a UV-managed Python 3.9
    that doesn't exist on the cloud worker. We fix this by:
    1. Pointing the venv's python at the bundled standalone python39/bin/python3.9
    2. Updating pyvenv.cfg home path
    """
    venv_dir = install_dir / "venv39"
    python39_dir = install_dir / "python39"

    if not venv_dir.exists():
        log.warning("venv39 not found at %s", venv_dir)
        return

    venv_bin = venv_dir / "bin"
    py39 = venv_bin / "python3.9"
    py3 = venv_bin / "python3"
    python = venv_bin / "python"

    # If bundled standalone python39 exists, link venv to it
    if python39_dir.exists():
        real_py39 = python39_dir / "bin" / "python3.9"
        if real_py39.exists():
            for p in [python, py3, py39]:
                if p.is_symlink() or p.exists():
                    p.unlink()
            py39.symlink_to(str(real_py39))
            py3.symlink_to("python3.9")
            python.symlink_to("python3.9")
            log.debug("Linked venv39 python to bundled python39")

            # Update pyvenv.cfg
            cfg_file = venv_dir / "pyvenv.cfg"
            if cfg_file.exists():
                text = cfg_file.read_text()
                lines = []
                for line in text.splitlines():
                    if line.startswith("home ="):
                        lines.append(f"home = {python39_dir / 'bin'}")
                    else:
                        lines.append(line)
                cfg_file.write_text("\n".join(lines) + "\n")
            return

    # Fallback: if python3.9 is already a real binary, just fix symlinks
    if py39.exists() and not py39.is_symlink():
        for p in [python, py3]:
            if p.is_symlink() or p.exists():
                p.unlink()
            p.symlink_to("python3.9")
        log.debug("Fixed venv39 python symlinks (real binary)")


def process_station_gipsyx_remote(
    stnm: str,
    meta_dict: dict,
    s3_bucket: str,
    s3_prefix: str,
    year: int,
    s3_gipsyx_key: str = "gipsyx/gipsyx-cloud.tar.gz",
    s3_jpl_prefix: str = "gipsyx/jpl_gcore",
    s3_rinex_prefix: str | None = None,
    max_workers: int = 8,
    otl_model: str = "FES2014",
) -> dict:
    """Run GipsyX PPP for one station on a Coiled worker.

    Downloads RINEX + OTL file from S3, sets up GipsyX, processes
    through PPP with OCEANLOAD == On, uploads TDP results to S3.

    Args:
        stnm: Station code.
        meta_dict: Serialized StationMeta.
        s3_bucket: S3 bucket name.
        s3_prefix: S3 prefix for this run's data.
        year: Processing year.
        s3_gipsyx_key: S3 key for the GipsyX archive.
        s3_jpl_prefix: S3 prefix for GCORE JPL products.
        max_workers: Parallel gd2e.py workers per station.
        otl_model: Tide model name; GipsyX consumes {stnm}_{otl_model}.otl
            staged at s3://.../inputs/{stnm}/otl/.

    Returns:
        Dict with status, n_windows_ok, n_windows_fail, elapsed_sec.
    """
    _configure_worker_logging()

    import boto3

    from gotl.cloud.s3 import download_station_inputs
    from gotl.cloud.worker import deserialize_meta
    from gotl.config import Config

    t0 = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix=f"gotl_gipsy_{stnm}_")
    tmp = Path(tmpdir)

    try:
        s3_client = boto3.client("s3")

        # Step 1: Install GipsyX
        rc_path = setup_gipsyx_on_worker(s3_bucket, s3_gipsyx_key)
        gipsyx_install = rc_path.parent.parent  # /tmp/gipsyx

        # Step 2: Download GCORE JPL products
        jpl_dir = tmp / "jpl_gcore"
        jpl_dir.mkdir()
        _download_s3_prefix(
            s3_client, s3_bucket,
            f"{s3_jpl_prefix}/{year}/",
            jpl_dir,
        )

        # Step 3: Download RINEX for this station.
        # Shared prefix (s3_rinex_prefix) is preferred — it dedups across
        # runs. Falls back to the legacy per-run path if not set.
        rinex_dir = tmp / "rinex"
        rinex_s3_key = (
            f"{s3_rinex_prefix}/{stnm}/"
            if s3_rinex_prefix
            else f"{s3_prefix}/inputs/{stnm}/rinex/{stnm}/"
        )
        _download_s3_prefix(
            s3_client, s3_bucket,
            rinex_s3_key,
            rinex_dir / stnm,
        )

        # Step 3b: Download OTL file for this station
        otl_params_dir = tmp / "otl_params"
        _download_s3_prefix(
            s3_client, s3_bucket,
            f"{s3_prefix}/inputs/{stnm}/otl/",
            otl_params_dir,
        )

        # Step 4: Install system Python 3.9 if not already available.
        # GipsyX scripts need Python 3.9 (gcore is compiled .pyc for 3.9).
        # The bundled python39 from UV has a hardcoded prefix that requires
        # PYTHONHOME, which is fragile. Instead, install a proper system Python 3.9.
        _ensure_python39_on_worker(gipsyx_install)

        from gotl.gipsy.batch import process_station_rinex

        tdp_root = tmp / "tdp"
        tdp_root.mkdir()

        cfg = Config(
            results_dir=tmp / "results",
            tdp_root=tdp_root,
            hardisp_root=tmp / "hardisp",
            otl_params_dir=otl_params_dir,
            rinex_root=rinex_dir,
            gipsyx_rc=rc_path,
            jpl_products_dir=jpl_dir,
            gipsy_max_workers=max_workers,
            gipsy_otl_model=otl_model,
        )

        result = process_station_rinex(stnm, cfg, max_workers=max_workers)

        # Step 5: Upload TDP results to S3
        tdp_out = tdp_root / f"{stnm}_results"
        if tdp_out.is_dir():
            for f in sorted(tdp_out.iterdir()):
                if f.suffix == ".tdp":
                    key = f"{s3_prefix}/tdp/{stnm}_results/{f.name}"
                    s3_client.upload_file(str(f), s3_bucket, key)

        elapsed = time.monotonic() - t0
        return {
            "stnm": stnm,
            "status": "ok" if result.n_windows_ok > 0 else "error",
            "n_windows_ok": result.n_windows_ok,
            "n_windows_fail": result.n_windows_fail,
            "n_rinex": result.n_rinex,
            "elapsed_sec": round(elapsed, 1),
            "error": result.error,
        }

    except Exception as e:
        elapsed = time.monotonic() - t0
        log.warning("%s: GipsyX failed after %.1f s: %s", stnm, elapsed, e)
        return {
            "stnm": stnm,
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(elapsed, 1),
            "n_windows_ok": 0,
            "n_windows_fail": 0,
            "n_rinex": 0,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def process_station_batch_remote(
    station_args: list[tuple[str, dict]],
    s3_bucket: str,
    s3_prefix: str,
    year: int,
    s3_gipsyx_key: str = "gipsyx/gipsyx-cloud.tar.gz",
    s3_jpl_prefix: str = "gipsyx/jpl_gcore",
    s3_rinex_prefix: str | None = None,
    max_workers: int = 8,
    otl_model: str = "FES2014",
    skip_full_merge: bool = False,
) -> list[dict]:
    """Process multiple stations on one worker, amortizing setup cost.

    GipsyX install + Python 3.9 + JPL products are downloaded once
    and reused across all stations in the batch.

    Args:
        station_args: List of (stnm, meta_dict) tuples.
        s3_bucket: S3 bucket name.
        s3_prefix: S3 prefix for this run's data.
        year: Processing year.
        s3_gipsyx_key: S3 key for the GipsyX archive.
        s3_jpl_prefix: S3 prefix for GCORE JPL products.
        max_workers: Parallel gd2e.py workers per station.

    Returns:
        List of result dicts, one per station.
    """
    _configure_worker_logging()

    import boto3

    from gotl.cloud.worker import deserialize_meta
    from gotl.config import Config

    t0 = time.monotonic()
    results = []

    try:
        s3_client = boto3.client("s3")

        # One-time setup (shared across all stations)
        rc_path = setup_gipsyx_on_worker(s3_bucket, s3_gipsyx_key)
        gipsyx_install = rc_path.parent.parent

        # Download JPL products (shared)
        jpl_dir = Path(tempfile.mkdtemp(prefix="gotl_jpl_"))
        _download_s3_prefix(s3_client, s3_bucket, f"{s3_jpl_prefix}/{year}/", jpl_dir)

        _ensure_python39_on_worker(gipsyx_install)

        from gotl.gipsy.batch import process_station_rinex

        log.info("Batch: processing %d stations (shared setup)", len(station_args))

        # Process each station sequentially, reusing GipsyX + products
        for stnm, meta_dict in station_args:
            stn_t0 = time.monotonic()

            # Skip if S3 already has TDPs from a prior attempt of this run.
            # Spot reclaim mid-batch causes the whole task to retry on a new
            # worker; without this check we'd redo every station the prior
            # worker already finished. We require ≥1 .tdp upload as the
            # "done" signal — partial uploads are rare since TDPs are pushed
            # in a tight loop after gd2e completes for a station.
            tdp_prefix = f"{s3_prefix}/tdp/{stnm}_results/"
            existing = s3_client.list_objects_v2(
                Bucket=s3_bucket, Prefix=tdp_prefix, MaxKeys=1,
            )
            if existing.get("KeyCount", 0) > 0:
                log.info("%s: TDPs already in S3 (%s), skipping",
                         stnm, tdp_prefix)
                results.append({
                    "stnm": stnm,
                    "status": "ok",
                    "n_windows_ok": -1,  # unknown — was uploaded by prior attempt
                    "n_windows_fail": 0,
                    "n_rinex": 0,
                    "elapsed_sec": 0.0,
                    "error": None,
                    "skipped": True,
                })
                continue

            tmpdir = tempfile.mkdtemp(prefix=f"gotl_gipsy_{stnm}_")
            tmp = Path(tmpdir)

            try:
                # Download RINEX for this station only.
                # Prefer shared s3_rinex_prefix when provided.
                rinex_dir = tmp / "rinex"
                rinex_s3_key = (
                    f"{s3_rinex_prefix}/{stnm}/"
                    if s3_rinex_prefix
                    else f"{s3_prefix}/inputs/{stnm}/rinex/{stnm}/"
                )
                _download_s3_prefix(
                    s3_client, s3_bucket,
                    rinex_s3_key,
                    rinex_dir / stnm,
                )

                # Download OTL file for this station
                otl_params_dir = tmp / "otl_params"
                _download_s3_prefix(
                    s3_client, s3_bucket,
                    f"{s3_prefix}/inputs/{stnm}/otl/",
                    otl_params_dir,
                )

                tdp_root = tmp / "tdp"
                tdp_root.mkdir()

                cfg = Config(
                    results_dir=tmp / "results",
                    tdp_root=tdp_root,
                    hardisp_root=tmp / "hardisp",
                    otl_params_dir=otl_params_dir,
                    rinex_root=rinex_dir,
                    gipsyx_rc=rc_path,
                    jpl_products_dir=jpl_dir,
                    gipsy_max_workers=max_workers,
                    gipsy_otl_model=otl_model,
                    gipsy_skip_full_merge=skip_full_merge,
                )

                result = process_station_rinex(stnm, cfg, max_workers=max_workers)

                # Upload TDP results
                tdp_out = tdp_root / f"{stnm}_results"
                if tdp_out.is_dir():
                    for f in sorted(tdp_out.iterdir()):
                        if f.suffix == ".tdp":
                            key = f"{s3_prefix}/tdp/{stnm}_results/{f.name}"
                            s3_client.upload_file(str(f), s3_bucket, key)

                elapsed = time.monotonic() - stn_t0
                results.append({
                    "stnm": stnm,
                    "status": "ok" if result.n_windows_ok > 0 else "error",
                    "n_windows_ok": result.n_windows_ok,
                    "n_windows_fail": result.n_windows_fail,
                    "n_rinex": result.n_rinex,
                    "elapsed_sec": round(elapsed, 1),
                    "error": result.error,
                })
                log.info("%s: %d/%d windows in %.0fs",
                         stnm, result.n_windows_ok,
                         result.n_windows_ok + result.n_windows_fail, elapsed)

            except Exception as e:
                elapsed = time.monotonic() - stn_t0
                results.append({
                    "stnm": stnm,
                    "status": "error",
                    "error": str(e),
                    "elapsed_sec": round(elapsed, 1),
                    "n_windows_ok": 0,
                    "n_windows_fail": 0,
                    "n_rinex": 0,
                })
                log.warning("%s: failed after %.0fs: %s", stnm, elapsed, e)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        # Setup failure — affects all stations
        elapsed = time.monotonic() - t0
        for stnm, _ in station_args:
            if not any(r["stnm"] == stnm for r in results):
                results.append({
                    "stnm": stnm,
                    "status": "error",
                    "error": f"Setup failed: {e}",
                    "elapsed_sec": round(elapsed, 1),
                    "n_windows_ok": 0,
                    "n_windows_fail": 0,
                    "n_rinex": 0,
                })
    finally:
        if "jpl_dir" in dir() and jpl_dir.exists():
            shutil.rmtree(str(jpl_dir), ignore_errors=True)

    return results


def _download_s3_prefix(
    s3_client, bucket: str, prefix: str, local_dir: Path,
    max_threads: int = 20,
) -> int:
    """Download all objects under an S3 prefix to a local directory.

    Uses parallel threads for speed (362 RINEX files: ~30s with 20 threads
    vs ~5 min sequential).
    """
    from concurrent.futures import ThreadPoolExecutor

    local_dir.mkdir(parents=True, exist_ok=True)

    # List all objects first
    paginator = s3_client.get_paginator("list_objects_v2")
    to_download: list[tuple[str, Path]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            local_path = local_dir / rel
            to_download.append((key, local_path))

    if not to_download:
        return 0

    # Create all parent directories upfront
    for _, local_path in to_download:
        local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download in parallel using thread pool (boto3 is thread-safe)
    import boto3

    def _download_one(args):
        key, local_path = args
        # Each thread gets its own S3 client to avoid connection issues
        client = boto3.client("s3")
        client.download_file(bucket, key, str(local_path))

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        list(pool.map(_download_one, to_download))

    return len(to_download)
