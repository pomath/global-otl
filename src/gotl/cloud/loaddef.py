"""LoadDef cloud worker for fleet OTL parameter generation.

Runs ``gotl gen-otl-params`` on Coiled workers using the bundled
``loaddef-cloud.tar.gz`` (LoadDef source + tide-model grids) plus the
existing ``gipsyx-cloud.tar.gz`` (which carries the OpenMPI 1.10 +
mpi4py-against-it venv that LoadDef needs).

Each batch of stations is processed serially on one worker; the worker
does its own setup (~5 min) once and then iterates the stations.
Resumable: stations whose ``.otl`` file already exists in S3 are skipped.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from gotl.cloud.gipsyx import (
    _configure_worker_logging,
    _ensure_python39_on_worker,
    setup_gipsyx_on_worker,
)

log = logging.getLogger(__name__)


def setup_loaddef_on_worker(
    s3_bucket: str,
    s3_loaddef_key: str = "gipsyx/loaddef-cloud.tar.gz",
    install_dir: str = "/tmp/loaddef",
) -> Path:
    """Download and unpack the LoadDef cloud bundle. Returns install root.

    Layout after extraction (matches build_loaddef_bundle.sh):
        {install}/repo/                  — gotl source + scripts + station_registry
        {install}/repo/data/loaddef/     — supporting files + TPXO9 grids
        {install}/repo/data/otl_params/  — empty output dir
        {install}/LoadDef-main/          — LoadDef Python source
    """
    import boto3

    install = Path(install_dir)
    if install.exists():
        shutil.rmtree(install)
    install.mkdir(parents=True, exist_ok=True)
    archive = install / "loaddef-cloud.tar.gz"

    log.info("Downloading LoadDef bundle from S3...")
    s3 = boto3.client("s3")
    s3.download_file(s3_bucket, s3_loaddef_key, str(archive))

    log.info("Unpacking LoadDef bundle...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=str(install))
    archive.unlink()

    log.info("LoadDef installed at %s", install)
    return install


def process_loaddef_batch_remote(
    stations: list[str],
    s3_bucket: str,
    s3_otl_prefix: str,
    tide_model: str = "TPXO9",
    earth_model: str = "PREM",
    mpi_np: int = 4,
    s3_gipsyx_key: str = "gipsyx/gipsyx-cloud.tar.gz",
    s3_loaddef_key: str = "gipsyx/loaddef-cloud.tar.gz",
) -> list[dict]:
    """Run gen-otl-params for a batch of stations on a Coiled worker.

    Args:
        stations: 4-letter station codes to process.
        s3_bucket: S3 bucket for input bundles + output .otl files.
        s3_otl_prefix: S3 prefix for output .otl files (e.g. 'otl_params/TPXO9').
        tide_model: any key of gotl.loaddef.batch.TIDE_MODEL_PREFIXES
            (the bundle must include that model's convgf_* grids).
        earth_model: 'PREM' (default) or other.
        mpi_np: MPI ranks per station's convolution. 4 fits ~20 GB peak;
            higher uses more RAM, possibly faster. r6i.4xlarge has 128 GB
            so mpi_np=8 is safe.
        s3_gipsyx_key: GipsyX bundle (provides venv39 + mpi4py + OpenMPI).
        s3_loaddef_key: LoadDef bundle (LoadDef-main + grids).

    Returns:
        One result dict per station with keys:
            stnm, status ('ok'|'error'|'skipped'), elapsed_sec, error
    """
    _configure_worker_logging()

    import boto3

    t0 = time.monotonic()
    results: list[dict] = []

    s3 = boto3.client("s3")

    # ── One-time setup ────────────────────────────────────────────────
    rc_path = setup_gipsyx_on_worker(s3_bucket, s3_gipsyx_key)
    gipsyx_install = rc_path.parent.parent  # /tmp/gipsyx
    _ensure_python39_on_worker(gipsyx_install)

    loaddef_install = setup_loaddef_on_worker(s3_bucket, s3_loaddef_key)
    repo_root = loaddef_install / "repo"
    loaddef_root = loaddef_install / "LoadDef-main"
    loaddef_data_dir = repo_root / "data" / "loaddef"
    otl_params_dir = repo_root / "data" / "otl_params"
    registry_path = repo_root / "data" / "station_registry.csv"

    # Install gotl from the bundled source so the CLI is on PATH.
    venv_py = gipsyx_install / "venv39" / "bin" / "python"
    if not venv_py.is_file():
        # Fallback: any python3 in install
        venv_py = Path("/usr/bin/python3")

    # The gotl package needs Python 3.11+; venv39 is Python 3.9. Use
    # the system python from the worker image (Coiled provides 3.13)
    # to run the CLI, but mpirun calls back into venv39 for LoadDef.
    import sys
    cli_python = sys.executable
    log.info("Worker setup complete in %.0fs (gotl CLI via %s, mpirun via %s)",
             time.monotonic() - t0, cli_python, venv_py)

    # gotl needs to be installed into the worker's Python; the bundle's
    # source tree is at repo_root. pip install -e it.
    subprocess.run(
        [cli_python, "-m", "pip", "install", "-e", str(repo_root), "--quiet"],
        check=True,
    )

    # ── Per-station loop ──────────────────────────────────────────────
    for stnm in stations:
        stn_t0 = time.monotonic()

        # Skip if already in S3 (resumable across reclaims)
        otl_key = f"{s3_otl_prefix}/{stnm}_{tide_model}.otl"
        existing = s3.list_objects_v2(
            Bucket=s3_bucket, Prefix=otl_key, MaxKeys=1
        )
        if existing.get("KeyCount", 0) > 0:
            log.info("%s: %s already in S3, skipping", stnm, otl_key)
            results.append({
                "stnm": stnm, "status": "skipped",
                "elapsed_sec": 0.0, "error": None,
            })
            continue

        try:
            env = {
                "GOTL_RESULTS_DIR": str(repo_root / "results"),
                "GOTL_OTL_PARAMS_DIR": str(otl_params_dir),
                "GOTL_REGISTRY": str(registry_path),
                "GOTL_LOADDEF_ROOT": str(loaddef_root),
                "GOTL_LOADDEF_DATA_DIR": str(loaddef_data_dir),
                "GOTL_GIPSYX_RC": str(rc_path),
                "PATH": f"{Path(cli_python).parent}:/usr/bin:/bin",
            }
            cmd = [
                cli_python, "-m", "gotl.cli", "gen-otl-params",
                "--stations", stnm,
                "--tide-model", tide_model,
                "--earth-model", earth_model,
                "--mpi-np", str(mpi_np),
            ]
            log.info("%s: running gen-otl-params (TPXO9, mpi_np=%d)", stnm, mpi_np)
            result = subprocess.run(
                cmd, env=env, cwd=str(repo_root),
                capture_output=True, text=True, timeout=3600,
            )
            elapsed = time.monotonic() - stn_t0

            otl_file = otl_params_dir / f"{stnm}_{tide_model}.otl"
            if result.returncode == 0 and otl_file.is_file():
                # Upload to S3
                s3.upload_file(str(otl_file), s3_bucket, otl_key)
                log.info("%s: ok in %.0fs (uploaded to s3://%s/%s)",
                         stnm, elapsed, s3_bucket, otl_key)
                results.append({
                    "stnm": stnm, "status": "ok",
                    "elapsed_sec": round(elapsed, 1), "error": None,
                })
            else:
                err = (result.stderr or "")[-500:]
                log.warning("%s: failed in %.0fs: %s", stnm, elapsed, err)
                results.append({
                    "stnm": stnm, "status": "error",
                    "elapsed_sec": round(elapsed, 1), "error": err,
                })
        except Exception as e:
            elapsed = time.monotonic() - stn_t0
            log.warning("%s: exception in %.0fs: %s", stnm, elapsed, e)
            results.append({
                "stnm": stnm, "status": "error",
                "elapsed_sec": round(elapsed, 1), "error": str(e),
            })

    return results
