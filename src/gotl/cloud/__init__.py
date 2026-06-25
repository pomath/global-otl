"""Cloud pipeline: distribute station processing via Coiled (Dask on AWS).

Usage:
    from gotl.cloud import run_cloud
    from gotl.cloud.config import CloudConfig

    results = run_cloud(cfg, cloud_cfg, stations=["brux", "aboa"])

For local testing without AWS costs:
    from gotl.cloud.local import run_local_test
    results = run_local_test(cfg, stations=["aboa"])
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def run_cloud(
    cfg,
    cloud_cfg,
    stations: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Run the gotl pipeline on Coiled cloud workers.

    Steps:
    1. Build gotl wheel and create Coiled software environment
    2. Upload per-station input files to S3
    3. Submit process_station_remote for each station
    4. Collect results and download .h5 files from S3
    5. Return summary dict

    Args:
        cfg: Local Config with input data paths and registry.
        cloud_cfg: CloudConfig with S3/Coiled settings.
        stations: Station codes to process (None = all in registry).
        dry_run: Stage data and show plan without launching workers.

    Returns:
        Dict mapping station code -> result dict.
    """
    import boto3
    import coiled

    from gotl.cloud.config import CloudConfig
    from gotl.cloud.s3 import (
        download_station_result,
        generate_run_id,
        upload_manifest,
        upload_station_inputs,
    )
    from gotl.cloud.worker import process_station_remote, serialize_meta
    from gotl.registry import load_registry

    registry = load_registry(cfg.registry_path)

    if stations is None:
        stations = sorted(registry.keys())
    else:
        unknown = [s for s in stations if s not in registry]
        if unknown:
            log.warning("Stations not in registry (skipping): %s", unknown)
        stations = [s for s in stations if s in registry]

    if not stations:
        raise ValueError("No valid stations to process")

    run_id = generate_run_id()
    s3_prefix = f"{cloud_cfg.s3_prefix}/{run_id}"
    s3_client = boto3.client("s3", region_name=cloud_cfg.aws_region)

    log.info("Run ID: %s", run_id)
    log.info("S3 path: s3://%s/%s/", cloud_cfg.s3_bucket, s3_prefix)
    log.info("Stations: %s", ", ".join(stations))

    # Stage input data to S3
    total_files = 0
    for stnm in stations:
        n = upload_station_inputs(
            stnm=stnm,
            tdp_root=cfg.tdp_root,
            hardisp_root=cfg.hardisp_root,
            otl_params_dir=cfg.otl_params_dir,
            tide_models=cloud_cfg.tide_models,
            s3_client=s3_client,
            s3_bucket=cloud_cfg.s3_bucket,
            s3_prefix=s3_prefix,
        )
        total_files += n
        log.info("Staged %s: %d files", stnm, n)

    upload_manifest(run_id, stations, cloud_cfg, s3_client, cloud_cfg.s3_bucket, s3_prefix)
    log.info("Total files staged: %d", total_files)

    if dry_run:
        log.info("Dry run — data staged to S3, no workers launched.")
        return {s: {"stnm": s, "status": "staged"} for s in stations}

    # Create the Coiled function (package sync handles gotl + deps automatically)
    @coiled.function(
        name="gotl-process-station",
        vm_type=cloud_cfg.worker_vm_type,
        spot_policy="spot_with_fallback" if cloud_cfg.use_spot else "on-demand",
        idle_timeout=cloud_cfg.idle_timeout,
        region=cloud_cfg.aws_region,
        n_workers=cloud_cfg.n_workers,
    )
    def remote_fn(args):
        stnm, meta_dict = args
        return process_station_remote(
            stnm=stnm,
            meta_dict=meta_dict,
            s3_bucket=cloud_cfg.s3_bucket,
            s3_prefix=s3_prefix,
            tide_models=cloud_cfg.tide_models,
            force=cloud_cfg.force,
        )

    # Serialize metadata and submit
    meta_dicts = {stnm: serialize_meta(registry[stnm]) for stnm in stations}
    inputs = [(stnm, meta_dicts[stnm]) for stnm in stations]

    log.info("Submitting %d stations to Coiled...", len(stations))
    raw_results = remote_fn.map(inputs)

    # Collect results and download .h5 files
    results = {}
    for result in raw_results:
        stnm = result["stnm"]
        results[stnm] = result
        if result["status"] == "ok":
            try:
                download_station_result(
                    stnm, cfg.results_dir, s3_client, cloud_cfg.s3_bucket, s3_prefix
                )
                log.info("%s: OK (%.1fs), result downloaded", stnm, result["elapsed_sec"])
            except Exception as e:
                log.warning("%s: OK but download failed: %s", stnm, e)
        else:
            log.warning("%s: FAILED — %s", stnm, result.get("error", "unknown"))

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    log.info("Completed: %d/%d stations succeeded", n_ok, len(stations))
    return results


def ensure_software_env(cloud_cfg) -> None:
    """Build gotl wheel and create/update the Coiled software environment."""
    import coiled

    # Build wheel
    project_root = Path(__file__).resolve().parents[3]  # src/gotl/cloud -> project root
    log.info("Building gotl wheel...")
    result = subprocess.run(
        [sys.executable, "-m", "poetry", "build", "-f", "wheel"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"poetry build failed: {result.stderr}")

    # Find the wheel
    dist_dir = project_root / "dist"
    wheels = sorted(dist_dir.glob("gotl-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise FileNotFoundError(f"No wheel found in {dist_dir}")
    wheel_path = wheels[-1]
    log.info("Built wheel: %s", wheel_path.name)

    # Create software environment
    log.info("Creating Coiled software environment: %s", cloud_cfg.software_env_name)
    coiled.create_software_environment(
        name=cloud_cfg.software_env_name,
        pip=[
            str(wheel_path),
            "boto3>=1.34",
        ],
    )
    log.info("Software environment ready.")
