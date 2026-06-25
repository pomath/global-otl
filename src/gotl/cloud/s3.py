"""S3 data staging for cloud pipeline runs.

Uploads per-station input files (TDP, hardisp, OTL params) to S3 before
distributing work to Coiled workers, and downloads result .h5 files afterward.

S3 layout:
    s3://{bucket}/{prefix}/{run_id}/
        inputs/{stnm}/tdp/{stnm}_results/*.tdp
        inputs/{stnm}/hardisp/{stnm}/{stnm}-hardisp_*.txt
        inputs/{stnm}/otl_params/{stnm}_*.otl
        results/{stnm}/{stnm}.h5
        manifest.json
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_TDP_PATTERN = re.compile(r"^\d+\.tdp$")


def generate_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def upload_station_inputs(
    stnm: str,
    tdp_root: Path,
    hardisp_root: Path,
    otl_params_dir: Path,
    tide_models: list[str],
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
) -> int:
    """Upload one station's input files to S3.

    Returns the number of files uploaded.
    """
    count = 0

    # TDP files: {tdp_root}/{stnm}_results/*.tdp
    tdp_dir = tdp_root / f"{stnm}_results"
    if tdp_dir.is_dir():
        for f in sorted(tdp_dir.iterdir()):
            if _TDP_PATTERN.match(f.name):
                key = f"{s3_prefix}/inputs/{stnm}/tdp/{stnm}_results/{f.name}"
                s3_client.upload_file(str(f), s3_bucket, key)
                count += 1
        log.debug("%s: uploaded %d TDP files", stnm, count)
    else:
        log.warning("%s: TDP directory not found: %s", stnm, tdp_dir)

    # Hardisp files: {hardisp_root}/{stnm}/{stnm}-hardisp_*.txt
    hardisp_dir = hardisp_root / stnm
    hd_count = 0
    if hardisp_dir.is_dir():
        for f in sorted(hardisp_dir.iterdir()):
            if f.name.startswith(f"{stnm}-hardisp_") and f.suffix == ".txt":
                key = f"{s3_prefix}/inputs/{stnm}/hardisp/{stnm}/{f.name}"
                s3_client.upload_file(str(f), s3_bucket, key)
                hd_count += 1
        log.debug("%s: uploaded %d hardisp files", stnm, hd_count)
    else:
        log.warning("%s: hardisp directory not found: %s", stnm, hardisp_dir)
    count += hd_count

    # OTL params: {otl_params_dir}/{stnm}_{model}.otl (and legacy {stnm}_otl.db)
    otl_count = 0
    if otl_params_dir.is_dir():
        for f in sorted(otl_params_dir.iterdir()):
            if f.name.startswith(f"{stnm}_") and f.suffix in (".otl", ".db"):
                key = f"{s3_prefix}/inputs/{stnm}/otl_params/{f.name}"
                s3_client.upload_file(str(f), s3_bucket, key)
                otl_count += 1
        log.debug("%s: uploaded %d OTL param files", stnm, otl_count)
    count += otl_count

    return count


def download_station_inputs(
    stnm: str,
    local_dir: Path,
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
) -> None:
    """Download one station's input files from S3 to a local directory.

    Recreates the directory structure expected by Config:
        {local_dir}/tdp/{stnm}_results/*.tdp
        {local_dir}/hardisp/{stnm}/*.txt
        {local_dir}/otl_params/*.otl
    """
    input_prefix = f"{s3_prefix}/inputs/{stnm}/"
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=s3_bucket, Prefix=input_prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            # Strip the input_prefix to get the relative path
            rel = s3_key[len(input_prefix):]
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(s3_bucket, s3_key, str(local_path))

    log.debug("%s: downloaded inputs to %s", stnm, local_dir)


def upload_station_result(
    stnm: str,
    results_dir: Path,
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
) -> None:
    """Upload the station's .h5 result file to S3."""
    h5_path = results_dir / f"{stnm}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Result file not found: {h5_path}")
    key = f"{s3_prefix}/results/{stnm}/{stnm}.h5"
    s3_client.upload_file(str(h5_path), s3_bucket, key)
    log.debug("%s: uploaded result to s3://%s/%s", stnm, s3_bucket, key)


def download_station_result(
    stnm: str,
    local_results_dir: Path,
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
) -> Path:
    """Download one station's .h5 result from S3.

    Returns the local path of the downloaded file.
    """
    key = f"{s3_prefix}/results/{stnm}/{stnm}.h5"
    local_path = local_results_dir / f"{stnm}.h5"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(s3_bucket, key, str(local_path))
    log.debug("%s: downloaded result to %s", stnm, local_path)
    return local_path


def upload_manifest(
    run_id: str,
    stations: list[str],
    cloud_cfg,
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
) -> None:
    """Write a manifest.json to S3 describing this run."""
    manifest = {
        "run_id": run_id,
        "stations": stations,
        "aws_region": cloud_cfg.aws_region,
        "worker_vm_type": cloud_cfg.worker_vm_type,
        "n_workers": cloud_cfg.n_workers,
        "use_spot": cloud_cfg.use_spot,
        "tide_models": cloud_cfg.tide_models,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    key = f"{s3_prefix}/manifest.json"
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )
