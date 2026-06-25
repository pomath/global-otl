"""Remote worker function for Coiled cloud execution.

This module contains the function that runs on each Coiled worker.
It downloads station inputs from S3, runs the pipeline, and uploads results.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def serialize_meta(meta) -> dict:
    """Convert a StationMeta to a plain dict safe for cross-process transfer.

    pd.Timestamp objects are converted to ISO strings to avoid pickle issues.
    """
    return {
        "stnm": meta.stnm,
        "lat": meta.lat,
        "lon": meta.lon,
        "alt_m": meta.alt_m,
        "network": meta.network,
        "data_start": meta.data_start.isoformat() if meta.data_start else None,
        "data_end": meta.data_end.isoformat() if meta.data_end else None,
        "jump_epochs": [j.isoformat() for j in meta.jump_epochs],
    }


def deserialize_meta(d: dict):
    """Reconstruct a StationMeta from a plain dict."""
    from gotl.registry import StationMeta

    return StationMeta(
        stnm=d["stnm"],
        lat=d["lat"],
        lon=d["lon"],
        alt_m=d["alt_m"],
        network=d["network"],
        data_start=pd.Timestamp(d["data_start"]) if d["data_start"] else None,
        data_end=pd.Timestamp(d["data_end"]) if d["data_end"] else None,
        jump_epochs=[pd.Timestamp(j) for j in d["jump_epochs"]],
    )


def process_station_remote(
    stnm: str,
    meta_dict: dict,
    s3_bucket: str,
    s3_prefix: str,
    tide_models: list[str],
    force: bool = False,
) -> dict:
    """Run the gotl pipeline for one station on a remote worker.

    Downloads inputs from S3, processes via process_station(), uploads result.

    Args:
        stnm: Station code.
        meta_dict: Serialized StationMeta (from serialize_meta).
        s3_bucket: S3 bucket name.
        s3_prefix: S3 key prefix for this run (e.g. "cloud-runs/20260402-150000").
        tide_models: List of tide model names.
        force: Re-run even if cached.

    Returns:
        Dict with keys: stnm, status ("ok" or "error"), elapsed_sec, and
        optionally "error" (the error message on failure).
    """
    import boto3

    from gotl.cloud.s3 import download_station_inputs, upload_station_result
    from gotl.config import Config
    from gotl.pipeline import process_station

    t0 = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix=f"gotl_{stnm}_")

    try:
        tmppath = Path(tmpdir)
        s3_client = boto3.client("s3")

        # Download inputs from S3 into temp directory
        download_station_inputs(stnm, tmppath, s3_client, s3_bucket, s3_prefix)

        # Build Config pointing at temp directory layout
        results_dir = tmppath / "results"
        results_dir.mkdir(exist_ok=True)

        cfg = Config(
            results_dir=results_dir,
            tdp_root=tmppath / "tdp",
            hardisp_root=tmppath / "hardisp",
            otl_params_dir=tmppath / "otl_params",
            tide_models=tide_models,
        )

        # Reconstruct StationMeta
        meta = deserialize_meta(meta_dict)

        # Run pipeline
        process_station(meta, cfg, force=force)

        # Upload result .h5 to S3
        upload_station_result(stnm, results_dir, s3_client, s3_bucket, s3_prefix)

        elapsed = time.monotonic() - t0
        log.info("%s: completed in %.1f s", stnm, elapsed)
        return {"stnm": stnm, "status": "ok", "elapsed_sec": round(elapsed, 1)}

    except Exception as e:
        elapsed = time.monotonic() - t0
        log.warning("%s: failed after %.1f s: %s", stnm, elapsed, e)
        return {
            "stnm": stnm,
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(elapsed, 1),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
