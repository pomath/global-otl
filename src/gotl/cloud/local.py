"""Local test adapter for the cloud pipeline.

Validates the full cloud workflow (serialize, stage, process, collect) using
Dask LocalCluster and local filesystem instead of Coiled + S3. No AWS costs.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)


def run_local_test(
    cfg,
    stations: list[str],
    force: bool = False,
) -> dict[str, dict]:
    """Test the cloud workflow locally using Dask LocalCluster.

    Simulates the full cloud data flow:
    1. Serialize StationMeta for each station
    2. Stage input files to a temp "staging" directory (simulates S3 upload)
    3. Submit worker functions via Dask LocalCluster
    4. Workers read from staging, process, write results to staging
    5. Collect results back to cfg.results_dir

    Args:
        cfg: Local Config with paths to real input data.
        stations: Station codes to process.
        force: Re-run even if cached.

    Returns:
        Dict mapping station code -> result dict (status, elapsed_sec, etc.)
    """
    from dask.distributed import Client, LocalCluster

    from gotl.cloud.worker import deserialize_meta, serialize_meta
    from gotl.config import Config
    from gotl.pipeline import process_station
    from gotl.registry import load_registry

    registry = load_registry(cfg.registry_path)

    # Validate stations
    unknown = [s for s in stations if s not in registry]
    if unknown:
        log.warning("Stations not in registry (skipping): %s", unknown)
    stations = [s for s in stations if s in registry]
    if not stations:
        raise ValueError("No valid stations to process")

    staging_dir = tempfile.mkdtemp(prefix="gotl_local_staging_")
    staging = Path(staging_dir)

    try:
        # Stage input files (simulates S3 upload + download)
        log.info("Staging input files for %d stations...", len(stations))
        station_dirs = {}
        for stnm in stations:
            stn_staging = staging / stnm
            _stage_station_inputs(stnm, cfg, stn_staging)
            station_dirs[stnm] = stn_staging

        # Serialize station metadata
        meta_dicts = {stnm: serialize_meta(registry[stnm]) for stnm in stations}

        # Process via Dask LocalCluster
        log.info("Starting LocalCluster with 2 workers...")
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
        client = Client(cluster)

        try:
            futures = {}
            for stnm in stations:
                future = client.submit(
                    _local_worker,
                    stnm=stnm,
                    meta_dict=meta_dicts[stnm],
                    staging_dir=str(station_dirs[stnm]),
                    tide_models=cfg.tide_models,
                    force=force,
                )
                futures[stnm] = future

            # Collect results
            results = {}
            for stnm, future in futures.items():
                result = future.result()
                results[stnm] = result

                # Copy result .h5 back to local results dir
                if result["status"] == "ok":
                    src = Path(result["h5_path"])
                    dst = cfg.results_dir / f"{stnm}.h5"
                    if src.exists():
                        shutil.copy2(str(src), str(dst))
                        log.info("%s: result copied to %s", stnm, dst)

            return results
        finally:
            client.close()
            cluster.close()
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _stage_station_inputs(stnm: str, cfg, staging_dir: Path) -> None:
    """Copy one station's input files to a staging directory.

    Mirrors the S3 layout so the worker sees the same directory structure.
    """
    # TDP files
    src_tdp = cfg.tdp_root / f"{stnm}_results"
    dst_tdp = staging_dir / "tdp" / f"{stnm}_results"
    if src_tdp.is_dir():
        shutil.copytree(str(src_tdp), str(dst_tdp))
    else:
        log.warning("%s: TDP directory not found: %s", stnm, src_tdp)

    # Hardisp files
    src_hd = cfg.hardisp_root / stnm
    dst_hd = staging_dir / "hardisp" / stnm
    if src_hd.is_dir():
        shutil.copytree(str(src_hd), str(dst_hd))
    else:
        log.warning("%s: hardisp directory not found: %s", stnm, src_hd)

    # OTL params (all files matching this station)
    dst_otl = staging_dir / "otl_params"
    dst_otl.mkdir(parents=True, exist_ok=True)
    if cfg.otl_params_dir.is_dir():
        for f in cfg.otl_params_dir.iterdir():
            if f.name.startswith(f"{stnm}_") and f.suffix in (".otl", ".db"):
                shutil.copy2(str(f), str(dst_otl / f.name))


def _local_worker(
    stnm: str,
    meta_dict: dict,
    staging_dir: str,
    tide_models: list[str],
    force: bool = False,
) -> dict:
    """Worker function that runs in a Dask LocalCluster subprocess.

    Same logic as process_station_remote but reads from local staging
    directory instead of S3.
    """
    from gotl.cloud.worker import deserialize_meta
    from gotl.config import Config
    from gotl.pipeline import process_station

    t0 = time.monotonic()
    staging = Path(staging_dir)

    try:
        results_dir = staging / "results"
        results_dir.mkdir(exist_ok=True)

        cfg = Config(
            results_dir=results_dir,
            tdp_root=staging / "tdp",
            hardisp_root=staging / "hardisp",
            otl_params_dir=staging / "otl_params",
            tide_models=tide_models,
        )

        meta = deserialize_meta(meta_dict)
        process_station(meta, cfg, force=force)

        elapsed = time.monotonic() - t0
        h5_path = results_dir / f"{stnm}.h5"
        return {
            "stnm": stnm,
            "status": "ok",
            "elapsed_sec": round(elapsed, 1),
            "h5_path": str(h5_path),
        }

    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "stnm": stnm,
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(elapsed, 1),
            "h5_path": "",
        }
