"""Batch runner: process all (or a subset of) stations, continuing past errors."""

from __future__ import annotations

import logging
import time

from gotl.config import Config
from gotl.pipeline import process_station
from gotl.registry import load_registry

log = logging.getLogger(__name__)


def run_all(
    cfg: Config,
    stations: list[str] | None = None,
    force: bool = False,
) -> dict[str, str | None]:
    """Run the full pipeline for each station.

    Args:
        cfg: Config with path references.
        stations: Station codes to process; defaults to all in registry.
        force: If True, re-run all steps even if cached.

    Returns:
        Dict mapping station code → None (success) or error message (failure).
    """
    registry = load_registry(cfg.registry_path)

    if stations is None:
        stations = sorted(registry.keys())
    else:
        unknown = [s for s in stations if s not in registry]
        if unknown:
            log.warning("Stations not in registry (skipping): %s", unknown)
        stations = [s for s in stations if s in registry]

    results: dict[str, str | None] = {}
    t0 = time.monotonic()

    for stnm in stations:
        log.info("=== %s ===", stnm)
        try:
            process_station(registry[stnm], cfg, force=force)
            results[stnm] = None
        except Exception as e:
            log.warning("Station %s failed: %s", stnm, e)
            results[stnm] = str(e)

    elapsed = time.monotonic() - t0
    n_ok = sum(v is None for v in results.values())
    log.info("Completed %d/%d stations in %.1f s", n_ok, len(stations), elapsed)
    return results
