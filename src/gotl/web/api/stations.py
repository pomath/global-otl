"""Station API endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from gotl.io.cache import StationCache
from gotl.web.deps import get_config, get_registry

router = APIRouter(tags=["stations"])


@router.get("/stations")
async def list_stations():
    registry = get_registry()
    cfg = get_config()
    result = []
    for stnm in sorted(registry):
        meta = registry[stnm]
        cache = StationCache(Path(cfg.gotl.results_dir) / f"{stnm}.h5")
        result.append({
            "stnm": stnm,
            "network": meta.network,
            "lat": meta.lat,
            "lon": meta.lon,
            "steps": {
                s: cache.has(s)
                for s in ("tdp", "hardisp", "combined", "otl_coeff")
            },
        })
    return result


@router.get("/stations/{stnm}/status")
async def station_status(stnm: str):
    cfg = get_config()
    cache = StationCache(Path(cfg.gotl.results_dir) / f"{stnm}.h5")
    return {
        s: cache.has(s)
        for s in ("tdp", "hardisp", "combined", "otl_coeff")
    }
