"""Station list and detail page routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from gotl.io.cache import StationCache
from gotl.web.app import templates
from gotl.web.deps import get_config, get_registry

router = APIRouter()


@router.get("/stations")
async def station_list(request: Request):
    cfg = get_config()
    registry = get_registry()

    stations_info = []
    for stnm in sorted(registry):
        meta = registry[stnm]
        cache = StationCache(Path(cfg.gotl.results_dir) / f"{stnm}.h5")
        info = {
            "stnm": stnm,
            "network": meta.network,
            "lat": meta.lat,
            "lon": meta.lon,
            "has_tdp": cache.has("tdp"),
            "has_hardisp": cache.has("hardisp"),
            "has_combined": cache.has("combined"),
            "has_otl_coeff": cache.has("otl_coeff"),
            "quality": None,
        }
        if cache.has("combined"):
            try:
                attrs = cache.read_combined_attrs()
                n_total = attrs.get("n_total", 0)
                n_dropped = attrs.get("n_dropped", 0)
                if n_total > 0:
                    info["quality"] = {
                        "n_total": int(n_total),
                        "n_dropped": int(n_dropped),
                        "pct": 100.0 * n_dropped / n_total,
                    }
            except Exception:
                pass
        stations_info.append(info)

    return templates.TemplateResponse(request, "stations.html", {
        "stations": stations_info,
    })


@router.get("/station/{stnm}")
async def station_detail(request: Request, stnm: str):
    registry = get_registry()
    if stnm not in registry:
        raise HTTPException(status_code=404, detail=f"Station {stnm} not found")

    cfg = get_config()
    meta = registry[stnm]
    cache = StationCache(Path(cfg.gotl.results_dir) / f"{stnm}.h5")

    steps = {
        "tdp": cache.has("tdp"),
        "hardisp": cache.has("hardisp"),
        "combined": cache.has("combined"),
        "otl_coeff": cache.has("otl_coeff"),
    }

    quality = None
    if cache.has("combined"):
        try:
            attrs = cache.read_combined_attrs()
            n_total = attrs.get("n_total", 0)
            n_dropped = attrs.get("n_dropped", 0)
            if n_total > 0:
                quality = {
                    "n_total": int(n_total),
                    "n_dropped": int(n_dropped),
                    "pct": 100.0 * n_dropped / n_total,
                    "max_abs_u": attrs.get("max_abs_u"),
                    "max_abs_s": attrs.get("max_abs_s"),
                    "max_abs_w": attrs.get("max_abs_w"),
                }
        except Exception:
            pass

    return templates.TemplateResponse(request, "station_detail.html", {
        "stnm": stnm,
        "meta": meta,
        "steps": steps,
        "quality": quality,
    })
