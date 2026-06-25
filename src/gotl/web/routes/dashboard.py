"""Dashboard page route."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from gotl.io.cache import StationCache
from gotl.web.app import templates
from gotl.web.deps import get_config, get_registry

router = APIRouter()


@router.get("/")
async def dashboard(request: Request):
    cfg = get_config()
    registry = get_registry()

    complete = 0
    partial = 0
    unprocessed = 0

    for stnm in registry:
        cache = StationCache(Path(cfg.gotl.results_dir) / f"{stnm}.h5")
        steps_done = sum(
            cache.has(s) for s in ("tdp", "hardisp", "combined", "otl_coeff")
        )
        if steps_done == 4:
            complete += 1
        elif steps_done > 0:
            partial += 1
        else:
            unprocessed += 1

    return templates.TemplateResponse(request, "dashboard.html", {
        "total": len(registry),
        "complete": complete,
        "partial": partial,
        "unprocessed": unprocessed,
    })
