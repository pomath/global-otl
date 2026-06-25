"""Compare page route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from gotl import CONSTITUENTS
from gotl.web.app import templates
from gotl.web.deps import get_config

router = APIRouter()


@router.get("/compare")
async def compare_page(request: Request):
    return templates.TemplateResponse(request, "compare.html")


@router.get("/compare/map")
async def compare_map_page(request: Request):
    cfg = get_config()
    return templates.TemplateResponse(request, "compare_map.html", {
        "constituents": CONSTITUENTS,
        "models": cfg.gotl.tide_models,
    })
