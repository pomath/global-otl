"""Jobs page route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from gotl.web.app import templates

router = APIRouter()


@router.get("/jobs")
async def jobs_page(request: Request):
    return templates.TemplateResponse(request, "jobs.html")
