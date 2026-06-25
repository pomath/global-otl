"""Job management API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from gotl.web.app import templates
from gotl.web.deps import get_job_manager

router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_class=HTMLResponse)
async def create_job(
    request: Request,
    command: str = Form(...),
    station: str = Form(None),
    tide_model: str = Form(None),
    earth_model: str = Form(None),
    force: str = Form(None),
    years: str = Form(None),
    model: str = Form(None),
    start: str = Form(None),
    end: str = Form(None),
    max_workers: str = Form(None),
    stations: str = Form(None),
    archives: str = Form(None),
):
    mgr = get_job_manager()
    station = station.strip().lower() if station and station.strip() else None

    # Collect non-empty options into a dict
    options = {}
    if archives and archives.strip():
        options["archives"] = archives.strip()
    if tide_model and tide_model.strip():
        options["tide_model"] = tide_model.strip()
    if earth_model and earth_model.strip():
        options["earth_model"] = earth_model.strip()
    if force and force.strip():
        options["force"] = "true"
    if years and years.strip():
        options["years"] = years.strip()
    if model and model.strip():
        options["model"] = model.strip()
    if start and start.strip():
        options["start"] = start.strip()
    if end and end.strip():
        options["end"] = end.strip()
    if max_workers and max_workers.strip():
        options["max_workers"] = max_workers.strip()
    if stations and stations.strip():
        options["stations"] = stations.strip()

    job = await mgr.submit(command=command, station=station, options=options)
    desc = job.command
    if job.station:
        desc += f" {job.station}"
    if options.get("tide_model"):
        desc += f" ({options['tide_model']})"
    return (
        f'<p style="color:#2ecc71">Job <strong>{job.id}</strong> submitted '
        f'({desc}). <a href="/jobs">View jobs</a></p>'
    )


@router.get("/jobs", response_class=HTMLResponse)
async def list_jobs():
    mgr = get_job_manager()
    jobs = await mgr.list_jobs(limit=50)
    return _jobs_to_json(jobs)


@router.get("/jobs/table", response_class=HTMLResponse)
async def jobs_table(request: Request):
    mgr = get_job_manager()
    jobs = await mgr.list_jobs(limit=50)
    elapsed = _compute_elapsed(jobs)
    return templates.TemplateResponse(request, "_partials/job_table.html", {
        "jobs": jobs,
        "elapsed": elapsed,
    })


@router.get("/jobs/recent-table", response_class=HTMLResponse)
async def recent_jobs_table(request: Request):
    mgr = get_job_manager()
    jobs = await mgr.list_jobs(limit=10)
    elapsed = _compute_elapsed(jobs)
    return templates.TemplateResponse(request, "_partials/job_table.html", {
        "jobs": jobs,
        "elapsed": elapsed,
    })


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    mgr = get_job_manager()
    job = await mgr.get(job_id)
    if job is None:
        return {"error": "Job not found"}
    return job.model_dump(mode="json")


@router.get("/jobs/{job_id}/log", response_class=HTMLResponse)
async def get_job_log(job_id: str, tail: int = 200):
    mgr = get_job_manager()
    log_text = await mgr.get_log(job_id, tail=tail)
    if not log_text:
        return "<pre>No log output yet.</pre>"
    import html
    return f"<pre>{html.escape(log_text)}</pre>"


@router.delete("/jobs/{job_id}", response_class=HTMLResponse)
async def cancel_job(job_id: str):
    mgr = get_job_manager()
    job = await mgr.cancel(job_id)
    if job is None:
        return "<td colspan='7'>Job not found</td>"
    return f"<td colspan='7'>Job {job_id} cancelled</td>"


def _jobs_to_json(jobs):
    return [j.model_dump(mode="json") for j in jobs]


def _compute_elapsed(jobs) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    elapsed = {}
    for j in jobs:
        if j.finished_at and j.started_at:
            dt = j.finished_at - j.started_at
        elif j.started_at:
            dt = now - j.started_at
        else:
            dt = None
        if dt is not None:
            secs = int(dt.total_seconds())
            if secs < 60:
                elapsed[j.id] = f"{secs}s"
            elif secs < 3600:
                elapsed[j.id] = f"{secs // 60}m {secs % 60}s"
            else:
                elapsed[j.id] = f"{secs // 3600}h {(secs % 3600) // 60}m"
        else:
            elapsed[j.id] = "—"
    return elapsed
