"""FastAPI application for the gotl web dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from gotl.web.deps import get_config, set_job_manager
from gotl.web.jobs.manager import JobManager

_WEB_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    mgr = JobManager(cfg.db_path, max_concurrent=cfg.max_concurrent_jobs)
    await mgr.init()
    set_job_manager(mgr)
    yield
    await mgr.shutdown()


app = FastAPI(title="gotl dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")

templates = Jinja2Templates(directory=_WEB_DIR / "templates")

# Register routes
from gotl.web.routes import dashboard, stations, compare, jobs  # noqa: E402
from gotl.web.api import stations as api_stations  # noqa: E402
from gotl.web.api import plots as api_plots  # noqa: E402
from gotl.web.api import jobs as api_jobs  # noqa: E402
from gotl.web.api import compare as api_compare  # noqa: E402

app.include_router(dashboard.router)
app.include_router(stations.router)
app.include_router(compare.router)
app.include_router(jobs.router)
app.include_router(api_stations.router, prefix="/api")
app.include_router(api_plots.router, prefix="/api")
app.include_router(api_jobs.router, prefix="/api")
app.include_router(api_compare.router, prefix="/api")
