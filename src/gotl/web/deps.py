"""FastAPI dependency injection for shared state."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gotl.registry import StationMeta
    from gotl.web.config import WebConfig
    from gotl.web.jobs.manager import JobManager


_job_manager: JobManager | None = None


@lru_cache
def get_config() -> WebConfig:
    from gotl.web.config import WebConfig

    return WebConfig.from_env()


@lru_cache
def get_registry() -> dict[str, StationMeta]:
    from gotl.registry import load_registry

    cfg = get_config()
    return load_registry(cfg.gotl.registry_path)


def get_job_manager() -> JobManager:
    assert _job_manager is not None, "JobManager not initialised"
    return _job_manager


def set_job_manager(mgr: JobManager) -> None:
    global _job_manager
    _job_manager = mgr
