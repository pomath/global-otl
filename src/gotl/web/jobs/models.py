"""Pydantic models for job management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class JobCreate(BaseModel):
    command: str  # run, run-all, compare, gen-hardisp, gen-otl-params, process-rinex
    station: str | None = None
    options: dict[str, str] = {}


class Job(BaseModel):
    id: str
    command: str
    station: str | None = None
    options: dict[str, str] = {}
    status: JobStatus = JobStatus.queued
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    log_path: str | None = None
    pid: int | None = None
