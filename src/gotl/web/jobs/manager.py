"""Async job manager backed by SQLite."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from gotl.web.jobs.models import Job, JobStatus

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    command     TEXT NOT NULL,
    station     TEXT,
    options     TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'queued',
    created_at  TEXT,
    started_at  TEXT,
    finished_at TEXT,
    exit_code   INTEGER,
    log_path    TEXT,
    pid         INTEGER
);
"""


class JobManager:
    """Manages gotl CLI jobs via async subprocesses + SQLite."""

    def __init__(self, db_path: Path, max_concurrent: int = 2) -> None:
        self.db_path = db_path
        self.max_concurrent = max_concurrent
        self._db: aiosqlite.Connection | None = None
        self._runner_task: asyncio.Task | None = None
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        # Mark stale running jobs as failed
        await self._db.execute(
            "UPDATE jobs SET status = 'failed', finished_at = ? "
            "WHERE status = 'running'",
            (_now_iso(),),
        )
        await self._db.commit()
        self._runner_task = asyncio.create_task(self._runner_loop())

    async def shutdown(self) -> None:
        if self._runner_task:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
        for job_id, proc in list(self._processes.items()):
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                proc.kill()
        if self._db:
            await self._db.close()

    async def submit(self, command: str, station: str | None = None,
                     options: dict[str, str] | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        opts = options or {}
        now = _now_iso()

        cfg = self._get_gotl_config()
        log_dir = Path(cfg.results_dir) / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"{job_id}.log")

        await self._db.execute(
            "INSERT INTO jobs (id, command, station, options, status, created_at, log_path) "
            "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (job_id, command, station, json.dumps(opts), now, log_path),
        )
        await self._db.commit()
        return await self.get(job_id)

    async def get(self, job_id: str) -> Job | None:
        async with self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_job(row)

    async def list_jobs(self, limit: int = 50) -> list[Job]:
        async with self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_job(r) for r in rows]

    async def cancel(self, job_id: str) -> Job | None:
        job = await self.get(job_id)
        if job is None:
            return None
        if job.status == JobStatus.running and job_id in self._processes:
            proc = self._processes[job_id]
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                proc.kill()
        await self._db.execute(
            "UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE id = ? AND status IN ('queued', 'running')",
            (_now_iso(), job_id),
        )
        await self._db.commit()
        return await self.get(job_id)

    async def get_log(self, job_id: str, tail: int = 200) -> str:
        job = await self.get(job_id)
        if job is None or job.log_path is None:
            return ""
        try:
            with open(job.log_path) as f:
                lines = f.readlines()
            return "".join(lines[-tail:])
        except FileNotFoundError:
            return ""

    def _get_gotl_config(self):
        from gotl.web.deps import get_config
        return get_config().gotl

    @staticmethod
    def _gotl_exe() -> str:
        """Return absolute path to the gotl entry-point script."""
        return str(Path(sys.executable).parent / "gotl")

    def _build_command(self, job: Job) -> list[str]:
        cfg = self._get_gotl_config()
        base = [
            self._gotl_exe(),
            "--results-dir", str(cfg.results_dir),
            "--tdp-root", str(cfg.tdp_root),
            "--hardisp-root", str(cfg.hardisp_root),
            "--otl-params-dir", str(cfg.otl_params_dir),
            "--registry", str(cfg.registry_path),
            "--hardisp-exe", str(cfg.hardisp_exe),
            "--tide-models", ",".join(cfg.tide_models),
        ]
        # GipsyX global flags (needed by process-rinex)
        if cfg.rinex_root:
            base += ["--rinex-root", str(cfg.rinex_root)]
        if cfg.gipsyx_rc:
            base += ["--gipsyx-rc", str(cfg.gipsyx_rc)]
        if cfg.jpl_products_dir:
            base += ["--jpl-products", str(cfg.jpl_products_dir)]
        if cfg.vmf1_dir:
            base += ["--vmf1-dir", str(cfg.vmf1_dir)]
        base += ["--gipsy-max-workers", str(cfg.gipsy_max_workers)]

        cmd = job.command
        opts = job.options

        if cmd == "run":
            return base + ["run", job.station] + (["--force"] if opts.get("force") else [])
        elif cmd == "run-all":
            parts = base + ["run-all"]
            if opts.get("force"):
                parts.append("--force")
            if opts.get("stations"):
                parts += ["--stations", opts["stations"]]
            return parts
        elif cmd == "compare":
            return base + ["compare", job.station]
        elif cmd == "gen-hardisp":
            parts = base + ["gen-hardisp", job.station]
            if opts.get("years"):
                parts += ["--years", opts["years"]]
            if opts.get("model"):
                parts += ["--model", opts["model"]]
            return parts
        elif cmd == "gen-otl-params":
            parts = base + ["gen-otl-params"]
            if job.station:
                parts.append(job.station)
            # LoadDef flags are on the command, not global
            if cfg.loaddef_root:
                parts += ["--loaddef-root", str(cfg.loaddef_root)]
            if cfg.loaddef_data_dir:
                parts += ["--loaddef-data-dir", str(cfg.loaddef_data_dir)]
            if opts.get("tide_model"):
                parts += ["--tide-model", opts["tide_model"]]
            if opts.get("earth_model"):
                parts += ["--earth-model", opts["earth_model"]]
            if opts.get("force"):
                parts.append("--force")
            if opts.get("all"):
                parts.append("--all")
            return parts
        elif cmd == "process-rinex":
            parts = base + ["process-rinex", job.station]
            if opts.get("start"):
                parts += ["--start", opts["start"]]
            if opts.get("end"):
                parts += ["--end", opts["end"]]
            if opts.get("force"):
                parts.append("--force")
            if opts.get("max_workers"):
                parts += ["--max-workers", opts["max_workers"]]
            return parts
        elif cmd == "download-rinex":
            parts = base + ["download-rinex", job.station]
            if opts.get("start"):
                parts += ["--start", opts["start"]]
            if opts.get("end"):
                parts += ["--end", opts["end"]]
            if opts.get("archives"):
                parts += ["--archives", opts["archives"]]
            if opts.get("force"):
                parts.append("--force")
            return parts
        else:
            return base + [cmd] + ([job.station] if job.station else [])

    async def _runner_loop(self) -> None:
        while True:
            try:
                await self._start_queued_jobs()
            except Exception:
                log.exception("Error in job runner loop")
            await asyncio.sleep(3)

    async def _start_queued_jobs(self) -> None:
        running_count = 0
        async with self._db.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'running'"
        ) as cur:
            row = await cur.fetchone()
            running_count = row[0]

        slots = self.max_concurrent - running_count
        if slots <= 0:
            return

        async with self._db.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT ?",
            (slots,),
        ) as cur:
            rows = await cur.fetchall()

        for row in rows:
            job = _row_to_job(row)
            await self._start_job(job)

    async def _start_job(self, job: Job) -> None:
        cmd = self._build_command(job)
        log.info("Starting job %s: %s", job.id, " ".join(cmd))

        log_file = open(job.log_path, "w")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ},
            )
        except Exception as exc:
            log_file.write(f"Failed to start process: {exc}\n")
            log_file.close()
            log.error("Job %s failed to spawn: %s", job.id, exc)
            await self._db.execute(
                "UPDATE jobs SET status = 'failed', started_at = ?, finished_at = ?, exit_code = -1 WHERE id = ?",
                (_now_iso(), _now_iso(), job.id),
            )
            await self._db.commit()
            return
        self._processes[job.id] = proc

        await self._db.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, pid = ? WHERE id = ?",
            (_now_iso(), proc.pid, job.id),
        )
        await self._db.commit()

        asyncio.create_task(self._wait_job(job.id, proc, log_file))

    async def _wait_job(self, job_id: str, proc: asyncio.subprocess.Process,
                        log_file) -> None:
        try:
            await proc.wait()
        finally:
            log_file.close()
            self._processes.pop(job_id, None)

        status = "done" if proc.returncode == 0 else "failed"
        await self._db.execute(
            "UPDATE jobs SET status = ?, finished_at = ?, exit_code = ? WHERE id = ?",
            (status, _now_iso(), proc.returncode, job_id),
        )
        await self._db.commit()
        log.info("Job %s finished: %s (exit %s)", job_id, status, proc.returncode)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_job(row) -> Job:
    return Job(
        id=row["id"],
        command=row["command"],
        station=row["station"],
        options=json.loads(row["options"]) if row["options"] else {},
        status=JobStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        exit_code=row["exit_code"],
        log_path=row["log_path"],
        pid=row["pid"],
    )
