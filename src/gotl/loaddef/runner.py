"""LoadDef subprocess execution via mpirun.

Creates temporary working directories that mirror LoadDef's expected
layout, generates MPI scripts from templates, and runs them via mpirun.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from gotl.loaddef.env import LoadDefEnv
from gotl.loaddef.templates import (
    CONVOLUTION_SCRIPT,
    GREENS_FUNCTIONS_SCRIPT,
    LOVE_NUMBERS_SCRIPT,
)

log = logging.getLogger(__name__)


@dataclass
class LoadDefResult:
    """Result of a LoadDef subprocess call."""

    step: str
    success: bool
    output_files: list[Path] = field(default_factory=list)
    elapsed_sec: float = 0.0
    error: str | None = None


def _prepare_workdir(
    tmpdir: Path,
    loaddef_root: Path,
) -> Path:
    """Create LoadDef-compatible directory structure.

    LoadDef expects to run from ``working/`` with ``../input/`` and
    ``../output/`` as siblings.  We symlink ``input/`` from the LoadDef
    source and create an empty ``output/`` tree.

    Returns:
        Path to the ``working/`` directory (use as cwd).
    """
    working = tmpdir / "working"
    working.mkdir(parents=True, exist_ok=True)
    output = tmpdir / "output"
    output.mkdir(exist_ok=True)

    # Symlink input from LoadDef installation
    input_link = tmpdir / "input"
    if not input_link.exists():
        input_link.symlink_to(loaddef_root / "input")

    return working


def _run_mpi_script(
    script: str,
    cwd: Path,
    env: LoadDefEnv,
    mpi_np: int,
    timeout_sec: int,
) -> tuple[int, str, str]:
    """Write script to temp file and execute via mpirun.

    Returns:
        (returncode, stdout, stderr)
    """
    script_path = cwd / "_loaddef_runner.py"
    script_path.write_text(script)

    cmd = [env.mpirun, "-np", str(mpi_np), env.python_exe, str(script_path)]
    log.debug("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env.subprocess_env,
        timeout=timeout_sec,
    )
    return result.returncode, result.stdout, result.stderr


def run_love_numbers(
    env: LoadDefEnv,
    planet_model: str,
    output_dir: Path,
    mpi_np: int = 4,
    timeout_sec: int = 600,
) -> LoadDefResult:
    """Compute Love Numbers from an Earth model.

    Args:
        env: LoadDef environment.
        planet_model: Earth model name (e.g. "PREM").  Resolved to
                      ``{data_dir}/Planet_Models/{planet_model}.txt``
                      or ``{loaddef_root}/input/Planet_Models/{planet_model}.txt``.
        output_dir: Where to copy output files.
        mpi_np: Number of MPI processes.
        timeout_sec: Subprocess timeout.

    Returns:
        LoadDefResult with output file paths.
    """
    t0 = time.monotonic()

    # Resolve planet model path
    pm_path = _resolve_planet_model(env, planet_model)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="loaddef_ln_") as tmpdir:
        tmpdir = Path(tmpdir)
        working = _prepare_workdir(tmpdir, env.loaddef_root)

        # Copy planet model to where LoadDef expects it
        pm_dest = tmpdir / "input" / "Planet_Models" / f"{planet_model}.txt"
        pm_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pm_path, pm_dest)

        script = LOVE_NUMBERS_SCRIPT.format(
            loaddef_root=env.loaddef_root,
            planet_model_path=str(pm_dest),
            output_dir=str(tmpdir / "output"),
            working_dir=str(working),
            earth_model=planet_model,
        )

        try:
            rc, stdout, stderr = _run_mpi_script(script, working, env, mpi_np, timeout_sec)
        except subprocess.TimeoutExpired:
            return LoadDefResult("love_numbers", False,
                                elapsed_sec=time.monotonic() - t0,
                                error=f"Timeout after {timeout_sec}s")

        if rc != 0:
            return LoadDefResult("love_numbers", False,
                                elapsed_sec=time.monotonic() - t0,
                                error=f"Exit {rc}: {stderr.strip()[:500]}")

        # Copy outputs
        outputs = []
        lln_dir = tmpdir / "output" / "Love_Numbers" / "LLN"
        if lln_dir.exists():
            for f in lln_dir.iterdir():
                dest = output_dir / "Love_Numbers" / "LLN" / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                outputs.append(dest)
                log.info("Love numbers output: %s", dest)

    return LoadDefResult("love_numbers", True, outputs,
                         elapsed_sec=time.monotonic() - t0)


def run_greens_functions(
    env: LoadDefEnv,
    lln_file: Path,
    earth_model: str,
    output_dir: Path,
    mpi_np: int = 4,
    timeout_sec: int = 600,
) -> LoadDefResult:
    """Compute Green's Functions from Love Numbers.

    Args:
        env: LoadDef environment.
        lln_file: Path to Load Love Number file (output from run_love_numbers).
        earth_model: Earth model name (for output filename).
        output_dir: Where to copy output files.
        mpi_np: Number of MPI processes.
        timeout_sec: Subprocess timeout.

    Returns:
        LoadDefResult with output file paths.
    """
    t0 = time.monotonic()
    lln_file = Path(lln_file).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not lln_file.exists():
        return LoadDefResult("greens_functions", False,
                             error=f"LLN file not found: {lln_file}")

    with tempfile.TemporaryDirectory(prefix="loaddef_gf_") as tmpdir:
        tmpdir = Path(tmpdir)
        working = _prepare_workdir(tmpdir, env.loaddef_root)

        # Copy LLN file to output tree (GF reads from ../output/Love_Numbers/LLN/)
        lln_dest = tmpdir / "output" / "Love_Numbers" / "LLN" / lln_file.name
        lln_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lln_file, lln_dest)

        # Create Greens_Functions output dir
        (tmpdir / "output" / "Greens_Functions").mkdir(parents=True, exist_ok=True)

        script = GREENS_FUNCTIONS_SCRIPT.format(
            loaddef_root=env.loaddef_root,
            lln_file=str(lln_dest),
            output_dir=str(tmpdir / "output"),
            working_dir=str(working),
            earth_model=earth_model,
        )

        try:
            rc, stdout, stderr = _run_mpi_script(script, working, env, mpi_np, timeout_sec)
        except subprocess.TimeoutExpired:
            return LoadDefResult("greens_functions", False,
                                elapsed_sec=time.monotonic() - t0,
                                error=f"Timeout after {timeout_sec}s")

        if rc != 0:
            return LoadDefResult("greens_functions", False,
                                elapsed_sec=time.monotonic() - t0,
                                error=f"Exit {rc}: {stderr.strip()[:500]}")

        # Copy outputs
        outputs = []
        gf_dir = tmpdir / "output" / "Greens_Functions"
        for f in gf_dir.iterdir():
            if f.is_file() and f.suffix == ".txt":
                dest = output_dir / "Greens_Functions" / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                outputs.append(dest)
                log.info("Greens function output: %s", dest)

    return LoadDefResult("greens_functions", True, outputs,
                         elapsed_sec=time.monotonic() - t0)


def run_convolution(
    env: LoadDefEnv,
    grn_file: Path,
    grid_dir: Path,
    grid_prefix: str,
    lsmask_file: Path,
    station_file: Path,
    output_dir: Path,
    rfm: str = "cm",
    earth_model: str = "PREM",
    mpi_np: int = 4,
    timeout_sec: int = 3600,
) -> LoadDefResult:
    """Run OTL convolution for stations.

    Args:
        env: LoadDef environment.
        grn_file: Green's function file (e.g. ce_PREM.txt).
        grid_dir: Directory with convolution grid NetCDF files.
        grid_prefix: Filename prefix (e.g. "convgf_FES2014").
        lsmask_file: Land-sea mask file.
        station_file: Station location file (lat lon name).
        output_dir: Where to write convolution output.
        rfm: Reference frame ("cm" or "cf").
        earth_model: Earth model name for output filenames.
        mpi_np: Number of MPI processes.
        timeout_sec: Subprocess timeout.

    Returns:
        LoadDefResult with output file paths.
    """
    t0 = time.monotonic()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, name in [(grn_file, "Green's function"), (lsmask_file, "land-sea mask"),
                        (station_file, "station file")]:
        if not Path(path).exists():
            return LoadDefResult("convolution", False, error=f"{name} not found: {path}")

    grid_dir = Path(grid_dir).resolve()
    if not grid_dir.is_dir():
        return LoadDefResult("convolution", False, error=f"Grid directory not found: {grid_dir}")

    with tempfile.TemporaryDirectory(prefix="loaddef_cn_") as tmpdir:
        tmpdir = Path(tmpdir)
        working = _prepare_workdir(tmpdir, env.loaddef_root)

        script = CONVOLUTION_SCRIPT.format(
            loaddef_root=env.loaddef_root,
            grn_file=str(Path(grn_file).resolve()),
            loadfile_directory=str(grid_dir),
            loadfile_prefix=grid_prefix,
            lsmask_file=str(Path(lsmask_file).resolve()),
            sta_file=str(Path(station_file).resolve()),
            rfm=rfm,
            earth_model=earth_model,
            output_dir=str(tmpdir / "output"),
            working_dir=str(working),
        )

        try:
            rc, stdout, stderr = _run_mpi_script(script, working, env, mpi_np, timeout_sec)
        except subprocess.TimeoutExpired:
            return LoadDefResult("convolution", False,
                                elapsed_sec=time.monotonic() - t0,
                                error=f"Timeout after {timeout_sec}s")

        log.debug("Convolution stdout: %s", stdout[:500])
        if rc != 0:
            return LoadDefResult("convolution", False,
                                elapsed_sec=time.monotonic() - t0,
                                error=f"Exit {rc}: {stderr.strip()[:500]}")

        # Copy convolution outputs (LoadDef writes to ../output/Convolution/ from cwd)
        conv_out_actual = tmpdir / "output" / "Convolution"
        outputs = []
        if conv_out_actual.exists():
            for f in conv_out_actual.iterdir():
                if f.is_file():
                    dest = output_dir / f.name
                    shutil.copy2(f, dest)
                    outputs.append(dest)
                    log.info("Convolution output: %s", dest)

    return LoadDefResult("convolution", True, outputs,
                         elapsed_sec=time.monotonic() - t0)


def _resolve_planet_model(env: LoadDefEnv, name: str) -> Path:
    """Find planet model file by name."""
    # Check data dir first
    for candidate in [
        env.data_dir / "Planet_Models" / f"{name}.txt",
        env.loaddef_root / "input" / "Planet_Models" / f"{name}.txt",
    ]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Planet model '{name}' not found in {env.data_dir}/Planet_Models/ "
        f"or {env.loaddef_root}/input/Planet_Models/"
    )
