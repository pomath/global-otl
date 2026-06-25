"""LoadDef environment setup.

Validates LoadDef installation, locates MPI runtime, and builds
the environment dict for subprocess calls.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class LoadDefEnv:
    """Validated LoadDef environment for subprocess calls."""

    loaddef_root: Path
    data_dir: Path
    mpirun: str
    python_exe: str
    subprocess_env: dict[str, str]


def load_loaddef_env(
    loaddef_root: Path,
    data_dir: Path,
    mpirun: str | None = None,
    python_exe: str | None = None,
) -> LoadDefEnv:
    """Validate LoadDef installation and build environment.

    Args:
        loaddef_root: Path to LoadDef-main/ directory.
        data_dir: Path to data/loaddef/ with pre-computed data.
        mpirun: Path to mpirun. Auto-detected if None.
        python_exe: Python with mpi4py/numpy/scipy. Auto-detected if None.

    Returns:
        LoadDefEnv with validated paths and subprocess env.

    Raises:
        FileNotFoundError: if LoadDef or required data missing.
        RuntimeError: if mpirun or python not found.
    """
    loaddef_root = Path(loaddef_root).resolve()
    data_dir = Path(data_dir).resolve()

    # Validate LoadDef source
    for subdir in ("CONVGF", "LOADGF"):
        if not (loaddef_root / subdir).is_dir():
            raise FileNotFoundError(
                f"LoadDef module {subdir}/ not found in {loaddef_root}"
            )

    # Validate data directory
    if not data_dir.is_dir():
        raise FileNotFoundError(f"LoadDef data directory not found: {data_dir}")

    # Find mpirun
    if mpirun is None:
        mpirun = _find_mpirun()
    elif not shutil.which(mpirun):
        raise RuntimeError(f"mpirun not found: {mpirun}")

    # Find python with mpi4py
    if python_exe is None:
        python_exe = _find_loaddef_python()

    # Build subprocess environment
    env = dict(os.environ)
    # Add LoadDef to PYTHONPATH
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{loaddef_root}:{existing_pp}" if existing_pp else str(loaddef_root)

    # Ensure MPI libraries are on LD_LIBRARY_PATH
    mpirun_dir = Path(mpirun).resolve().parent
    mpi_lib = mpirun_dir.parent / "lib"
    if mpi_lib.is_dir():
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{mpi_lib}:{existing_ld}" if existing_ld else str(mpi_lib)
        env["OPAL_PREFIX"] = str(mpi_lib.parent)

    log.info("LoadDef root: %s", loaddef_root)
    log.info("LoadDef data: %s", data_dir)
    log.info("mpirun: %s", mpirun)
    log.info("python: %s", python_exe)

    return LoadDefEnv(
        loaddef_root=loaddef_root,
        data_dir=data_dir,
        mpirun=mpirun,
        python_exe=python_exe,
        subprocess_env=env,
    )


def _find_mpirun() -> str:
    """Locate mpirun, checking GipsyX bundled OpenMPI first."""
    # Check GipsyX bundled OpenMPI
    gipsyx_mpirun = Path.home() / "GipsyX/GipsyX-2.0/lib/openmpi/bin/mpirun"
    if gipsyx_mpirun.exists():
        return str(gipsyx_mpirun)

    # Check system PATH
    system_mpirun = shutil.which("mpirun")
    if system_mpirun:
        return system_mpirun

    raise RuntimeError(
        "mpirun not found. Install OpenMPI or set --mpirun-exe / GOTL_MPIRUN_EXE."
    )


def _find_loaddef_python() -> str:
    """Locate a Python interpreter with mpi4py installed."""
    # Check GipsyX venv39 (where we installed mpi4py)
    venv39 = Path.home() / "GipsyX/venv39/bin/python"
    if venv39.exists():
        return str(venv39)

    # Fall back to system python
    system_python = shutil.which("python3")
    if system_python:
        return system_python

    raise RuntimeError(
        "Python with mpi4py not found. Install mpi4py in a Python 3.9+ environment."
    )
