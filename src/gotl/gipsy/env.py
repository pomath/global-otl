"""GipsyX environment setup.

GipsyX tools require environment variables set by rc_GipsyX.sh.
This module sources that script and captures the resulting environment
for use with subprocess calls.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def load_gipsyx_env(rc_path: Path | None = None) -> dict[str, str]:
    """Source rc_GipsyX.sh and return the resulting environment dict.

    Sets LD_LIBRARY_PATH for bundled HDF5/Boost/OpenMPI libs, and
    PYTHONPATH for the gcore Python module.  If a Python 3.9 venv
    exists alongside the GipsyX install (``venv39/``), its ``bin/``
    directory is prepended to PATH so that ``#!/usr/bin/env python3``
    in GipsyX scripts resolves to the compatible interpreter.

    Args:
        rc_path: Path to rc_GipsyX.sh. If None, returns a copy of the
                 current environment (assumes GipsyX is already on PATH).

    Returns:
        Environment dict suitable for subprocess.run(env=...).

    Raises:
        FileNotFoundError: if rc_path does not exist.
        RuntimeError: if sourcing the RC file fails.
    """
    if rc_path is None:
        return dict(os.environ)

    rc_path = Path(rc_path)
    if not rc_path.exists():
        raise FileNotFoundError(f"GipsyX RC file not found: {rc_path}")

    cmd = f"source {rc_path} && env -0"
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to source {rc_path}: {result.stderr.strip()}"
        )

    env: dict[str, str] = {}
    for entry in result.stdout.split("\0"):
        if "=" in entry:
            key, _, val = entry.partition("=")
            env[key] = val

    # Add bundled shared libraries to LD_LIBRARY_PATH
    gipsyx_dir = rc_path.resolve().parent
    log.info("gipsyx_dir=%s, parent=%s", gipsyx_dir, gipsyx_dir.parent)
    lib_dirs = [
        gipsyx_dir / "lib" / "hdf5",
        gipsyx_dir / "lib" / "boost",
        gipsyx_dir / "lib" / "openmpi" / "lib",
    ]
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(str(d) for d in lib_dirs if d.is_dir())
    env["LD_LIBRARY_PATH"] = f"{new_ld}:{existing_ld}" if existing_ld else new_ld

    # Add gcore Python module to PYTHONPATH (prefer 3.9, fall back to others)
    py_lib = None
    for ver in ("python3.9", "python3.8", "python3.7", "python3.6"):
        candidate = gipsyx_dir / "lib" / ver
        if candidate.is_dir():
            py_lib = candidate
            break
    if py_lib:
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{py_lib}:{existing_pp}" if existing_pp else str(py_lib)

    # If a venv39 exists next to the GipsyX install, put it first on PATH
    # so #!/usr/bin/env python3 resolves to the compatible interpreter.
    venv39_bin = gipsyx_dir.parent / "venv39" / "bin"
    if venv39_bin.is_dir():
        env["PATH"] = f"{venv39_bin}:{env.get('PATH', '')}"
        log.info("Using Python 3.9 venv: %s", venv39_bin)

    # If a standalone python39 is bundled (cloud deployment), set PYTHONHOME
    # so the interpreter can find its stdlib (encodings, etc.)
    # Check multiple locations: next to GipsyX, or via env var
    python39_dir = gipsyx_dir.parent / "python39"
    if not python39_dir.is_dir():
        p39_env = env.get("GOTL_PYTHON39_HOME", "") or os.environ.get("GOTL_PYTHON39_HOME", "")
        if p39_env:
            python39_dir = Path(p39_env)
    if python39_dir.is_dir():
        env["PYTHONHOME"] = str(python39_dir)
        # Also add venv site-packages to PYTHONPATH so gcore + deps are found
        venv_sp = gipsyx_dir.parent / "venv39" / "lib" / "python3.9" / "site-packages"
        pp_parts = []
        if py_lib:
            pp_parts.append(str(py_lib))
        if venv_sp.is_dir():
            pp_parts.append(str(venv_sp))
        existing_pp = env.get("PYTHONPATH", "")
        if existing_pp:
            pp_parts.append(existing_pp)
        if pp_parts:
            env["PYTHONPATH"] = ":".join(pp_parts)
        log.info("Set PYTHONHOME for bundled python39: %s", python39_dir)

    return env
