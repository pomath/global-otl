"""LoadDef batch orchestrator.

Generates .otl parameter files for stations by running the full
LoadDef pipeline: Love Numbers → Green's Functions → Convolution → .otl.
Checks for pre-computed data and skips steps when possible.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from gotl.loaddef.convert import convert_convolution_to_otl, write_station_file
from gotl.loaddef.env import LoadDefEnv
from gotl.loaddef.runner import run_convolution, run_greens_functions, run_love_numbers

log = logging.getLogger(__name__)

# Mapping from user-facing tide model name to convolution grid prefix
TIDE_MODEL_PREFIXES = {
    "FES2014": "convgf_FES2014",
    "FES2014b": "convgf_FES2014",
    "TPXO9": "convgf_TPXO9-Atlas",
    "EOT20": "convgf_EOT20",
}


@dataclass
class OTLGenResult:
    """Result of OTL parameter generation for one station."""

    station: str
    tide_model: str
    earth_model: str
    otl_path: Path | None = None
    steps_run: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    error: str | None = None


def generate_otl_params(
    stnm: str,
    lat: float,
    lon: float,
    alt: float,
    env: LoadDefEnv,
    otl_params_dir: Path,
    tide_model: str = "FES2014",
    earth_model: str = "PREM",
    rfm: str = "cm",
    mpi_np: int = 4,
    force: bool = False,
) -> OTLGenResult:
    """Generate .otl file for one station.

    Pipeline:
        1. Check for pre-computed Green's functions → skip LN+GF if found
        2. If not: run Love Numbers → run Green's Functions → cache result
        3. Run convolution
        4. Convert convolution output → Harpos .otl format

    Args:
        stnm: 4-letter station code.
        lat: Latitude (degrees).
        lon: Longitude (degrees, -180 to 180).
        alt: Altitude (metres).
        env: LoadDef environment.
        otl_params_dir: Output directory for .otl files.
        tide_model: Tide model name.
        earth_model: Earth model name.
        rfm: Reference frame ("cm" or "cf").
        mpi_np: Number of MPI processes.
        force: Regenerate even if output exists.

    Returns:
        OTLGenResult with status and output path.
    """
    result = OTLGenResult(station=stnm, tide_model=tide_model, earth_model=earth_model)

    # Check if output already exists
    otl_path = Path(otl_params_dir) / f"{stnm}_{tide_model}.otl"
    if otl_path.exists() and not force:
        log.info("OTL file already exists: %s (use --force to regenerate)", otl_path)
        result.otl_path = otl_path
        result.steps_skipped = ["love_numbers", "greens_functions", "convolution", "convert"]
        return result

    # Step 1-2: Get Green's functions (pre-computed or compute)
    grn_file = _ensure_greens_functions(env, earth_model, rfm, mpi_np, result)
    if grn_file is None:
        return result  # error set by _ensure_greens_functions

    # Step 3: Run convolution
    grid_prefix = TIDE_MODEL_PREFIXES.get(tide_model)
    if grid_prefix is None:
        result.error = (
            f"Unknown tide model '{tide_model}'. "
            f"Known: {', '.join(TIDE_MODEL_PREFIXES.keys())}"
        )
        return result

    grid_dir = env.data_dir / "Grid_Files" / "nc" / "OTL"
    lsmask_file = env.data_dir / "ETOPO1_Ice_g_gmt4_wADD.txt"

    if not lsmask_file.exists():
        result.error = f"Land-sea mask not found: {lsmask_file}"
        return result

    with tempfile.TemporaryDirectory(prefix="loaddef_otl_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Create station file
        sta_file = tmpdir / "stations.txt"
        write_station_file([(stnm, lat, lon)], sta_file)

        # Convolution output dir
        conv_dir = tmpdir / "convolution"

        log.info("Running convolution for %s (%s, %s)...", stnm, tide_model, earth_model)
        cn_result = run_convolution(
            env=env,
            grn_file=grn_file,
            grid_dir=grid_dir,
            grid_prefix=grid_prefix,
            lsmask_file=lsmask_file,
            station_file=sta_file,
            output_dir=conv_dir,
            rfm=rfm,
            earth_model=earth_model,
            mpi_np=mpi_np,
        )

        if not cn_result.success:
            result.error = f"Convolution failed: {cn_result.error}"
            result.steps_run.append("convolution (failed)")
            return result

        result.steps_run.append("convolution")
        log.info("Convolution complete (%.1fs)", cn_result.elapsed_sec)

        # Step 4: Convert to .otl format
        # Find the convolution output file for this station
        conv_file = _find_convolution_output(conv_dir, stnm)
        if conv_file is None:
            result.error = (
                f"Convolution output not found for {stnm} in {conv_dir}. "
                f"Files: {[f.name for f in conv_dir.iterdir()] if conv_dir.exists() else '(none)'}"
            )
            return result

        convert_convolution_to_otl(
            convolution_file=conv_file,
            station=stnm,
            earth_model=earth_model,
            tide_model=tide_model,
            lat=lat,
            lon=lon,
            alt=alt,
            output_path=otl_path,
        )

        result.steps_run.append("convert")
        result.otl_path = otl_path

    return result


def _ensure_greens_functions(
    env: LoadDefEnv,
    earth_model: str,
    rfm: str,
    mpi_np: int,
    result: OTLGenResult,
) -> Path | None:
    """Get Green's functions, computing if necessary.

    Returns path to Green's function file, or None on error.
    """
    # Check for pre-computed Green's functions
    # Try multiple naming conventions: cm_PREM.txt, ce_PREM.txt, cf_PREM.txt
    grn_dir = env.data_dir / "Greens_Functions"
    grn_file = None
    for prefix in (rfm, "ce", "cf", "cm"):
        candidate = grn_dir / f"{prefix}_{earth_model}.txt"
        if candidate.exists():
            grn_file = candidate
            break

    if grn_file is not None:
        log.info("Using pre-computed Green's functions: %s", grn_file)
        result.steps_skipped.extend(["love_numbers", "greens_functions"])
        return grn_file

    # Need to compute: Love Numbers → Green's Functions
    log.info("No pre-computed Green's functions for %s. Computing...", earth_model)

    # Love Numbers
    log.info("Step 1: Computing Love Numbers for %s...", earth_model)
    ln_result = run_love_numbers(
        env=env, planet_model=earth_model,
        output_dir=env.data_dir, mpi_np=mpi_np,
    )
    if not ln_result.success:
        result.error = f"Love number computation failed: {ln_result.error}"
        result.steps_run.append("love_numbers (failed)")
        return None

    result.steps_run.append("love_numbers")
    log.info("Love numbers complete (%.1fs)", ln_result.elapsed_sec)

    # Find LLN output
    lln_file = env.data_dir / "Love_Numbers" / "LLN" / f"lln_{earth_model}.txt"
    if not lln_file.exists():
        result.error = f"Love number output not found: {lln_file}"
        return None

    # Green's Functions
    log.info("Step 2: Computing Green's Functions for %s...", earth_model)
    gf_result = run_greens_functions(
        env=env, lln_file=lln_file, earth_model=earth_model,
        output_dir=env.data_dir, mpi_np=mpi_np,
    )
    if not gf_result.success:
        result.error = f"Green's function computation failed: {gf_result.error}"
        result.steps_run.append("greens_functions (failed)")
        return None

    result.steps_run.append("greens_functions")
    log.info("Green's functions complete (%.1fs)", gf_result.elapsed_sec)

    # Check output
    if grn_file.exists():
        return grn_file

    # Try alternative naming
    for candidate in (env.data_dir / "Greens_Functions").iterdir():
        if candidate.is_file() and earth_model in candidate.name:
            return candidate

    result.error = f"Green's function output not found at {grn_file}"
    return None


def _find_convolution_output(conv_dir: Path, stnm: str) -> Path | None:
    """Find the convolution output file for a station."""
    if not conv_dir.exists():
        return None

    # Pattern: cn_OceanOnly_{stn}_cm_convgf_{TM}_{EM}.txt
    # or just: {stn}_cm_convgf_{TM}_{EM}.txt
    stnm_lower = stnm.lower()
    for f in conv_dir.iterdir():
        if f.is_file() and stnm_lower in f.name.lower() and f.suffix == ".txt":
            return f

    return None


def generate_otl_params_batch(
    stations: Iterable[tuple[str, float, float, float]],
    env: LoadDefEnv,
    otl_params_dir: Path,
    tide_model: str = "FES2014",
    earth_model: str = "PREM",
    rfm: str = "cm",
    mpi_np: int = 4,
    force: bool = False,
    timeout_sec: int | None = None,
) -> list[OTLGenResult]:
    """Generate .otl files for many stations in a single convolution pass.

    LoadDef's ``load_convolution`` accepts a station file with many rows
    and streams each tide-model grid only once, interpolating it onto
    every station. For N stations and K constituents we pay one grid
    read × K (shared) instead of one grid read × K × N. At the 58M-point
    TPXO9 grid scale this is the difference between minutes and hours.

    Args:
        stations: iterable of ``(stnm, lat, lon, alt)`` tuples. ``alt``
            is only used for the Harpos header line.

    Returns one ``OTLGenResult`` per input station (skipped stations keep
    ``steps_skipped`` set; failures surface via ``error``).
    """
    stations = list(stations)
    results = {stnm: OTLGenResult(station=stnm, tide_model=tide_model,
                                   earth_model=earth_model)
               for stnm, _, _, _ in stations}

    # Skip stations whose OTL already exists.
    todo: list[tuple[str, float, float, float]] = []
    for stnm, lat, lon, alt in stations:
        otl_path = Path(otl_params_dir) / f"{stnm}_{tide_model}.otl"
        if otl_path.exists() and not force:
            log.info("OTL file already exists: %s", otl_path)
            results[stnm].otl_path = otl_path
            results[stnm].steps_skipped = [
                "love_numbers", "greens_functions", "convolution", "convert",
            ]
        else:
            todo.append((stnm, lat, lon, alt))

    if not todo:
        return [results[s] for s, _, _, _ in stations]

    # Shared Green's function (same for every station).
    placeholder = OTLGenResult(station="__shared__", tide_model=tide_model,
                                earth_model=earth_model)
    grn_file = _ensure_greens_functions(env, earth_model, rfm, mpi_np, placeholder)
    if grn_file is None:
        for stnm, _, _, _ in todo:
            results[stnm].error = placeholder.error
        return [results[s] for s, _, _, _ in stations]

    grid_prefix = TIDE_MODEL_PREFIXES.get(tide_model)
    if grid_prefix is None:
        for stnm, _, _, _ in todo:
            results[stnm].error = (
                f"Unknown tide model '{tide_model}'. "
                f"Known: {', '.join(TIDE_MODEL_PREFIXES.keys())}"
            )
        return [results[s] for s, _, _, _ in stations]

    grid_dir = env.data_dir / "Grid_Files" / "nc" / "OTL"
    lsmask_file = env.data_dir / "ETOPO1_Ice_g_gmt4_wADD.txt"
    if not lsmask_file.exists():
        for stnm, _, _, _ in todo:
            results[stnm].error = f"Land-sea mask not found: {lsmask_file}"
        return [results[s] for s, _, _, _ in stations]

    with tempfile.TemporaryDirectory(prefix="loaddef_otl_batch_") as tmpdir:
        tmpdir = Path(tmpdir)
        sta_file = tmpdir / "stations.txt"
        write_station_file([(s, la, lo) for s, la, lo, _ in todo], sta_file)

        conv_dir = tmpdir / "convolution"
        log.info("Running convolution for %d stations (%s, %s)...",
                 len(todo), tide_model, earth_model)
        conv_kwargs = dict(
            env=env, grn_file=grn_file, grid_dir=grid_dir,
            grid_prefix=grid_prefix, lsmask_file=lsmask_file,
            station_file=sta_file, output_dir=conv_dir,
            rfm=rfm, earth_model=earth_model, mpi_np=mpi_np,
        )
        if timeout_sec is not None:
            conv_kwargs["timeout_sec"] = timeout_sec
        cn_result = run_convolution(**conv_kwargs)
        if not cn_result.success:
            for stnm, _, _, _ in todo:
                results[stnm].error = f"Convolution failed: {cn_result.error}"
                results[stnm].steps_run.append("convolution (failed)")
            return [results[s] for s, _, _, _ in stations]

        log.info("Convolution complete (%.1fs)", cn_result.elapsed_sec)

        for stnm, lat, lon, alt in todo:
            results[stnm].steps_skipped.extend(["love_numbers", "greens_functions"])
            results[stnm].steps_run.append("convolution")
            conv_file = _find_convolution_output(conv_dir, stnm)
            if conv_file is None:
                results[stnm].error = (
                    f"Convolution output not found for {stnm} in {conv_dir}"
                )
                continue

            otl_path = Path(otl_params_dir) / f"{stnm}_{tide_model}.otl"
            convert_convolution_to_otl(
                convolution_file=conv_file, station=stnm,
                earth_model=earth_model, tide_model=tide_model,
                lat=lat, lon=lon, alt=alt, output_path=otl_path,
            )
            results[stnm].steps_run.append("convert")
            results[stnm].otl_path = otl_path

    return [results[s] for s, _, _, _ in stations]
