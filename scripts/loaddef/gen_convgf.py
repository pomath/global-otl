#!/usr/bin/env python
"""Build LoadDef convgf grids from raw tide-model NetCDF files.

Wraps the per-model readers in ``LoadDef-main/GRDGEN/utility/`` (used by
the hardcoded ``gen_otl.py`` in the upstream LoadDef repo) so we can run
them over an arbitrary constituent list without editing source files.

Usage::

    # TPXO9-Atlas v5: raw files are h_{const}_tpxo9_atlas_30_v5.nc
    python scripts/loaddef/gen_convgf.py tpxo9 \
        --input /path/to/tide_models/TPXO9_atlas_v5 \
        --output data/loaddef/Grid_Files/nc/OTL

    # EOT20: raw files are {Const}_ocean_eot20.nc
    python scripts/loaddef/gen_convgf.py eot20 \
        --input /path/to/tide_models/EOT20 \
        --output data/loaddef/Grid_Files/nc/OTL

Output filenames follow the LoadDef convention
``convgf_{MODEL}-{CONST}.nc``, matching ``TIDE_MODEL_PREFIXES`` in
``src/gotl/loaddef/batch.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "LoadDef-main"))

# Constituents in the gotl pipeline (see src/gotl/__init__.py). TPXO9-Atlas v5
# does not ship SSA; the convert step will fill it with zeros downstream.
_GOTL_CONSTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"]


def _tpxo9_input(input_dir: Path, const: str) -> Path | None:
    candidate = input_dir / f"h_{const.lower()}_tpxo9_atlas_30_v5.nc"
    return candidate if candidate.exists() else None


def _eot20_input(input_dir: Path, const: str) -> Path | None:
    # EOT20 file naming in the wild is inconsistent: DGFI's published archive
    # uses mixed case (Mf, Mm, Ssa) but some mirrors uppercase everything.
    # Probe both spellings rather than hard-coding one.
    name_map = {
        "M2": ["M2"], "S2": ["S2"], "N2": ["N2"], "K2": ["K2"],
        "K1": ["K1"], "O1": ["O1"], "P1": ["P1"], "Q1": ["Q1"],
        "MF": ["Mf", "MF"], "MM": ["Mm", "MM"], "SSA": ["Ssa", "SSA"],
    }
    for stem in name_map.get(const.upper(), []):
        candidate = input_dir / f"{stem}_ocean_eot20.nc"
        if candidate.exists():
            return candidate
    return None


def _write_convgf(
    output_path: Path,
    olat: np.ndarray,
    olon: np.ndarray,
    amp: np.ndarray,
    pha: np.ndarray,
) -> None:
    """Write a LoadDef convgf NetCDF (flat 1-D lon/lat/amp/pha)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(olat)
    with netCDF4.Dataset(output_path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("latitude", n)
        ds.createDimension("longitude", n)
        ds.createDimension("amplitude", n)
        ds.createDimension("phase", n)
        lat_v = ds.createVariable("latitude", float, ("latitude",))
        lon_v = ds.createVariable("longitude", float, ("longitude",))
        amp_v = ds.createVariable("amplitude", float, ("amplitude",))
        pha_v = ds.createVariable("phase", float, ("phase",))
        lat_v.units = "degree_north"
        lon_v.units = "degree_east"
        amp_v.units = "m"
        pha_v.units = "degree"
        lat_v[:] = olat
        lon_v[:] = olon
        amp_v[:] = amp
        pha_v[:] = pha


def _read_tpxo9(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from GRDGEN.utility import read_tpxo9atlas
    olat, olon, amp, pha, *_ = read_tpxo9atlas.main(str(filename))
    return olat, olon, amp, pha


def _read_eot20(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EOT20 reader (DGFI-TUM format).

    The sample we have uses ``amplitude`` (cm) + ``phase`` (deg) on a
    ``(lat, lon)`` grid. ``real``/``imag`` (cm) are also present and
    used as a fallback if amplitude/phase are missing.
    """
    from math import pi
    with netCDF4.Dataset(filename) as f:
        variables = set(f.variables)
        lon = np.asarray(f.variables["lon"][:])
        lat = np.asarray(f.variables["lat"][:])
        if "amplitude" in variables and "phase" in variables:
            amp = np.asarray(f.variables["amplitude"][:], dtype=float) / 100.0
            pha = np.asarray(f.variables["phase"][:], dtype=float)
        elif "real" in variables and "imag" in variables:
            re = np.asarray(f.variables["real"][:], dtype=float)
            im = np.asarray(f.variables["imag"][:], dtype=float)
            amp = np.abs(re + 1j * im) / 100.0
            pha = np.arctan2(im, re) * 180.0 / pi
        else:
            raise RuntimeError(
                f"{filename.name}: unrecognized EOT20 layout; "
                f"variables={sorted(variables)}"
            )

    amp = np.ma.filled(amp, fill_value=0.0)
    pha = np.ma.filled(pha, fill_value=0.0)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    return grid_lat.flatten(), grid_lon.flatten(), amp.flatten(), pha.flatten()


_MODELS = {
    "tpxo9": {
        "label": "TPXO9-Atlas",
        "input_fn": _tpxo9_input,
        "read_fn": _read_tpxo9,
    },
    "eot20": {
        "label": "EOT20",
        "input_fn": _eot20_input,
        "read_fn": _read_eot20,
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(_MODELS.keys()))
    ap.add_argument("--input", required=True, type=Path,
                    help="Directory with raw per-constituent .nc files.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write convgf_<MODEL>-<CONST>.nc files.")
    ap.add_argument("--constituents", default=",".join(_GOTL_CONSTS),
                    help="Comma-separated constituents (default: 11 gotl consts).")
    args = ap.parse_args()

    spec = _MODELS[args.model]
    label = spec["label"]
    consts = [c.strip().upper() for c in args.constituents.split(",") if c.strip()]

    args.output.mkdir(parents=True, exist_ok=True)

    converted = 0
    for const in consts:
        src = spec["input_fn"](args.input, const)
        if src is None:
            print(f"[skip] {const}: no input in {args.input}", file=sys.stderr)
            continue
        dst = args.output / f"convgf_{label}-{const}.nc"
        print(f"[read] {src.name}")
        olat, olon, amp, pha = spec["read_fn"](src)
        print(f"[write] {dst.name}  (n={len(olat)})")
        _write_convgf(dst, olat, olon, amp, pha)
        converted += 1

    print(f"\nConverted {converted}/{len(consts)} constituents for {label}.")
    return 0 if converted else 1


if __name__ == "__main__":
    raise SystemExit(main())
