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

    # Same pattern for tpxo10 | fes2022 | fes2004 | dtu23 | got55.
    # These probe filenames case-insensitively (per-constituent .nc with
    # the model token in the name); the gridded reader sniffs variable
    # names and units, so archive-layout drift usually just works.

Output filenames follow the LoadDef convention
``convgf_{MODEL}-{CONST}.nc``, matching ``TIDE_MODEL_PREFIXES`` in
``src/gotl/loaddef/batch.py``.
"""
from __future__ import annotations

import argparse
import re
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


def _token_input(*tokens: str):
    """Build an input probe for per-constituent .nc files.

    Matches (case-insensitively) any ``.nc`` in the input directory whose
    name contains every ``token`` and the constituent as a delimited token
    (so ``m2`` hits ``h_m2_tpxo10_atlas_30_v2.nc`` but not ``mm2``).
    Exact archive naming varies by provider/mirror; probing beats
    hard-coding one spelling (cf. the EOT20 case mess above).
    """
    def probe(input_dir: Path, const: str) -> Path | None:
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(const.lower())}(?![a-z0-9])")
        hits = sorted(
            p for p in input_dir.glob("*.nc")
            if all(t in p.name.lower() for t in tokens)
            and pat.search(p.name.lower())
        )
        if hits:
            return hits[0]
        # Some archives name files bare, e.g. GOT's "m2.nc".
        bare = input_dir / f"{const.lower()}.nc"
        return bare if bare.exists() else None
    return probe


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


def _amp_scale(var) -> float:
    """Metres-per-unit from a NetCDF variable's ``units`` attribute.

    Tide-model grids ship in cm (FES, EOT, GOT convention), mm, or m;
    fall back to cm when the attribute is absent since that is by far
    the most common.
    """
    units = str(getattr(var, "units", "")).strip().lower()
    if units.startswith("mm") or "millimet" in units:
        return 1e-3
    if units.startswith("cm") or "centimet" in units:
        return 1e-2
    if units.startswith("m") and "deg" not in units:
        return 1.0
    return 1e-2


def _read_gridded(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generic reader for regular lat/lon amp/phase constituent grids.

    Covers the FES2004/FES2022/DTU23/GOT-style deliveries: 1-D lat + lon
    coordinates with 2-D amplitude+phase (or real+imag) fields. Variable
    names and amplitude units are probed rather than assumed.
    """
    from math import pi
    with netCDF4.Dataset(filename) as f:
        low = {name.lower(): name for name in f.variables}

        def pick(*candidates):
            for c in candidates:
                if c in low:
                    return f.variables[low[c]]
            return None

        lat_v = pick("lat", "latitude", "y")
        lon_v = pick("lon", "longitude", "x")
        amp_v = pick("amplitude", "amp", "ha", "h_amp", "tide_amp")
        pha_v = pick("phase", "pha", "hg", "phase_lag", "h_pha", "tide_pha")
        if lat_v is None or lon_v is None:
            raise RuntimeError(
                f"{filename.name}: no lat/lon coordinates; "
                f"variables={sorted(f.variables)}"
            )
        lat = np.asarray(lat_v[:])
        lon = np.asarray(lon_v[:])

        if amp_v is not None and pha_v is not None:
            amp = np.asarray(amp_v[:], dtype=float) * _amp_scale(amp_v)
            pha = np.asarray(pha_v[:], dtype=float)
        else:
            re_v = pick("real", "hre", "h_re", "wr")
            im_v = pick("imag", "him", "h_im", "wi")
            if re_v is None or im_v is None:
                raise RuntimeError(
                    f"{filename.name}: no amplitude/phase or real/imag pair; "
                    f"variables={sorted(f.variables)}"
                )
            scale = _amp_scale(re_v)
            re_ = np.asarray(re_v[:], dtype=float)
            im_ = np.asarray(im_v[:], dtype=float)
            amp = np.abs(re_ + 1j * im_) * scale
            pha = np.arctan2(im_, re_) * 180.0 / pi

    amp = np.ma.filled(amp, fill_value=0.0)
    pha = np.ma.filled(pha, fill_value=0.0)
    # Grids may be (lat, lon) or (lon, lat); orient to (lat, lon).
    if amp.shape == (len(lon), len(lat)) and len(lat) != len(lon):
        amp, pha = amp.T, pha.T
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
    # TPXO10-Atlas ships in the same h_{const}_tpxo10_atlas_* transposed
    # layout as TPXO9-Atlas, so LoadDef's tpxo9atlas reader applies.
    "tpxo10": {
        "label": "TPXO10-Atlas",
        "input_fn": _token_input("tpxo10"),
        "read_fn": _read_tpxo9,
    },
    "fes2022": {
        "label": "FES2022b",
        "input_fn": _token_input("fes2022"),
        "read_fn": _read_gridded,
    },
    "fes2004": {
        "label": "FES2004",
        "input_fn": _token_input("fes2004"),
        "read_fn": _read_gridded,
    },
    "dtu23": {
        "label": "DTU23",
        "input_fn": _token_input("dtu23"),
        "read_fn": _read_gridded,
    },
    "got55": {
        "label": "GOT55",
        "input_fn": _token_input("got"),
        "read_fn": _read_gridded,
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
