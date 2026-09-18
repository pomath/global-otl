#!/usr/bin/env python
"""Generate .otl files for the current station fleet in a single batch pass.

Drives ``generate_otl_params_batch`` so each tide-model grid is streamed once
and interpolated onto every station, rather than re-read per station (the
CLI path calls the per-station function which is fine for one-offs but
wasteful when you're filling in a new tide model across the fleet).

Usage::

    scripts/loaddef/gen_otl_fleet.py EOT20
    scripts/loaddef/gen_otl_fleet.py TPXO9 --stations aboa,brux
    scripts/loaddef/gen_otl_fleet.py FES2014 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gotl.loaddef.batch import generate_otl_params_batch
from gotl.loaddef.env import load_loaddef_env
from gotl.registry import load_registry

DEFAULT_LOADDEF_ROOT = _REPO_ROOT / "LoadDef-main"
DEFAULT_LOADDEF_DATA = _REPO_ROOT / "data" / "loaddef"
DEFAULT_OTL_PARAMS = _REPO_ROOT / "data" / "otl_params"
DEFAULT_REGISTRY = _REPO_ROOT / "data" / "station_registry.csv"


def _current_fleet(otl_params: Path) -> list[str]:
    """Stations that already have a FES2014 .otl — our canonical fleet."""
    return sorted(p.name.removesuffix("_FES2014.otl")
                  for p in otl_params.glob("*_FES2014.otl"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tide_model",
                    help="Any key of gotl.loaddef.batch.TIDE_MODEL_PREFIXES "
                         "(FES2014 | FES2022b | FES2004 | TPXO9 | TPXO10 | "
                         "EOT20 | DTU23 | GOT55)")
    ap.add_argument("--stations", default=None,
                    help="Comma-separated station list (default: stations "
                         "with an existing FES2014 .otl).")
    ap.add_argument("--earth-model", default="PREM")
    ap.add_argument("--mpi-np", type=int, default=4)
    ap.add_argument("--timeout-sec", type=int, default=3600,
                    help="Convolution timeout in seconds. Raise for large "
                         "station batches (FES2014 ≈ 240 s/station).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--loaddef-root", type=Path, default=DEFAULT_LOADDEF_ROOT)
    ap.add_argument("--loaddef-data", type=Path, default=DEFAULT_LOADDEF_DATA)
    ap.add_argument("--otl-params", type=Path, default=DEFAULT_OTL_PARAMS)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    registry = load_registry(args.registry)
    if args.stations:
        stn_list = [s.strip() for s in args.stations.split(",") if s.strip()]
    else:
        stn_list = _current_fleet(args.otl_params)

    missing = [s for s in stn_list if s not in registry]
    if missing:
        print(f"ERROR: stations not in registry: {missing}", file=sys.stderr)
        return 1

    stations = [(s, registry[s].lat, registry[s].lon, registry[s].alt_m)
                for s in stn_list]
    print(f"Generating {args.tide_model} .otl for {len(stations)} stations: "
          f"{', '.join(s for s, *_ in stations)}")

    env = load_loaddef_env(args.loaddef_root, args.loaddef_data)

    results = generate_otl_params_batch(
        stations=stations,
        env=env,
        otl_params_dir=args.otl_params,
        tide_model=args.tide_model,
        earth_model=args.earth_model,
        mpi_np=args.mpi_np,
        force=args.force,
        timeout_sec=args.timeout_sec,
    )

    ok = fail = skipped = 0
    for r in results:
        if r.error:
            print(f"FAIL {r.station}: {r.error}", file=sys.stderr)
            fail += 1
        elif "convolution" not in r.steps_run and "convert" not in r.steps_run:
            print(f"  -- {r.station}: already had {args.tide_model} .otl")
            skipped += 1
        else:
            print(f"  OK {r.station}: {r.otl_path}")
            ok += 1

    print(f"\n{ok} generated, {skipped} pre-existing, {fail} failed.")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
