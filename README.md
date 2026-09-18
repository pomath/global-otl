# global-otl (`gotl`)

Global GPS ocean tidal loading (OTL) analysis. `gotl` processes worldwide IGS/GNSS
GPS stations through a GipsyX kinematic-PPP -> trajectory-model -> harmonic-analysis
pipeline to estimate per-station ocean tidal loading displacement coefficients, then
evaluates ocean tide models (TPXO9/TPXO10-Atlas, FES2004/FES2014b/FES2022b, EOT20,
DTU23, GOT5.5) against the GPS-observed signal.

## What it does

The pipeline runs in five phases:

1. **GipsyX kinematic PPP** - position time series from RINEX, with OTL applied during PPP.
2. **LoadDef OTL prediction** - Love Numbers -> Green's Functions -> convolution with a
   tide model -> per-station Harpos `.otl` loading coefficients.
3. **Hardisp** - IERS `hardisp` time series of the predicted loading displacement.
4. **SLTM + VTide** - a Standard Linear Trajectory Model (trend + jumps + oscillations,
   fit by IRWLS) strips non-tidal signal; VTide harmonic analysis estimates 11 tidal
   constituents (M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA) with uncertainties.
5. **Multi-model comparison** - vector misfit of GPS-derived vs modelled loading per constituent.

Algorithms, sign conventions, and file formats are documented in the module docstrings
under `src/gotl/`.

## Install

Requires Python >=3.11 and [Poetry](https://python-poetry.org/).

```bash
poetry install                 # core pipeline
poetry install --with loaddef  # + mpi4py for LoadDef convolution (needs MPI dev headers)
```

This installs the `gotl` CLI.

## External dependencies (not included)

This repository contains the analysis code only. To run the full pipeline you also need:

- **GipsyX 2.x** (JPL) - kinematic PPP engine + reference data.
- **[LoadDef](https://github.com/hrmartens/LoadDef)** (Martens et al.) - load-deformation toolkit (Phase 2).
- **IERS `hardisp`** - Fortran source is vendored under `hardisp/`; build with
  `cd hardisp/src && make`, then link `hardisp_exe`.
- **Ocean tide model grids** - per-constituent NetCDF files from the model providers
  (TPXO9/TPXO10-atlas from OSU, FES2004/FES2014b/FES2022b from AVISO+, EOT20 from
  DGFI-TUM, DTU23 from DTU Space, GOT5.5 from NASA GSFC), converted to LoadDef
  `convgf_*` grids with `scripts/loaddef/gen_convgf.py`.

Large inputs/outputs (RINEX, tide grids, results, caches) are gitignored and supplied via
environment variables (`GOTL_*`).

## Usage

```bash
gotl run-all                                   # all stations in data/station_registry.csv
gotl run aboa                                  # single station
gotl status                                    # per-station completion table
gotl compare aboa                              # GPS vs tide-model residuals
gotl gen-otl-params aboa --tide-model FES2014  # LoadDef OTL params
gotl gen-hardisp aboa --years 2000,2001        # hardisp time series
```

## Layout

```
src/gotl/   gotl package (pipeline, sltm, tides, io, gipsy, loaddef, compare, cloud, web)
scripts/    LoadDef / download / cloud helper scripts
hardisp/    IERS hardisp Fortran source
tests/      pytest suite (synthetic fixtures)
data/       station_registry.csv (all other data is external / gitignored)
```

## Tests

```bash
poetry run pytest tests/ -v
```

## License

MIT - see [LICENSE](LICENSE).

The vendored `hardisp/` Fortran is IERS Conventions software and retains its own headers.
`gotl` builds on GipsyX (JPL), LoadDef (Martens et al.), and VTide/UTide; obtain and use
those under their respective licenses.
