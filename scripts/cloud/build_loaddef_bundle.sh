#!/bin/bash
# Build a tarball of everything a Coiled worker needs to run
# gen_otl_fleet.py for the TPXO9 tide model.
#
# Includes:
#   - gotl source (src/, scripts/, station_registry, pyproject)
#   - LoadDef-main source (minus doc/ and input/Land_Sea/)
#   - data/loaddef/ supporting files + TPXO9 grids only (FES2014 / EOT20 excluded)
#
# Output: /tmp/loaddef-cloud.tar.gz  (~3.5 GB)
#
# The worker reuses the existing gipsyx-cloud.tar.gz bundle for
# mpirun + venv39 (has mpi4py prebuilt against OpenMPI 1.10), so
# this bundle only needs LoadDef-specific bits.
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT=/tmp/loaddef-cloud.tar.gz
STAGE=/tmp/loaddef_bundle_stage
rm -rf "$STAGE" "$OUT"
mkdir -p "$STAGE"

# --- 1. gotl source ---
mkdir -p "$STAGE/repo"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  src/ "$STAGE/repo/src/"
rsync -a scripts/ "$STAGE/repo/scripts/"
cp pyproject.toml poetry.lock "$STAGE/repo/"
cp -r data/station_registry.csv "$STAGE/repo/data/" 2>/dev/null || (
  mkdir -p "$STAGE/repo/data"
  cp data/station_registry.csv "$STAGE/repo/data/station_registry.csv"
)

# --- 2. LoadDef-main (strip doc/ and input/Land_Sea/ — saved ~100MB) ---
rsync -a --exclude='__pycache__' --exclude='doc' \
  --exclude='input/Land_Sea' \
  LoadDef-main/ "$STAGE/LoadDef-main/"

# --- 3. LoadDef data: only TPXO9 grids + supporting files ---
mkdir -p "$STAGE/repo/data/loaddef/Grid_Files/nc/OTL"
mkdir -p "$STAGE/repo/data/loaddef/Greens_Functions"
cp data/loaddef/ETOPO1_Ice_g_gmt4_wADD.txt "$STAGE/repo/data/loaddef/"
cp data/loaddef/Greens_Functions/ce_PREM.txt "$STAGE/repo/data/loaddef/Greens_Functions/"
cp -r data/loaddef/Planet_Models "$STAGE/repo/data/loaddef/"
# TPXO9 grids only
cp data/loaddef/Grid_Files/nc/OTL/convgf_TPXO9*.nc \
   "$STAGE/repo/data/loaddef/Grid_Files/nc/OTL/"

# --- 4. otl_params output dir (empty) ---
mkdir -p "$STAGE/repo/data/otl_params"

echo "Stage tree size:"
du -sh "$STAGE"/*

echo ""
echo "Tarring..."
tar -C "$STAGE" -czf "$OUT" . --checkpoint=10000 --checkpoint-action=dot
echo ""
echo "Bundle: $OUT ($(du -sh "$OUT" | awk '{print $1}'))"
