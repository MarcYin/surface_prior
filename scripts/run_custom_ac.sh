#!/usr/bin/env bash
# MAIAC-select + 6S-correct custom atmospheric-correction composite pipeline.
# Runs the 3 phases across the base (GEE) and SIAC rt6s (native 6S) envs.
#
# Usage:
#   scripts/run_custom_ac.sh <site_lon> <site_lat> <w> <s> <e> <n> <start> <end> <aeronet_site> <out>
# Example (Cairo / Nile Delta, Jul 2022):
#   scripts/run_custom_ac.sh 31.290 30.081 31.07 29.90 31.51 30.26 2022-07-01 2022-08-01 Cairo_EMA_2 /tmp/cac
set -euo pipefail
LON=$1; LAT=$2; W=$3; S=$4; E=$5; N=$6; START=$7; END=$8; AER=$9; OUT=${10}
BASE=/home/users/marcyin/.pixi/envs/base/bin/python
RT6S=/home/users/marcyin/SIAC/.pixi/envs/rt6s/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "### Phase 1: select low-AOD days + sample pixels (GEE)"
$BASE "$HERE/custom_ac_phase1.py" --site "$LON" "$LAT" --bbox "$W" "$S" "$E" "$N" \
    --month "$START" "$END" --aeronet "$AER" --frac 0.6 --out "$OUT"

echo "### Phase 2: per-day 6S coefficients over a TCWV LUT (SIAC native 6S)"
PYTHONPATH=/home/users/marcyin/SIAC/python $RT6S "$HERE/custom_ac_phase2_sixs.py" \
    --meta "${OUT}_meta.json" --out "${OUT}_coeffs.npz"

echo "### Phase 3: correct (per-pixel WVP) + composite + compare vs AERONET truth"
$BASE "$HERE/custom_ac_phase3.py" --meta "${OUT}_meta.json" \
    --pixels "${OUT}_pixels.npz" --coeffs "${OUT}_coeffs.npz"
