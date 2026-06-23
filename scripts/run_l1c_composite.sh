#!/usr/bin/env bash
# End-to-end production L1C custom-AC composite:
#   MAIAC-select low-aerosol days -> GEE L1C TOA + Cloud Score+ -> 6S-correct -> best-pixel composite.
#
# Stages 1-3 build the per-scene atmosphere sidecar (MAIAC AOD + WVP + geometry -> 6S coeffs).
# Stage 4 is now the productionised package: the `surface-priors` CLI drives the
# S2L1CGeeSource (getPixels TOA + 6S correction + Cloud Score+ best-pixel) through
# the scout/select/windowed-fetch/compositor machinery and writes per-band GeoTIFFs.
#
#   export GEE_SERVICE_ACCOUNT=python-gee@gee-marc.iam.gserviceaccount.com
#   export GEE_SERVICE_ACCOUNT_KEY=/home/users/marcyin/gee-service-account.json
#   scripts/run_l1c_composite.sh 31.0 29.9 31.5 30.3 2022-07-01 2022-08-01 /tmp/l1c_cairo /tmp/l1c_cache
set -euo pipefail

X0=$1 Y0=$2 X1=$3 Y1=$4 M0=$5 M1=$6 SIDECAR_BASE=$7 CACHE=$8
FRAC=${9:-0.6}; RES=${10:-60}
BASE=/home/users/marcyin/.pixi/envs/base/bin/python
RT6S=/home/users/marcyin/SIAC/.pixi/envs/rt6s/bin/python
SD=$(dirname "$0")

echo "== 1/4 enumerate L1C scenes + MAIAC/WVP/geometry (GEE) =="
$BASE "$SD/build_atmo_sidecar.py" --bbox "$X0" "$Y0" "$X1" "$Y1" --month "$M0" "$M1" --out "${SIDECAR_BASE}"

echo "== 2/4 6S coefficients over TCWV LUT (SIAC native 6S) =="
PYTHONPATH=/home/users/marcyin/SIAC/python $RT6S "$SD/custom_ac_phase2_sixs.py" \
  --meta "${SIDECAR_BASE}_meta.json" --out "${SIDECAR_BASE}_coeffs.npz"

echo "== 3/4 merge -> AtmoSidecar JSON =="
$BASE "$SD/atmo_sidecar_merge.py" --meta "${SIDECAR_BASE}_meta.json" \
  --coeffs "${SIDECAR_BASE}_coeffs.npz" --out "${SIDECAR_BASE}_sidecar.json"

echo "== 4/4 productionised composite (surface-priors CLI, S2L1CGeeSource) =="
$BASE -m surface_priors.cli build \
  --product-id "l1c-$(basename "$SIDECAR_BASE")" --wgs84-bounds "$X0" "$Y0" "$X1" "$Y1" --resolution "$RES" \
  --gee-product s2_l1c --atmosphere-sidecar "${SIDECAR_BASE}_sidecar.json" --low-aod-frac "$FRAC" \
  --temporal-range "$M0" "$M1" --composite-period "${M0:0:7}" --cache-dir "$CACHE"
echo "done -> per-band GeoTIFFs under $CACHE"
