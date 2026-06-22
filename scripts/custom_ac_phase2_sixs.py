"""MAIAC-select + 6S-correct composite — Phase 2 (SIAC rt6s env).

Per-day 6S coefficients for MAIAC and AERONET AOD, over a TCWV LUT so Phase 3
can interpolate the correction by per-pixel water vapour. Uses the cached native
6S build (auto_build=False) and S2A RSRs from the spectral-library schema.

  PYTHONPATH=~/SIAC/python ~/SIAC/.pixi/envs/rt6s/bin/python \
      scripts/custom_ac_phase2_sixs.py --meta /tmp/cac_meta.json --out /tmp/cac_coeffs.npz
"""
import argparse
import json
from datetime import datetime

import numpy as np
import xarray as xr
from siac.algorithms.rt.direct.sixs import SixSBackend
from siac.config import SixSAlgorithmConfig
from siac.domain.sensors import SensorBand
from siac.runtime import AtmosphericState, GeometryAngles

MOD = "/home/users/marcyin/.cache/siac/rt6s/release/_siac_rt6s_native.cpython-311-x86_64-linux-gnu.so"
SCHEMA = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3/sensor_schema.json"
TCWV_LUT = [0.5, 1.5, 2.5, 3.5, 4.5]          # cm precipitable water
ORDER = ["ultra_blue", "blue", "green", "red", "nir", "swir1", "swir2"]
CENT = {"ultra_blue": 443, "blue": 490, "green": 560, "red": 665, "nir": 865, "swir1": 1610, "swir2": 2190}


def field(v):
    return xr.DataArray(np.array([[float(v)]], np.float32), dims=("y", "x"))


def build_bands():
    s2 = next(s for s in json.load(open(SCHEMA))["sensors"] if "sentinel-2a" in str(s).lower())
    bd = {b["band_id"]: b for b in s2["bands"]}
    out = []
    for i, bid in enumerate(ORDER):
        rd = bd[bid]["response_definition"]
        out.append(SensorBand(name=bid, center_wavelength=float(CENT[bid]), bandwidth=30.0, resolution=20.0,
                              band_index=i, rsrf_wavelengths_nm=np.array(rd["wavelength_nm"], float),
                              rsrf_response=np.array(rd["response"], float)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bands = build_bands()
    cfg = SixSAlgorithmConfig(build_profile="release", output_variables=("xap", "xbp", "xcp"),
                              native_threads=2).model_copy(update={"module_path": MOD, "auto_build": False})
    backend = SixSBackend(sixs_config=cfg); backend.set_observation_time(datetime(2022, 7, 15, 8, 45))
    meta = json.load(open(a.meta))["selected"]
    # day, [maiac,aeronet], tcwv-node, [xap,xbp,xcp], band
    out = np.full((len(meta), 2, len(TCWV_LUT), 3, 7), np.nan)
    for di, d in enumerate(meta):
        geom = GeometryAngles.from_degrees(field(d["sza"]), field(d["saa"]), field(d["vza"]), field(d["vaa"]))
        aods = [d["maiac"], d["aeronet_op"] if d["aeronet_op"] is not None else d["maiac"]]
        for ai, aot in enumerate(aods):
            for wi, w in enumerate(TCWV_LUT):
                atm = AtmosphericState(aot=field(aot), tcwv=field(w), tco3=field(0.30),
                                       aot_unc=field(0.02), tcwv_unc=field(0.1), tco3_unc=field(0.01), elevation=field(0.05))
                cs = backend.compute_coefficients_multi(geom, atm, bands)
                for vi, v in enumerate(["xap", "xbp", "xcp"]):
                    out[di, ai, wi, vi] = [float(c.get_output(v).values.ravel()[0]) for c in cs]
        print(f"{d['day']}: 6S LUT done (aot {aods[0]:.3f}/{aods[1]:.3f} x {len(TCWV_LUT)} tcwv)")
    np.savez(a.out, coeffs=out, bands=ORDER, tcwv=np.array(TCWV_LUT))
    print(f"saved {a.out} {out.shape}")


if __name__ == "__main__":
    main()
