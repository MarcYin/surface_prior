"""Merge meta + 6S coefficients into the Rust AtmoSidecar JSON.

Reads `<out>_meta.json` (build_atmo_sidecar) and `<out>_coeffs.npz`
(custom_ac_phase2_sixs) and writes `<out>_sidecar.json` keyed by L1C STAC item
id, in the format `surface_priors_rs::atcorr::AtmoSidecar` deserialises.

  python scripts/atmo_sidecar_merge.py --meta /tmp/sidecar_meta.json \
      --coeffs /tmp/sidecar_coeffs.npz --out /tmp/sidecar_sidecar.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# crate canonical band names for the 6S band order [B1,B2,B3,B4,B8A,B11,B12]
BANDS = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.meta))["selected"]
    Z = np.load(a.coeffs)
    co = Z["coeffs"]                 # (days, [maiac,aeronet], ntcwv, 3, 7)
    tcwv = [float(x) for x in Z["tcwv"]]
    scenes = {}
    for i, d in enumerate(meta):
        c = co[i, 0]                # MAIAC-AOD coefficients: (ntcwv, 3, 7)
        scenes[d["l1c_id"]] = {
            "maiac_aod": float(d["maiac"]),
            "wvp": float(d["wvp"]),
            "tcwv_nodes": tcwv,
            "xap": [[float(v) for v in c[n, 0]] for n in range(len(tcwv))],
            "xbp": [[float(v) for v in c[n, 1]] for n in range(len(tcwv))],
            "xcp": [[float(v) for v in c[n, 2]] for n in range(len(tcwv))],
        }
    json.dump({"bands": BANDS, "scenes": scenes}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(scenes)} scenes, bands {BANDS}")


if __name__ == "__main__":
    main()
