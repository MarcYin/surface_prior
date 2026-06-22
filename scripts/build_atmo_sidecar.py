"""Atmosphere sidecar generator for the L1C custom-AC composite (production).

Enumerates earth-search `sentinel-2-l1c` scenes for an AOI/period (the same
STAC items the Rust crate fetches), then for each scene pulls MAIAC AOD550 +
scene water vapour + sun/view geometry from GEE. Writes `<out>_meta.json` in
the format consumed by `custom_ac_phase2_sixs.py` (6S coefficients), which is
then merged into the Rust `AtmoSidecar` JSON by `atmo_sidecar_merge.py`.

  python scripts/build_atmo_sidecar.py --bbox 31.0 29.9 31.5 30.3 \
      --month 2022-07-01 2022-08-01 --out /tmp/sidecar
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import ee

from scene_aod_gee import ES_SEARCH, search_scenes


def init():
    ee.Initialize(ee.ServiceAccountCredentials(
        "python-gee@gee-marc.iam.gserviceaccount.com", "/home/users/marcyin/gee-service-account.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True)
    ap.add_argument("--month", nargs=2, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    init()
    bb = a.bbox
    AOI = ee.Geometry.Rectangle(bb)
    # enumerate the exact earth-search L1C items the crate will fetch
    scenes = search_scenes(["sentinel-2-l1c"], bb, f"{a.month[0]}/{a.month[1]}", stac_url=ES_SEARCH)
    print(f"{len(scenes)} earth-search L1C scenes")
    out = []
    seen = set()
    for sid, isodt in scenes:
        # id e.g. S2A_36RTU_20220710_0_L1C  -> tile 36RTU, date 20220710
        parts = sid.split("_")
        tile, ymd = parts[1], parts[2]
        day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        if (tile, day) in seen:
            continue
        seen.add((tile, day))
        t1 = (dt.datetime.strptime(day, "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        gee = ee.Image(ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(AOI)
                       .filterDate(day, t1).filter(ee.Filter.eq("MGRS_TILE", tile)).first())
        maiac = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").filterDate(day, t1)
                 .select("Optical_Depth_055").mean().multiply(0.001)
                 .reduceRegion(ee.Reducer.mean(), AOI, 1000, bestEffort=True).get("Optical_Depth_055"))
        wvp = (ee.Image(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI)
               .filterDate(day, t1).filter(ee.Filter.eq("MGRS_TILE", tile)).first()).select("WVP")
               .reduceRegion(ee.Reducer.mean(), AOI, 500, bestEffort=True).get("WVP"))
        info = ee.Dictionary({
            "maiac": maiac, "wvp": wvp,
            "sza": gee.get("MEAN_SOLAR_ZENITH_ANGLE"), "saa": gee.get("MEAN_SOLAR_AZIMUTH_ANGLE"),
            "vza": gee.get("MEAN_INCIDENCE_ZENITH_ANGLE_B4"), "vaa": gee.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B4"),
        }).getInfo()
        if info["maiac"] is None or info["wvp"] is None or info["sza"] is None:
            print(f"  skip {sid} (missing aux)")
            continue
        out.append({"l1c_id": sid, "day": day, "sysidx": sid, "tile": tile,
                    "datetime": isodt[:16], "maiac": info["maiac"], "wvp": info["wvp"] / 1000.0,
                    "sza": info["sza"], "saa": info["saa"], "vza": info["vza"], "vaa": info["vaa"],
                    "aeronet_op": None})
        print(f"  {sid}  MAIAC {info['maiac']:.3f}  wvp {info['wvp']/1000:.2f}cm")
    json.dump({"selected": out, "bands": ["B1", "B2", "B3", "B4", "B8A", "B11", "B12"]},
              open(f"{a.out}_meta.json", "w"))
    print(f"saved {a.out}_meta.json ({len(out)} scenes)")


if __name__ == "__main__":
    main()
