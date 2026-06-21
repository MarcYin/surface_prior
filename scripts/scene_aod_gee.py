"""Per-scene CAMS + MERRA-2 AOD over an AOI, via the GEE service account.

The bestpixel selection ranks scenes by cloud-free fraction only (S2 SCL has no
aerosol; HLS Fmask carries just a coarse climatology flag). This adds a
quantitative per-acquisition AOD so selection can downweight hazy days for a
cleaner surface prior.

Sources (both, per request):
  CAMS NRT  ECMWF/CAMS/NRT  total_aerosol_optical_depth_at_550nm_surface (~40km, 3h)
  MERRA-2   NASA/GSFC/MERRA/aer/2  TOTEXTTAU (AOT 550nm, ~50km, 1h)

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/scene_aod_gee.py
"""
from __future__ import annotations

import json
import urllib.request

import ee

GEE_KEY = "/home/users/marcyin/gee-service-account.json"
GEE_EMAIL = "python-gee@gee-marc.iam.gserviceaccount.com"
PC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
ES_SEARCH = "https://earth-search.aws.element84.com/v1/search"
CAMS_BAND = "total_aerosol_optical_depth_at_550nm_surface"


def init_ee():
    ee.Initialize(ee.ServiceAccountCredentials(GEE_EMAIL, GEE_KEY))


def scene_aod(scene_times, aoi_bbox, cams_win_h=3, merra_win_h=1):
    """scene_times: list of (scene_id, iso_datetime). Returns
    {scene_id: {"cams": aod|None, "merra": aod|None}} over aoi_bbox."""
    cams = ee.ImageCollection("ECMWF/CAMS/NRT").select(CAMS_BAND)
    merra = ee.ImageCollection("NASA/GSFC/MERRA/aer/2").select("TOTEXTTAU")
    geom = ee.Geometry.Rectangle(list(aoi_bbox))
    feats = []
    for sid, dt in scene_times:
        t = ee.Date(dt)
        cimg = cams.filterDate(t.advance(-cams_win_h, "hour"), t.advance(cams_win_h, "hour")).mean()
        mimg = merra.filterDate(t.advance(-merra_win_h, "hour"), t.advance(merra_win_h, "hour")).mean()
        feats.append(ee.Feature(geom.centroid(1), {
            "sid": sid,
            "cams": cimg.reduceRegion(ee.Reducer.mean(), geom, 40000, bestEffort=True).get(CAMS_BAND),
            "merra": mimg.reduceRegion(ee.Reducer.mean(), geom, 50000, bestEffort=True).get("TOTEXTTAU"),
        }))
    fc = ee.FeatureCollection(feats).getInfo()
    out = {}
    for f in fc["features"]:
        p = f["properties"]
        out[p["sid"]] = {"cams": p.get("cams"), "merra": p.get("merra")}
    return out


def search_scenes(collections, bbox, datetime_range, limit=400, retries=4, stac_url=PC_SEARCH):
    # normalize "YYYY-MM-DD/YYYY-MM-DD" -> RFC3339 (earth-search requires it; PC accepts it)
    if "T" not in datetime_range and "/" in datetime_range:
        s, e = datetime_range.split("/")
        datetime_range = f"{s}T00:00:00Z/{e}T23:59:59Z"
    body = {"collections": collections, "bbox": list(bbox),
            "datetime": datetime_range, "limit": limit}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(stac_url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                feats = json.load(r)["features"]
            return [(f["id"], f["properties"]["datetime"]) for f in feats]
        except Exception as e:  # transient 5xx / timeouts from PC STAC
            last = e
            import time
            time.sleep(3 * (attempt + 1))
    raise last


def main():
    init_ee()
    BBOX = (30.5, 30.5, 31.6, 31.5)
    for label, collections in [("HLS", ["hls2-s30", "hls2-l30"]),
                               ("S2", ["sentinel-2-l2a"])]:
        scenes = search_scenes(collections, BBOX, "2022-07-01/2022-07-31")
        aod = scene_aod(scenes, BBOX)
        rows = []
        for sid, dt in scenes:
            a = aod.get(sid, {})
            c, m = a.get("cams"), a.get("merra")
            mean = None if (c is None or m is None) else (c + m) / 2
            rows.append((dt[:10], sid.split(".")[-2] if "." in sid else sid[:24], c, m, mean))
        rows.sort(key=lambda r: (r[4] is None, r[4] or 0))
        print(f"\n=== {label}: {len(scenes)} scenes, July 2022, Nile Delta (sorted by mean AOD) ===")
        print(f"  {'date':10} {'scene':26} {'CAMS':>6} {'MERRA':>6} {'mean':>6}")
        vals = []
        for d, s, c, m, mn in rows:
            cs = f"{c:.3f}" if c is not None else "  -  "
            ms = f"{m:.3f}" if m is not None else "  -  "
            mns = f"{mn:.3f}" if mn is not None else "  -  "
            print(f"  {d:10} {s:26} {cs:>6} {ms:>6} {mns:>6}")
            if mn is not None:
                vals.append(mn)
        if vals:
            import statistics as st
            print(f"  -> AOD mean {st.mean(vals):.3f}, range {min(vals):.3f}-{max(vals):.3f}, "
                  f"n>0.2: {sum(v > 0.2 for v in vals)}/{len(vals)}")


if __name__ == "__main__":
    main()
