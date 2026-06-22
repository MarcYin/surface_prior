"""MAIAC-select + 6S-correct composite — Phase 1: select days, sample pixels.

Over an AOI/month: list S2 days, get per-day MAIAC AOD + geometry + scene WVP +
AERONET truth, rank by MAIAC, keep the lowest `--frac`, and sample co-located
L1C TOA / L2A SR / per-pixel WVP / Cloud Score+ on a fixed grid across the
selected days. Writes <out>_meta.json and <out>_pixels.npz.

  python scripts/custom_ac_phase1.py --site 31.290 30.081 --bbox 31.07 29.90 31.51 30.26 \
      --month 2022-07-01 2022-08-01 --aeronet Cairo_EMA_2 --frac 0.6 --out /tmp/cac
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request

import ee
import numpy as np

B = ["B1", "B2", "B3", "B4", "B8A", "B11", "B12"]


def init():
    ee.Initialize(ee.ServiceAccountCredentials(
        "python-gee@gee-marc.iam.gserviceaccount.com", "/home/users/marcyin/gee-service-account.json"))


def aeronet_timeseries(site, y0, m0, y1, m1):
    url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site={site}"
           f"&year={y0}&month={m0}&day=1&year2={y1}&month2={m1}&day2=28&AOD15=1&AVG=10")
    L = urllib.request.urlopen(url, timeout=60).read().decode("latin-1").splitlines()
    hi = next(i for i, l in enumerate(L) if "AOD_500nm" in l)
    H = L[hi].split(",")
    cd, ct, c5, ca = (H.index(x) for x in
                      ["Date(dd:mm:yyyy)", "Time(hh:mm:ss)", "AOD_500nm", "440-870_Angstrom_Exponent"])
    A = []
    for l in L[hi + 1:]:
        p = l.split(",")
        if len(p) <= ca:
            continue
        try:
            a5, ang = float(p[c5]), float(p[ca])
        except ValueError:
            continue
        if a5 <= -9:
            continue
        d, mo, y = p[cd].split(":"); h, mi, s = p[ct].split(":")
        A.append((dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s)), a5 * (550 / 500.) ** (-ang)))
    return A


def aeronet_at(A, t):
    if not A:
        return None
    b = min(A, key=lambda x: abs((x[0] - t).total_seconds()))
    return round(b[1], 3) if abs((b[0] - t).total_seconds()) < 3600 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=float, nargs=2, required=True, help="lon lat of AERONET/centre")
    ap.add_argument("--bbox", type=float, nargs=4, required=True)
    ap.add_argument("--month", nargs=2, required=True, help="start end ISO dates")
    ap.add_argument("--aeronet", required=True)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--grid", type=int, nargs=2, default=[36, 30])
    ap.add_argument("--out", default="/tmp/cac")
    a = ap.parse_args()
    init()
    site = a.site; AOI = ee.Geometry.Point(site).buffer(20000).bounds()
    bb = a.bbox
    y0, m0 = a.month[0][:4], int(a.month[0][5:7]); y1, m1 = a.month[1][:4], int(a.month[1][5:7])
    AER = aeronet_timeseries(a.aeronet, y0, m0, y1, m1)

    nx, ny = a.grid
    lons = np.linspace(bb[0] + 0.01, bb[2] - 0.01, nx); lats = np.linspace(bb[1] + 0.01, bb[3] - 0.01, ny)
    feats = [ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"pid": int(i)})
             for i, (la, lo) in enumerate((p, q) for p in lats for q in lons)]
    PTS = ee.FeatureCollection(feats); NPIX = len(feats)

    sc = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(AOI).filterDate(*a.month)
    ids = sc.aggregate_array("system:index").getInfo()
    days = {}
    for sid in ids:
        img = ee.Image(sc.filter(ee.Filter.eq("system:index", sid)).first())
        t = dt.datetime.fromtimestamp(img.get("system:time_start").getInfo() / 1000, dt.timezone.utc).replace(tzinfo=None)
        day = t.strftime("%Y-%m-%d")
        if day in days:
            continue
        tile = sid.split("_")[-1][1:]; t1 = (t + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        maiac = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").filterDate(day, t1)
                 .select("Optical_Depth_055").mean().multiply(0.001)
                 .reduceRegion(ee.Reducer.mean(), AOI, 1000, bestEffort=True).get("Optical_Depth_055"))
        l2a = ee.Image(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI)
                       .filterDate(day, t1).filter(ee.Filter.eq("MGRS_TILE", tile)).first())
        info = ee.Dictionary({
            "maiac": maiac, "wvp": l2a.select("WVP").reduceRegion(ee.Reducer.mean(), AOI, 500, bestEffort=True).get("WVP"),
            "sza": img.get("MEAN_SOLAR_ZENITH_ANGLE"), "saa": img.get("MEAN_SOLAR_AZIMUTH_ANGLE"),
            "vza": img.get("MEAN_INCIDENCE_ZENITH_ANGLE_B4"), "vaa": img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B4")}).getInfo()
        if info["maiac"] is None or info["wvp"] is None:
            continue
        days[day] = {"sysidx": sid, "tile": tile, "datetime": t.strftime("%Y-%m-%dT%H:%M"),
                     "maiac": info["maiac"], "wvp": info["wvp"] / 1000.0,
                     "sza": info["sza"], "saa": info["saa"], "vza": info["vza"], "vaa": info["vaa"],
                     "aeronet_op": aeronet_at(AER, t)}

    items = sorted(days.items(), key=lambda kv: kv[1]["maiac"])
    k = int(round(a.frac * len(items))); selected = items[:k]
    print(f"{len(items)} S2 days; selecting lowest {k} by MAIAC AOD (frac {a.frac}):")
    cube = np.full((k, NPIX, 16), np.nan)               # toa7, sr7, wvp, cs
    meta = []
    for i, (day, d) in enumerate(selected):
        l1c = ee.Image(ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filter(ee.Filter.eq("system:index", d["sysidx"])).first())
        t1 = (dt.datetime.strptime(day, "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        l2a = ee.Image(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI)
                       .filterDate(day, t1).filter(ee.Filter.eq("MGRS_TILE", d["tile"])).first())
        csp = ee.Image(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filter(ee.Filter.eq("system:index", d["sysidx"])).first())
        stack = (l1c.select(B, [f"t{b}" for b in B]).divide(10000)
                 .addBands(l2a.select(B, [f"s{b}" for b in B]).divide(10000))
                 .addBands(l2a.select("WVP").divide(1000).rename("wvp")).addBands(csp.select("cs")))
        fc = stack.sampleRegions(collection=PTS, properties=["pid"], scale=60, geometries=False).getInfo()
        cols = [f"t{b}" for b in B] + [f"s{b}" for b in B] + ["wvp", "cs"]
        for f in fc["features"]:
            p = f["properties"]; cube[i, p["pid"]] = [p.get(c, np.nan) for c in cols]
        nclear = int((cube[i, :, -1] > 0.6).sum())
        print(f"  {day}  MAIAC {d['maiac']:.3f}  AERONET {str(d['aeronet_op']):>5}  wvp {d['wvp']:.2f}cm  clear {nclear}/{NPIX}")
        meta.append({"day": day, **d})
    json.dump({"selected": meta, "rejected": [k2 for k2, _ in items[k:]], "bands": B, "npix": NPIX}, open(f"{a.out}_meta.json", "w"))
    np.savez(f"{a.out}_pixels.npz", cube=cube)
    print(f"saved {a.out}_meta.json + {a.out}_pixels.npz  cube {cube.shape}")


if __name__ == "__main__":
    main()
