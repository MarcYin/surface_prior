"""Clean monthly HLS composites for the Nile Delta, every month 2020-2024.

A cloud-free monthly surface time series (60 composites) to serve as the clean
reference for the SWIR+NIR -> visible prediction experiment. Same AOI/params as
gen_egypt_5y_prior_hls.py but all 12 months x 5 years.
"""
from __future__ import annotations

import calendar
import json
import time
from pathlib import Path

import bestpixel as bp
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

BBOX = (30.5, 30.5, 31.6, 31.5)
YEARS = [2020, 2021, 2022, 2023, 2024]
MONTHS = list(range(1, 13))
RESOLUTION = 60.0
TOP_K = 6
ENDPOINT = "hls"
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
DISK_CACHE = "/tmp/spx-egypt-monthly-hls"
OUT_DIR = Path("egypt_monthly_5y_hls")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Per-(year,month) build; the 12-month batch call 400s on PC, single
    # periods are reliable.
    results = []
    for y in YEARS:
        for m in MONTHS:
            last = calendar.monthrange(y, m)[1]
            dt = f"{y}-{m:02d}-01/{y}-{m:02d}-{last:02d}"
            try:
                r = bp.build_composite(BBOX, dt, resolution=RESOLUTION, top_k=TOP_K,
                                       endpoint=ENDPOINT, disk_cache=DISK_CACHE, bands=BANDS)
                r["year"], r["month"] = y, m
                results.append(r)
            except Exception as e:
                print(f"  {y}-{m:02d}: build failed: {str(e)[:60]}")
    wall = time.time() - t0
    summary = []
    for r in results:
        y, m = r["year"], r["month"]
        g = r["grid"]; h, w = g["height"], g["width"]
        transform = Affine(*g["transform"]); crs = CRS.from_epsg(g["epsg"])
        bn = r["band_names"]
        stack = np.stack([np.asarray(r["bands"][b]) for b in bn], 0)
        out = OUT_DIR / f"egypt_{y}-{m:02d}_hls.tif"
        with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=len(bn),
                           dtype="int16", crs=crs, transform=transform,
                           compress="deflate", predictor=2, tiled=True) as dst:
            for i, b in enumerate(bn, 1):
                dst.write(stack[i-1], i); dst.set_band_description(i, b)
            dst.scales = [1e-4]*len(bn); dst.offsets = [0.0]*len(bn)
        valid = int((np.asarray(r["observation_count"]) > 0).sum())
        summary.append({"year": y, "month": m, "path": str(out),
                        "n_scenes": len(r["source_ids"]), "valid_pixels": valid,
                        "total_pixels": h*w})
        print(f"  {y}-{m:02d}: scenes={len(r['source_ids']):2d} valid={valid}/{h*w} "
              f"({100*valid/(h*w):.1f}%)")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} monthly composites to {OUT_DIR}/ (wall {wall:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
