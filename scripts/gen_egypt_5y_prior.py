"""Generate a 5-year monthly surface prior for the Egypt (Nile Delta) case.

One composite per year (July) for 2020..2024 over the benchmark AOI, written
as a multiband int16 GeoTIFF per period plus a JSON summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Import surface_priors_rs first: its package init pins PROJ_DATA so rasterio's
# GDAL/PROJ no longer clobbers the native WGS84->UTM transform — import order
# is no longer load-bearing, so rasterio can be imported normally up top.
import surface_priors_rs as spx
import rasterio
from affine import Affine
from rasterio.crs import CRS

BBOX = (30.5, 30.5, 31.6, 31.5)  # Nile Delta, WGS84
YEARS = [2020, 2021, 2022, 2023, 2024]
MONTH = 7  # July — the month used across the egypt-case benchmarks
RESOLUTION = 60.0
TOP_K = 6  # 3 left the under-observed SE swath-edge corner empty in 4/5 years;
           # 6 includes the additional clear July passes that cover it.
ENDPOINT = "pc"
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
DISK_CACHE = "/tmp/spx-egypt-5y"
OUT_DIR = Path("egypt_5y_prior")


def _write_geotiffs(results) -> list[dict]:
    summary = []
    for r in results:
        year, month = r["year"], r["month"]
        g = r["grid"]
        h, w = g["height"], g["width"]
        a, b, c, d, e, f = g["transform"]
        transform = Affine(a, b, c, d, e, f)
        crs = CRS.from_epsg(g["epsg"])

        band_names = r["band_names"]
        stack = np.stack([np.asarray(r["bands"][name]) for name in band_names], axis=0)

        out_path = OUT_DIR / f"egypt_{year}-{month:02d}_prior.tif"
        with rasterio.open(
            out_path, "w", driver="GTiff", height=h, width=w,
            count=len(band_names), dtype="int16", crs=crs, transform=transform,
            compress="deflate", predictor=2, tiled=True,
        ) as dst:
            for i, name in enumerate(band_names, start=1):
                dst.write(stack[i - 1], i)
                dst.set_band_description(i, name)
            # Reflectance = DN * 0.0001 (S2 N0400 +1000 offset harmonized in
            # the pipeline). GDAL-aware readers pick these up automatically.
            n = len(band_names)
            dst.scales = [0.0001] * n
            dst.offsets = [0.0] * n
            dst.update_tags(
                reflectance_scale="0.0001",
                reflectance_offset="0.0",
                note="surface reflectance = DN * 0.0001; S2 N0400 +1000 BOA offset harmonized",
            )

        valid = int((np.asarray(r["observation_count"]) > 0).sum())
        rec = {
            "year": year, "month": month, "path": str(out_path),
            "grid": f"{h}x{w}", "epsg": g["epsg"], "resolution": g["resolution"],
            "n_scenes": len(r["source_ids"]),
            "valid_pixels": valid, "total_pixels": h * w,
            "reflectance_scale": 0.0001, "reflectance_offset": 0.0,
            "timings": dict(r["timings"]),
        }
        summary.append(rec)
        print(
            f"  {year}-{month:02d}: {out_path.name}  grid={h}x{w}  "
            f"scenes={rec['n_scenes']}  valid={valid}/{h*w}  "
            f"total={r['timings'].get('total', 0):.2f}s"
        )
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # One batch call: build_monthly_composites shares a single scout pass
    # and overlaps the per-period STAC searches, then composes each period
    # sequentially so the fetches don't oversubscribe the connection pool.
    # ~13s cold for this 5-period Nile Delta batch vs ~15s looping
    # build_composite (and vs ~24s before the py.rs fixes).
    results = spx.build_monthly_composites(
        bbox=BBOX,
        years=YEARS,
        months=[MONTH],
        resolution=RESOLUTION,
        top_k=TOP_K,
        endpoint=ENDPOINT,
        disk_cache=DISK_CACHE,
        bands=BANDS,
    )
    wall = time.time() - t0

    summary = _write_geotiffs(results)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} composites to {OUT_DIR}/  (build wall {wall:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
