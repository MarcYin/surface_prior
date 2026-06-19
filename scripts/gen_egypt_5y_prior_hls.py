"""Generate a 5-year monthly HLS surface prior for the Egypt (Nile Delta) case.

HLS variant of `gen_egypt_5y_prior.py`: one composite per year (July) for
2020..2024 over the benchmark AOI, built from the Harmonized Landsat-Sentinel-2
pool (HLS v2.0 `hls2-l30` + `hls2-s30`) instead of raw Sentinel-2 L2A. Written
as a multiband int16 GeoTIFF per period plus a JSON summary, to a separate
output dir so it sits alongside (and stays comparable to) the S2 product.

HLS vs S2 to keep in mind when reading these:
  - HLS pools Landsat OLI + Sentinel-2 MSI, so ~2x the observations/pixel and
    gap-free coverage, but 30 m native (vs S2 10-20 m) and only the 7 harmonized
    bands.
  - HLS is NBAR (nadir BRDF-adjusted, LaSRC atmospheric correction); S2 L2A is
    directional BOA (Sen2Cor). HLS reads ~0.01-0.02 reflectance lower across
    bands, and coastal/blue diverge most. Same DN*0.0001 reflectance scale, so
    numerically aligned but NOT a drop-in substitute for S2 in those bands.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Import bestpixel first: its package init pins PROJ_DATA so rasterio's
# GDAL/PROJ no longer clobbers the native WGS84->UTM transform — import order
# is no longer load-bearing, so rasterio can be imported normally up top.
import bestpixel as bp
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

BBOX = (30.5, 30.5, 31.6, 31.5)  # Nile Delta, WGS84
YEARS = [2020, 2021, 2022, 2023, 2024]
MONTH = 7  # July — the month used across the egypt-case benchmarks
RESOLUTION = 60.0
TOP_K = 6  # match the S2 product; HLS covers the SE swath-edge corner easily.
ENDPOINT = "hls"  # HLS v2.0: hls2-l30 + hls2-s30 harmonized pool
# HLS exposes exactly the 7 harmonized NBAR bands; this is the full set.
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
# Fresh cache dir: COG headers cached before bestpixel 0.1.2 recorded HLS's
# signed Int16 bands as UInt16; a dedicated dir avoids any stale entry.
DISK_CACHE = "/tmp/spx-egypt-5y-hls"
OUT_DIR = Path("egypt_5y_prior_hls")


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
            # Reflectance = DN * 0.0001. HLS carries no S2 N0400-style offset;
            # signed Int16 fill/negatives are clamped to 0 at fetch.
            n = len(band_names)
            dst.scales = [0.0001] * n
            dst.offsets = [0.0] * n
            dst.update_tags(
                reflectance_scale="0.0001",
                reflectance_offset="0.0",
                source="HLS v2.0 (hls2-l30 + hls2-s30), NBAR / LaSRC",
                note="surface reflectance = DN * 0.0001; HLS NBAR, no N0400 offset",
            )

        valid = int((np.asarray(r["observation_count"]) > 0).sum())
        rec = {
            "year": year, "month": month, "path": str(out_path),
            "grid": f"{h}x{w}", "epsg": g["epsg"], "resolution": g["resolution"],
            "endpoint": ENDPOINT, "collection": r.get("collection"),
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
    results = bp.build_monthly_composites(
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
    print(f"\nwrote {len(summary)} HLS composites to {OUT_DIR}/  (build wall {wall:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
