"""Generate the Egypt 5-year July prior with the adaptive+windowed pipeline.

Same AOI/bands/period as scripts/gen_egypt_5y_prior.py but using the
fix/build-monthly-composites-perf selection path: per-chunk SCL-driven
adaptive depth, tile-aware selection, full-observer preference, and
Level-2 windowed fetch (coverage_target/min_k/max_k/windowed_fetch).

Writes one multiband int16 GeoTIFF per year plus an RGB quicklook PNG and
a JSON summary to OUT_DIR (which can be copied into the public folder).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Import bestpixel first: its package init pins PROJ_DATA so rasterio's
# GDAL/PROJ no longer clobbers the native WGS84->UTM transform — import order
# is no longer load-bearing, so rasterio/PIL can be imported normally up top.
import bestpixel as bp
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.crs import CRS

BBOX = (30.5, 30.5, 31.6, 31.5)  # Nile Delta, WGS84
YEARS = [2020, 2021, 2022, 2023, 2024]
MONTH = 7
RESOLUTION = 60.0
ENDPOINT = "pc"
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
DISK_CACHE = "/tmp/spx-egypt-5y-adaptive"
OUT_DIR = Path("egypt_5y_prior_adaptive")
# fix-branch selection: adaptive depth + windowed fetch (full-observer
# preference is on by default).
SEL = {"coverage_target": 0.95, "min_k": 2, "max_k": 6, "windowed_fetch": True}


# Natural-colour: DN -> reflectance, one fixed gain across all channels
# (preserves colour balance; a per-band stretch distorts it) + optional gamma.
REFL_SCALE = 1e-4
GAIN = 3.5
GAMMA = 1.0


def _write_preview(r, prev_dir: Path) -> None:
    bd = r["bands"]
    rr, gg, bb = (np.asarray(bd[c]) for c in ("red", "green", "blue"))
    valid = np.asarray(r["observation_count"]) > 0
    x = np.dstack([rr, gg, bb]).astype(np.float32) * REFL_SCALE * GAIN
    x = np.clip(x, 0.0, 1.0)
    if GAMMA != 1.0:
        x = x ** (1.0 / GAMMA)
    rgb = (x * 255).astype(np.uint8)
    rgb[~valid] = 0
    out = prev_dir / f"egypt_{r['year']}-{r['month']:02d}_prior_rgb.png"
    Image.fromarray(rgb, "RGB").save(out, optimize=True)


def _write_outputs(results) -> list[dict]:
    prev_dir = OUT_DIR / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in results:
        year, month = r["year"], r["month"]
        g = r["grid"]
        h, w = g["height"], g["width"]
        transform = Affine(*g["transform"])
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
            # Reflectance = DN * scale + offset = DN * 0.0001 (S2 N0400
            # +1000 offset already harmonized in the pipeline). GDAL-aware
            # readers (rasterio, stackstac, QGIS) pick these up automatically.
            n = len(band_names)
            dst.scales = [0.0001] * n
            dst.offsets = [0.0] * n
            dst.update_tags(
                reflectance_scale="0.0001",
                reflectance_offset="0.0",
                note="surface reflectance = DN * 0.0001; S2 N0400 +1000 BOA offset harmonized",
            )

        _write_preview(r, prev_dir)
        valid = int((np.asarray(r["observation_count"]) > 0).sum())
        summary.append({
            "year": year, "month": month, "path": str(out_path),
            "grid": f"{h}x{w}", "epsg": g["epsg"], "n_scenes": len(r["source_ids"]),
            "valid_pixels": valid, "total_pixels": h * w,
            "reflectance_scale": 0.0001, "reflectance_offset": 0.0,
            "selection": SEL, "timings": dict(r["timings"]),
        })
        print(
            f"  {year}-{month:02d}: scenes={len(r['source_ids']):2d} "
            f"valid={valid}/{h*w} ({valid/(h*w)*100:.1f}%) "
            f"total={r['timings'].get('total', 0):.2f}s"
        )
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Egypt 5y adaptive prior  AOI={BBOX}  years={YEARS} month={MONTH}  sel={SEL}")
    t0 = time.time()
    results = bp.build_monthly_composites(
        bbox=BBOX, years=YEARS, months=[MONTH], resolution=RESOLUTION,
        endpoint=ENDPOINT, disk_cache=DISK_CACHE, bands=BANDS, **SEL,
    )
    wall = time.time() - t0
    summary = _write_outputs(results)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} composites + previews to {OUT_DIR}/  (build wall {wall:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
