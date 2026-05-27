"""Make true-colour RGB preview PNGs for the Egypt 5-year composites.

Reads each multiband GeoTIFF in egypt_5y_prior/, stacks red/green/blue,
applies a per-band 2-98 percentile stretch over valid pixels, and writes a
downsampled PNG quicklook. Output dir is configurable so the previews can
be dropped into a shared/public folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# Band order in the composite GeoTIFFs (1-indexed):
# coastal=1 blue=2 green=3 red=4 nir=5 swir16=6 swir22=7
RGB = {"red": 4, "green": 3, "blue": 2}
SRC_DIR = Path("egypt_5y_prior")


# Natural-colour rendering: convert DN -> surface reflectance and apply a
# single fixed gain across all three channels so the inter-band colour
# balance is preserved (a per-band stretch distorts the colours). Fixed
# (not per-image) so the years look consistent.
REFL_SCALE = 1e-4  # DN -> reflectance (S2 quantification 10000)
GAIN = 3.5
GAMMA = 1.0


def natural_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.dstack([r, g, b]).astype(np.float32) * REFL_SCALE * GAIN
    x = np.clip(x, 0.0, 1.0)
    if GAMMA != 1.0:
        x = x ** (1.0 / GAMMA)
    rgb = (x * 255).astype(np.uint8)
    rgb[~valid] = 0  # nodata -> black
    return rgb


def make_preview(tif: Path, out_dir: Path, max_px: int) -> Path:
    with rasterio.open(tif) as ds:
        # Decimated read for a lightweight quicklook.
        dec = max(1, int(max(ds.height, ds.width) / max_px))
        oh, ow = ds.height // dec, ds.width // dec
        r = ds.read(RGB["red"], out_shape=(oh, ow))
        g = ds.read(RGB["green"], out_shape=(oh, ow))
        b = ds.read(RGB["blue"], out_shape=(oh, ow))
    valid = (r > 0) | (g > 0) | (b > 0)
    rgb = natural_rgb(r, g, b, valid)
    out_path = out_dir / (tif.stem + "_rgb.png")
    Image.fromarray(rgb, "RGB").save(out_path, optimize=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=SRC_DIR, help="Dir of egypt_*_prior.tif composites.")
    ap.add_argument("--out-dir", type=Path, default=None, help="PNG output dir (default <src>/previews).")
    ap.add_argument("--max-px", type=int, default=1200, help="Longest side of the PNG.")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.src_dir / "previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = sorted(args.src_dir.glob("egypt_*_prior.tif"))
    if not tifs:
        raise SystemExit(f"no composites found in {args.src_dir}/")
    args.out_dir = out_dir
    for tif in tifs:
        p = make_preview(tif, args.out_dir, args.max_px)
        kb = p.stat().st_size / 1024
        print(f"  {p}  ({kb:.0f} KB)")
    print(f"\nwrote {len(tifs)} RGB previews to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
