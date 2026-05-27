"""Smoke-test the bestpixel Python module.

Runs a tiny single-month composite at 60 m and checks that we get
back numpy arrays with the expected dtype/shape — no GeoTIFF writes.
"""
import time

import bestpixel as spx
import numpy as np

# Nile Delta AOI used across the CLI benchmarks — spans multiple MGRS
# tiles, so this exercises the tile-aware selector + multi-source
# fetch path, not just a single-tile fast path.
bbox = (30.5, 30.5, 31.6, 31.5)
datetime_str = "2024-07-01/2024-07-31"

t0 = time.time()
out = spx.build_composite(
    bbox=bbox,
    datetime=datetime_str,
    resolution=60.0,
    top_k=3,
    max_cloud_cover=80.0,
    concurrency=120,
    endpoint="pc",
    disk_cache="/tmp/spx-py-smoke",
)
dt = time.time() - t0

print(f"build_composite OK in {dt:.2f}s")
print(f"keys: {sorted(out.keys())}")

bands = out["bands"]
print(f"bands: {sorted(bands.keys())}")
b = bands["red"]
print(f"red: dtype={b.dtype} shape={b.shape} min={b.min()} max={b.max()} non_nodata={(b != 0).sum()}")

q = out["quality"]
oc = out["observation_count"]
sel = out["selected_observation"]
print(f"quality: dtype={q.dtype} shape={q.shape}")
print(f"observation_count: dtype={oc.dtype} shape={oc.shape} max={oc.max()}")
print(f"selected_observation: dtype={sel.dtype} shape={sel.shape}")

# Sanity: arrays should be numpy ndarrays, not lists.
assert isinstance(b, np.ndarray)
assert b.dtype == np.uint16
assert b.shape == q.shape == oc.shape == sel.shape

# Spot-check that the bands array is contiguous and can flow into
# downstream numpy/SciPy code without a copy.
assert b.flags["C_CONTIGUOUS"], "bands should be C-contiguous for zero-copy interop"

# Demonstrate a downstream computation (NDVI) without touching disk.
nir = bands["nir"].astype(np.float32)
red = bands["red"].astype(np.float32)
mask = (nir > 0) & (red > 0)
ndvi = np.where(mask, (nir - red) / np.maximum(nir + red, 1), np.nan)
print(f"NDVI: valid_px={mask.sum()} median={np.nanmedian(ndvi):.3f}")

# Grid metadata is also surfaced for georeferencing.
print(f"grid: {out['grid']}")
