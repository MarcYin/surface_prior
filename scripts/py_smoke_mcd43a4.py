"""Smoke-test the MCD43A4 endpoint.

Outputs are in MODIS Sinusoidal, not UTM — the grid dict reports
proj4 instead of a real EPSG.
"""
import time

import bestpixel as spx
import numpy as np

t0 = time.time()
out = spx.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=500.0,
    top_k=3,
    endpoint="mcd43a4",
    disk_cache="/tmp/spx-mcd43a4",
)
dt = time.time() - t0
print(f"MCD43A4 OK in {dt:.2f}s")
print(f"collection: {out['collection']}")
print(f"grid: {out['grid']}")
print(f"band_names: {out['band_names']}")
print(f"source scenes: {len(out['source_ids'])}")
print(f"partition tiles: {out['partition_tiles']}")
for k, v in out["bands"].items():
    valid = (v != 65535).sum()
    mn = v[v != 65535].min() if valid else 0
    mx = v[v != 65535].max() if valid else 0
    print(f"  {k}: shape={v.shape} dtype={v.dtype} valid_px={valid} min={mn} max={mx}")
q = out["quality"]
print("quality histogram:", dict(zip(*[a.tolist() for a in np.unique(q, return_counts=True)])))
print("obs_count max:", out["observation_count"].max())
