"""Smoke-test MCD43A4 with output_crs="utm" — reproject sinusoidal → UTM on the fly."""
import time

import bestpixel as bp

BBOX = (30.5, 30.5, 31.6, 31.5)

print("=== MCD43A4, output_crs='native' (sinusoidal) ===")
t0 = time.time()
native = bp.build_composite(
    bbox=BBOX,
    datetime="2024-07-01/2024-07-31",
    resolution=500.0,
    top_k=3,
    endpoint="mcd43a4",
    disk_cache="/tmp/spx-mcd43a4-utm",
    output_crs="native",
)
print(f"  {time.time() - t0:.2f}s  grid={native['grid']['width']}x{native['grid']['height']}  "
      f"crs={native['grid']['crs'][:40]}...  red dtype={native['bands']['red'].dtype}")

print("=== MCD43A4, output_crs='utm' (reproject) ===")
t0 = time.time()
utm = bp.build_composite(
    bbox=BBOX,
    datetime="2024-07-01/2024-07-31",
    resolution=500.0,
    top_k=3,
    endpoint="mcd43a4",
    disk_cache="/tmp/spx-mcd43a4-utm",
    output_crs="utm",
)
print(f"  {time.time() - t0:.2f}s  grid={utm['grid']['width']}x{utm['grid']['height']}  "
      f"crs={utm['grid']['crs']}  red dtype={utm['bands']['red'].dtype}")

# Compare aggregate stats — values should be in the same range
# (~10000 = reflectance 1.0).
for b in ["red", "nir", "swir16"]:
    n = native["bands"][b]
    u = utm["bands"][b]
    n_valid = n[n != 65535]
    u_valid = u[u != 65535]
    print(
        f"  {b}: native mean={n_valid.mean():.0f} std={n_valid.std():.0f}  "
        f"utm mean={u_valid.mean():.0f} std={u_valid.std():.0f}"
    )

print()
print("=== sanity: UTM bounds and EPSG ===")
print(f"native bounds: {native['grid']['bounds']}")
print(f"utm bounds:    {utm['grid']['bounds']}")
print(f"utm epsg:      {utm['grid']['epsg']}")
assert utm["grid"]["epsg"] > 30000, "expected UTM EPSG"
assert native["grid"]["epsg"] == 0, "native should be non-EPSG sinusoidal"
