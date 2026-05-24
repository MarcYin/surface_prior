# surface-priors-rs

Production Rust port of the surface-priors monthly composite pipeline,
with optional Python bindings that return numpy arrays directly (no
GeoTIFF writes). Supports three STAC sources out of the box:

| Endpoint | Collection(s) | Bands | Quality mask |
|---|---|---|---|
| `pc` (default) | Sentinel-2 L2A | 12 (full S2 set) | SCL |
| `earth-search` | Sentinel-2 L2A | 12 (full S2 set) | SCL |
| `hls` | HLS L30 + S30 (combined harmonized pool) | 7 common bands | Fmask |

## What it does

- Asynchronously fetches scenes from Planetary Computer (default), the
  Element84 earth-search S2 L2A mirror, or PC's Harmonized Landsat-
  Sentinel-2 (HLS v2.0) via STAC search + range-request COG reads.
- Selects the top-k clearest observations per MGRS-tile-aware chunk
  using a coarse quality scout pass (SCL classes for S2 L2A; Fmask
  bit-flags for HLS).
- Decodes DEFLATE-compressed COG tiles via libdeflate, undoes
  TIFF predictor=2, and reprojects to a user-specified UTM grid with
  an AVX2/FMA bilinear resampler.
- Composes a best-pixel monthly composite ranked by per-pixel quality.

For a 100 × 100 km AOI at 60 m resolution, top-k=3 from Planetary
Computer, on a single 16-core Zen 4 node:

| Workload | Time |
|---|---|
| 1 year × 1 month | ~2 s |
| 5 years × 1 month, sequential | ~11 s |
| 5 years × 1 month, 5-thread parallel | ~6 s |

The 5-way parallel floor is set by network throughput to PC, not CPU.

## Python support

Built as an [abi3](https://docs.python.org/3/c-api/stable.html) wheel
against the Python 3.9 stable ABI, so a single wheel installs and
runs on CPython 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14.

## Use from Python

Install from a prebuilt wheel (attached to GitHub Releases):

```bash
pip install <wheel-url-from-release>
```

Or build locally with maturin:

```bash
pip install maturin
cd surface_priors_rs
maturin develop --release --features python
```

Then:

```python
import surface_priors_rs as spx

out = spx.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),         # west, south, east, north (WGS84)
    datetime="2024-07-01/2024-07-31",      # STAC datetime range
    resolution=60.0,                        # metres
    top_k=3,                                # observations per chunk
    endpoint="pc",                          # "pc" | "earth-search" | "auto"
    bands=["coastal", "blue", "green", "red", "nir", "swir16", "swir22"],
)

red = out["bands"]["red"]                   # uint16 ndarray, (H, W)
quality = out["quality"]                    # uint16, 0=clear, 1=marginal, 2=dark, 65535=nodata
print(out["grid"])                          # bounds, epsg, transform — for georeferencing
```

Available band names (stable across endpoints):
`coastal, blue, green, red, rededge1, rededge2, rededge3, nir, nir08,
nir09, swir16, swir22`. SCL / Fmask is consumed internally to derive
the `quality` raster — kept as a discrete class label (nearest
resampling) all the way through, so quality buckets stay categorical.

Pass `bands=None` (or omit) to fetch all bands the endpoint supports
(12 for S2 L2A, 7 for HLS).

### Harmonized Landsat-Sentinel-2 (HLS)

`endpoint="hls"` pulls from PC's `hls2-l30` + `hls2-s30` collections
in a single combined pool and composites them together. HLS already
applies the Roy et al. c-factor NBAR-style normalisation, so Landsat-8/9
OLI and Sentinel-2 MSI observations are bit-comparable.

Only the 7 harmonized common bands are exposed: `coastal, blue, green,
red, nir, swir16, swir22`. The "nir" band uses Landsat's B05 / Sentinel-2's
B8A (narrow NIR — the harmonized choice from Roy 2021), not S2's B08
broad NIR.

```python
out = spx.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=60.0,
    endpoint="hls",
)
# 5-year × 1-month over a 100 km AOI, parallel-5: ~7 s
```

Internally HLS scenes resolve their per-collection asset names
on-the-fly: for the same band slot ("red"), L30 reads `B04` and S30
reads `B04` (they happen to align), while "nir" reads `B05` on L30 and
`B8A` on S30. Quality scoring uses the bit-packed Fmask: cloud,
cirrus, cloud shadow, snow, and high aerosol bits each weight into a
lower-is-better score that drives best-pixel selection.

## Use from the command line

The crate also produces a native binary that writes tiled
DEFLATE-compressed GeoTIFFs:

```bash
cargo build --release
./target/release/spx-build \
    --bbox 30.5 30.5 31.6 31.5 \
    --datetime 2024-07-01/2024-07-31 \
    --resolution 60 \
    --top-k 3 \
    --endpoint pc \
    --out-dir /tmp/spx-out
```

The CLI supports a persistent disk cache (`--disk-cache <dir>`) so
repeated runs over the same AOI skip STAC + scout work.

## Architecture

```
src/
  cog.rs            HTTP COG reader (TIFF/IFD, range tiles, DEFLATE+predictor=2)
  stac.rs           Async STAC search client
  endpoint.rs       PC / Element84 endpoint config + SAS-token signing
  grid.rs           UTM grid math, bilinear AVX2 resampler, nearest u8 resampler (SCL)
  projx.rs          PROJ-backed coordinate transforms
  tile_classification.rs  Geometry-based exclusive-coverage MGRS chunk classifier
  pipeline.rs       scout / select / fetch / compose
  writer.rs         Tiled DEFLATE GeoTIFF output
  disk_cache.rs     JSON cache for search / scout / partition
  py.rs             PyO3 module exposing build_composite()
  bin/spx_build.rs  CLI entrypoint
```
