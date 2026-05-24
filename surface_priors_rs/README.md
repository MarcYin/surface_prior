# surface-priors-rs

Production Rust port of the surface-priors Sentinel-2 L2A monthly
composite pipeline, with optional Python bindings that return numpy
arrays directly (no GeoTIFF writes).

## What it does

- Asynchronously fetches Sentinel-2 L2A scenes from Planetary Computer
  (default) or Element84 earth-search via STAC search + range-request
  COG reads.
- Selects the top-k clearest observations per MGRS-tile-aware chunk
  using a coarse SCL-based scout pass.
- Decodes DEFLATE-compressed COG tiles via libdeflate, undoes
  TIFF predictor=2, and reprojects to a user-specified UTM grid with
  an AVX2/FMA bilinear resampler.
- Composes a best-pixel monthly composite ranked by SCL quality.

For a 100 × 100 km AOI at 60 m resolution, top-k=3 from Planetary
Computer, on a single 16-core Zen 4 node:

| Workload | Time |
|---|---|
| 1 year × 1 month | ~2 s |
| 5 years × 1 month, sequential | ~11 s |
| 5 years × 1 month, 5-thread parallel | ~6 s |

The 5-way parallel floor is set by network throughput to PC, not CPU.

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
nir09, swir16, swir22`. SCL is consumed internally to derive the
`quality` raster — kept as a discrete class label (nearest resampling)
all the way through, so quality buckets stay categorical.

Pass `bands=None` (or omit) to fetch all 12.

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
