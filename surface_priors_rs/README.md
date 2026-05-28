# bestpixel

Fast, Rust-backed **best-pixel cloud-free composites** straight from
STAC/COG sources — give it a bbox + date range, get cloud-free multiband
reflectance as numpy arrays (no downloads, no GeoTIFF writes). Supports
four STAC sources out of the box:

| Endpoint | Collection(s) | Bands | Quality mask | Output CRS |
|---|---|---|---|---|
| `pc` (default) | Sentinel-2 L2A | 12 (full S2 set) | SCL | UTM (from AOI) |
| `earth-search` | Sentinel-2 L2A | 12 (full S2 set) | SCL | UTM (from AOI) |
| `hls` | HLS L30 + S30 (combined harmonized pool) | 7 common bands | Fmask | UTM (from AOI) |
| `mcd43a4` | MCD43A4 v6.1 (MODIS NBAR, daily) | 6 (no rededge) | Mandatory_Quality_Band1 | MODIS Sinusoidal (native) |

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

## Install

```bash
pip install bestpixel
```

A single [abi3](https://docs.python.org/3/c-api/stable.html) wheel runs on
CPython 3.9–3.14. Or build locally with maturin:

```bash
pip install maturin
cd surface_priors_rs        # the crate directory
maturin develop --release --features python
```

## Use from Python

```python
import bestpixel as bp

out = bp.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),         # west, south, east, north (WGS84)
    datetime="2024-07-01/2024-07-31",      # STAC datetime range
    resolution=60.0,                        # metres
    endpoint="pc",                          # "pc" | "earth-search" | "hls" | "mcd43a4" | "auto"
    bands=["coastal", "blue", "green", "red", "nir", "swir16", "swir22"],
)

red = out["bands"]["red"]                   # uint16 ndarray, (H, W)
quality = out["quality"]                    # uint16, 0=clear, 1=marginal, 2=dark, 65535=nodata
print(out["grid"])                          # bounds, epsg, transform — for georeferencing
```

Values are int16 DN; **reflectance = DN / 10000** (the Sentinel-2 N0400
`+1000` BOA offset is harmonized internally so all years are comparable).

### Scene selection

By default each chunk takes a fixed `top_k` clearest observations. For
**adaptive depth**, set `coverage_target` in `(0, 1)`: the scout's coarse
per-chunk cloud masks drive how many scenes each chunk stacks — cloudy
chunks pull more, clear chunks stop early — with a `min_k` redundancy
floor and `max_k` cap. Add `windowed_fetch=True` to read each scene only
over the chunks that need it (far fewer bytes when depth is concentrated,
e.g. an under-observed swath-edge corner):

```python
out = bp.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    coverage_target=0.95, min_k=2, max_k=6,   # adaptive per-chunk depth
    windowed_fetch=True,                       # read only the windows that need depth
)
```

Available band names (stable across endpoints):
`coastal, blue, green, red, rededge1, rededge2, rededge3, nir, nir08,
nir09, swir16, swir22`. SCL / Fmask is consumed internally to derive
the `quality` raster — kept as a discrete class label (nearest
resampling) all the way through, so quality buckets stay categorical.

Pass `bands=None` (or omit) to fetch all bands the endpoint supports
(12 for S2 L2A, 7 for HLS).

### Multi-month / multi-year batches

`build_monthly_composites(bbox, years=[...], months=[...], ...)`
produces one composite per `(year, month)` combination from a single
STAC search + scout pass. Cleaner than looping `build_composite` in
Python, and faster because the search + scout work is amortised.

```python
out = bp.build_monthly_composites(
    bbox=(30.5, 30.5, 31.6, 31.5),
    years=[2018, 2019, 2020],
    months=[6, 7, 8],          # June / July / August
    resolution=60.0,
    top_k=3,
    endpoint="pc",
)
# 9 composites returned as a list, each carrying its own year/month:
for r in out:
    print(r["year"], r["month"], r["bands"]["red"].shape)
```

Per-period tasks are spawned concurrently on the shared runtime, so
the 9-period example above runs in ~22 s wall — about the same as
sequential `build_composite` calls but with one STAC search instead
of nine.

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
out = bp.build_composite(
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

### MODIS MCD43A4 (NBAR)

`endpoint="mcd43a4"` pulls the daily 500 m MODIS Nadir BRDF-Adjusted
Reflectance product from PC's `modis-43A4-061` collection.

Output stays in **MODIS Sinusoidal** end-to-end — no on-the-fly
reprojection to UTM. The returned `grid` dict reports `epsg=0` and
carries the proj4 string instead:

```python
out = bp.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=500.0,           # MODIS native
    endpoint="mcd43a4",
)
print(out["grid"]["proj4"])
# '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 ...'
```

If you need MCD43A4 stacked against an S2 or HLS UTM grid, `gdalwarp`
/ `rio warp` the returned arrays as a post-processing step.

Six harmonized bands are exposed: `blue, green, red, nir, swir16,
swir22` — mapped to MODIS bands 3/4/1/2/6/7 respectively. MODIS Band 5
(1240 nm) has no S2 / HLS counterpart and is omitted. Quality scoring
uses `BRDF_Albedo_Band_Mandatory_Quality_Band1`: `0 = full BRDF
inversion` (best), `1 = magnitude inversion` (acceptable), `255 = fill`.

#### Reprojecting to UTM

To stack MCD43A4 against an S2 or HLS UTM grid, pass
`output_crs="utm"`. The pipeline keeps the source COG reads in
sinusoidal coordinates but bilinearly reprojects to UTM during the
resample step (per-pixel PROJ transform; no extra disk passes).

```python
out = bp.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=500.0,
    endpoint="mcd43a4",
    output_crs="utm",
)
print(out["grid"]["epsg"])     # 32636 (UTM 36N)
```

Same flag works on `build_monthly_composites`. SCL / Fmask / mandatory-
quality stay nearest during reprojection, so categorical labels survive
the CRS change.

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
