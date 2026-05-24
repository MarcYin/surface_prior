# Surface Priors

`surface-priors` builds native-grid surface prior products and writes them as STAC/GeoTIFF assets.

The current implemented provider builds MODIS/VIIRS BRDF priors. The package
name and output schema are intentionally broader so future providers can add
direct surface reflectance priors from sensors such as Sentinel-2 or Landsat.

The package boundary is intentionally narrow:

- Accept observations that are already on the requested native grid.
- Fetch Google Earth Engine BRDF observations through `edown` when configured, optionally sampling one day every `N` days to reduce downloads.
- Composite the best pixel per BRDF band using quality and sample-index tie-breaks.
- Encode the prior as `uint16` with scale factor `10000`.
- Encode relative uncertainty as `uint8` percent from `0` to `200`, with `255` marking suspicious or missing uncertainty.
- Persist tiled, DEFLATE-compressed GeoTIFF assets with no overviews.
- Emit a STAC Item that points to those assets.

Calendar planning is not part of the builder. Target dates, adjacent months, seasons, and history years are usage policy owned by SIAC or another caller.

## Contract

```python
from surface_priors import Provider, ProviderConfig
from surface_priors.sources import InMemorySource

provider = Provider(ProviderConfig(cache_dir=".surface-cache", source=source))

product = provider.build_prior(
    product_id="example-brdf-prior",
    wgs84_bounds=(-1.0, 51.0, -0.99, 51.01),
    resolution=500.0,
    band_names=("brdf_iso_red",),
    composite_period="2024-07",
)
```

The returned `PriorProduct` contains the in-memory composite, output directory,
and STAC Item dictionary. `composite_period` is a caller-defined label used in
STAC metadata and asset paths; it does not make the package choose the month or
temporal input policy.

Input AOI bounds are always WGS84 `(west, south, east, north)`. The package converts them to the configured native CRS internally; the current BRDF default is MODIS/VIIRS Sinusoidal.

Sources that can resolve their native grid, such as the `edown` Google Earth Engine source, may replace that fallback grid with the exact downloaded GeoTIFF grid. The compositor still receives native-grid arrays and does not reproject them.

## Sentinel-2 L2A composites (Rust)

The repository also ships a separate Rust crate, **`surface-priors-rs`**,
that produces best-pixel monthly composites and returns them to Python
as numpy arrays. Supports Sentinel-2 L2A (Planetary Computer / Element84,
12 bands) and HLS v2.0 L30 + S30 (Planetary Computer, combined harmonized
7-band pool). On a 16-core node it builds 5 years × 1 month over a
100 × 100 km AOI at 60 m in roughly 6-7 seconds, network-bound against
Planetary Computer.

```python
import surface_priors_rs as spx

out = spx.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=60.0,
    top_k=3,
    endpoint="pc",   # or "hls" for Harmonized Landsat-Sentinel-2
)
red = out["bands"]["red"]   # uint16 ndarray (H, W)
```

Source and full docs live under
[`surface_priors_rs/`](https://github.com/MarcYin/surface_prior/tree/main/surface_priors_rs)
in the repo. Wheels for CPython 3.9–3.14 are attached to each GitHub
Release tagged `rs-v*`.
