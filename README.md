# Surface Priors

`surface-priors` builds native-grid surface prior products and persists them as a STAC Item with tiled, DEFLATE-compressed GeoTIFF assets. The current implementation provides a MODIS/VIIRS BRDF prior builder; the package boundary is intentionally broader so later providers can add direct surface reflectance priors from sensors such as Sentinel-2 or Landsat.

> **This repository ships two independent, separately-installed packages — pick the one you need:**
>
> | Package | Install | What it does |
> |---|---|---|
> | **[`surface-priors`](https://pypi.org/project/surface-priors/)** (this README) | `pip install surface-priors` | Builds native-grid surface **prior products** and writes them as STAC + GeoTIFF (MODIS/VIIRS BRDF today). |
> | **[`bestpixel`](https://pypi.org/project/bestpixel/)** (Rust, in [`surface_priors_rs/`](surface_priors_rs/)) | `pip install bestpixel` | Fast best-pixel **cloud-free composites** from STAC/COG sources (Sentinel-2 L2A, HLS, MCD43A4), returned as numpy arrays — see the [`bestpixel`](#bestpixel-fast-cloud-free-composites-rust) section below. |
>
> They share lineage but are distinct distributions; you do **not** need both.

Calendar planning is intentionally outside the core builder: callers decide which observations should enter a prior, then this package composites those observations and writes a consistent product.

```python
import numpy as np

from surface_priors import Observation, Provider, ProviderConfig
from surface_priors.sources import InMemorySource

obs = Observation(
    data=np.ones((1, 2, 2), dtype="float32") * 0.25,
    quality=np.zeros((2, 2), dtype="uint16"),
    uncertainty=np.ones((1, 2, 2), dtype="float32") * 12,
    band_names=("brdf_iso_red",),
)

provider = Provider(
    ProviderConfig(
        cache_dir=".surface-cache",
        source=InMemorySource((obs,), name="example"),
    )
)

product = provider.build_prior(
    product_id="example-brdf-prior",
    wgs84_bounds=(-1.0, 51.0, -0.99, 51.01),
    resolution=500.0,
    band_names=("brdf_iso_red",),
    composite_period="2024-07",
)
```

The output directory contains:

```text
<cache-root>/<request-hash>/
  stac-item.json
  assets/
    prior/
      2024-07/
        brdf_iso_red.tif
    uncertainty/
      2024-07/
        brdf_iso_red.tif
```

## Contract

Implemented now:

- MODIS/VIIRS BRDF prior fetching through Google Earth Engine or Earthaccess-backed sources.
- Native-grid best-pixel BRDF compositing and quality tie-breaking.
- STAC/GeoTIFF persistence with a generic `surface:*` metadata namespace.

Planned extension point:

- Direct surface reflectance prior sources such as Sentinel-2 or Landsat, provided they can deliver observations aligned to the requested native grid.

## `bestpixel`: fast cloud-free composites (Rust)

The `surface_priors_rs/` directory in this repo contains a separately
distributed Rust crate, published on PyPI as **[`bestpixel`](https://pypi.org/project/bestpixel/)**,
that builds best-pixel monthly composites end-to-end (STAC search → MGRS
tile-aware selection → COG fetch → resample → compose) and exposes them
to Python as numpy arrays — no GeoTIFF writes required. Four sources are
supported:
Sentinel-2 L2A via Planetary Computer (default) or Element84;
Harmonized Landsat-Sentinel-2 (HLS v2.0) L30 + S30 via PC, combined
into one harmonized 7-band pool; and MODIS MCD43A4 NBAR (daily 500 m,
output in MODIS Sinusoidal native).

On a 16-core node from JASMIN it produces 5 years × 1 month over a
100 × 100 km AOI at 60 m in about **6 seconds** parallel or **11
seconds** sequential, network-bound against Planetary Computer.

```bash
pip install bestpixel
```

```python
import bestpixel as bp

# Sentinel-2 L2A (12 bands)
out = bp.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=60.0,
    top_k=3,
    endpoint="pc",
)

# Harmonized Landsat-Sentinel-2 (7 common bands, Roy et al. NBAR)
out = bp.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=60.0,
    endpoint="hls",
)

# MODIS MCD43A4 NBAR (6 bands, native 500 m, output in MODIS Sinusoidal)
out = bp.build_composite(
    bbox=(30.5, 30.5, 31.6, 31.5),
    datetime="2024-07-01/2024-07-31",
    resolution=500.0,
    endpoint="mcd43a4",
)

red = out["bands"]["red"]      # uint16 ndarray (H, W)
quality = out["quality"]       # 0=clear, 1=marginal, 2=dark, 65535=nodata
print(out["grid"])             # bounds, epsg, transform
```

One abi3 wheel covers Python 3.9 through 3.14 (prebuilt for Linux x86_64
and macOS arm64; other platforms build from the sdist).

See [`surface_priors_rs/README.md`](surface_priors_rs/README.md) for
the full API, CLI usage, architecture, and benchmark details.

## `surface-priors` responsibilities

`surface-priors` owns:

- WGS84 AOI bounds conversion into the native prior data CRS.
- Google Earth Engine BRDF product downloading through `edown`.
- Native-grid best-pixel compositing from caller-supplied prior observations.
- Source quality and sample-index tie-breaking, currently implemented for BRDF products.
- Relative uncertainty propagation or fallback estimation.
- `uint16` prior encoding with scale factor `10000` and nodata `65535`.
- `uint8` relative uncertainty encoding in percent from `0` to `200`; values above `200%`, negative values, and non-finite values are stored as `255`.
- One-band tiled, DEFLATE-compressed GeoTIFF persistence optimized for remote chunked reads, without overviews.
- STAC Item creation with `projection` and `raster` extension metadata.

Callers own:

- Which observations to use for a prior.
- Any observation-day, month, season, or year logic.
- The optional `composite_period` label, such as `2024-07`, when assets should carry a monthly path segment.
- NASA/Earthdata search policy and temporal filtering.
- SIAC atmospheric correction, SWIR refine routing, spectral mapping, and downstream `SurfacePrior` construction.

The public AOI input is always WGS84 longitude/latitude bounds: `(west, south, east, north)`. The package converts those bounds to the configured native prior data CRS, which defaults to MODIS/VIIRS Sinusoidal for BRDF sources. The builder still does not reproject source arrays internally; observations must already match the derived native grid.

## Installing `surface-priors`

```bash
pip install surface-priors
```

Optional Earthdata search support:

```bash
pip install "surface-priors[earthdata]"
```

Optional Google Earth Engine download support through `edown`:

```bash
pip install "surface-priors[gee]"
```

Optional experiment dependencies for the GEE-vs-official comparison:

```bash
pip install "surface-priors[experiments]"
```

Development install:

```bash
python -m pip install -e ".[dev,docs]"
pytest
```

## Google Earth Engine Input

The built-in GEE preset uses `edown` to download `MODIS/061/MCD43A1` native-grid GeoTIFF observations. By default it requests `iso`, `vol`, and `geo` BRDF coefficients for red, green, blue, NIR, SWIR1, and SWIR2. The caller still supplies explicit temporal ranges; this package does not decide which days, months, or history windows to use. To reduce downloads, pass `sample_every_days` to query one-day windows at a fixed stride inside each temporal range.

```python
from surface_priors import Provider, ProviderConfig
from surface_priors.sources import EdownGeeSource

source = EdownGeeSource.for_product(
    "mcd43a1",
    temporal_ranges=(("2024-07-01", "2024-07-31"),),
    sample_every_days=7,
    output_root=".surface-gee-cache",
)

provider = Provider(ProviderConfig(cache_dir=".surface-cache", source=source))
product = provider.build_prior(
    product_id="mcd43a1-prior",
    wgs84_bounds=(-2.0, 51.0, -1.0, 52.0),
    resolution=500.0,
    composite_period="2024-07",
)
```

With `sample_every_days=7`, the July range above queries `2024-07-01`,
`2024-07-08`, `2024-07-15`, `2024-07-22`, and `2024-07-29` instead of every
matching image in the month.

`edown` handles Earth Engine authentication using `GEE_SERVICE_ACCOUNT`/`GEE_SERVICE_ACCOUNT_KEY`, existing Earth Engine user credentials, or Google Application Default Credentials.

## CLI

```bash
surface-priors build \
  --product-id example-brdf-prior \
  --wgs84-bounds -1.0 51.0 -0.99 51.01 \
  --resolution 500 \
  --band brdf_iso_red \
  --composite-period 2024-07 \
  --local-observations observations.json \
  --cache-dir .surface-cache
```

GEE MCD43A1 through `edown`:

```bash
surface-priors build \
  --product-id mcd43a1-prior \
  --gee-product mcd43a1 \
  --temporal-range 2024-07-01 2024-07-31 \
  --sample-every-days 7 \
  --composite-period 2024-07 \
  --wgs84-bounds -2.0 51.0 -1.0 52.0 \
  --resolution 500 \
  --cache-dir .surface-cache \
  --edown-output-root .surface-gee-cache
```

`observations.json` points to local native-grid NPZ observations used as input, not as output:

```json
{
  "name": "example",
  "band_names": ["brdf_iso_red"],
  "items": [
    {
      "path": "obs.npz",
      "data_key": "data",
      "quality_key": "quality",
      "uncertainty_key": "uncertainty",
      "sample_index_key": "sample_index"
    }
  ]
}
```

## Publishing

The repository includes GitHub Actions workflows for tests, package build checks, PyPI trusted publishing on GitHub releases, and MkDocs Material deployment to GitHub Pages.
