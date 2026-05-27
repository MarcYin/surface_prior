"""Rebuild the three reference monthly composites with tile-aware selection
and report whether the seam-stripe regression is fixed.

The original ``.real-builds`` artefacts showed a vertical nodata stripe at
cols 512-639 rows 1024+ across all three months, caused by the global
``min_usable_fraction`` floor rejecting T-tile scenes for chunks
dominated by U-tile coverage. This script rebuilds the same AOI/temporal
windows against the new selection path and prints:

  - whether the stripe pixels are now filled (vs the cached build);
  - ``tile_partition`` attrs (multi-tile / unreachable chunk counts).

Run with ``PYTHONPATH=src python scripts/verify_tile_aware_rebuild.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# Match the directory the original artefacts live in.
DEFAULT_BEFORE_CACHE = Path(__file__).resolve().parents[1] / ".real-builds"
DEFAULT_REBUILD_CACHE = Path(__file__).resolve().parents[1] / ".real-builds-tile-aware"
DEFAULT_BBOX = (30.5, 30.5, 31.6, 31.5)
DEFAULT_RESOLUTION = 60.0
DEFAULT_CHUNK_SIZE = 512
MONTHS = (
    ("2024-07", ("2024-07-01", "2024-07-31")),
    ("2024-08", ("2024-08-01", "2024-08-31")),
    ("2024-09", ("2024-09-01", "2024-09-30")),
)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_REBUILD_CACHE)
    parser.add_argument(
        "--before-cache-dir",
        type=Path,
        default=DEFAULT_BEFORE_CACHE,
        help="Existing artefact root used to read the pre-fix stripe fill.",
    )
    parser.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX))
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--probe-cols",
        nargs=2,
        type=int,
        default=[512, 640],
        help="Column slice of the suspect vertical stripe (half-open).",
    )
    parser.add_argument(
        "--probe-rows",
        nargs=2,
        type=int,
        default=[1024, 1880],
        help="Row slice of the suspect vertical stripe (half-open).",
    )
    parser.add_argument(
        "--probe-band",
        default="s2_b04_red",
        help="Band TIF used to measure stripe fill.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from surface_priors import Provider, ProviderConfig, SelectionPolicy
    from surface_priors.sources.stac_api import StacApiSource
    from surface_priors.types import DEFAULT_S2_L2A_BANDS

    source = StacApiSource.earth_search_s2_l2a(
        temporal_ranges=[("2024-07-01", "2024-09-30")],
        chunk_size=args.chunk_size,
    )
    provider = Provider(
        ProviderConfig(
            cache_dir=args.cache_dir,
            source=source,
            chunk_size=args.chunk_size,
            selection_policy=SelectionPolicy(top_k=3),
        )
    )

    bbox = tuple(args.bbox)
    bands = list(DEFAULT_S2_L2A_BANDS)

    rebuilt: list[dict] = []
    for period, temporal in MONTHS:
        before = _stripe_fill_from_cache(
            cache_dir=args.before_cache_dir,
            period=period,
            band=args.probe_band,
            rows=tuple(args.probe_rows),
            cols=tuple(args.probe_cols),
        )
        t0 = time.perf_counter()
        product = provider.build_prior(
            wgs84_bounds=bbox,
            resolution=args.resolution,
            product_id=f"verify-{period}",
            composite_period=period,
            band_names=bands,
            temporal_filter=temporal,
            rebuild=True,
        )
        elapsed = time.perf_counter() - t0
        attrs = dict(product.composite.attrs)
        after = _stripe_fill_from_array(
            data_band=_band_array(product.composite, args.probe_band, bands),
            rows=tuple(args.probe_rows),
            cols=tuple(args.probe_cols),
        )
        rebuilt.append(
            {
                "period": period,
                "elapsed_s": round(elapsed, 1),
                "stripe_fill_before": before,
                "stripe_fill_after": after,
                "tile_partition": attrs.get("tile_partition"),
                "empty_chunk_count": attrs.get("empty_chunk_count", 0),
            }
        )
        _print_period_summary(rebuilt[-1])

    print()
    print("=== Summary ===")
    for record in rebuilt:
        before = record["stripe_fill_before"]
        after = record["stripe_fill_after"]
        if before is None:
            before_pct = "-"
        else:
            before_pct = f"{before * 100:5.1f}%"
        after_pct = f"{after * 100:5.1f}%"
        partition = record["tile_partition"] or {}
        print(
            f"  {record['period']}  stripe fill {before_pct} → {after_pct}"
            f"  tiles={partition.get('tiles_used', [])}"
            f"  multi-tile chunks={partition.get('multi_tile_chunk_count', '-')}"
            f"  unreachable chunks={partition.get('unreachable_chunk_count', '-')}"
        )
    return 0


def _band_array(composite, band: str, band_names: list[str]) -> np.ndarray:
    index = band_names.index(band)
    return composite.data[index]


def _stripe_fill_from_array(
    *,
    data_band: np.ndarray,
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> float:
    r0, r1 = rows
    c0, c1 = cols
    r1 = min(r1, data_band.shape[0])
    c1 = min(c1, data_band.shape[1])
    block = data_band[r0:r1, c0:c1]
    if block.size == 0:
        return float("nan")
    finite = np.isfinite(block)
    return float(finite.mean())


def _stripe_fill_from_cache(
    *,
    cache_dir: Path,
    period: str,
    band: str,
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> float | None:
    """Read the same stripe from any existing cached build for the period.

    Used as a before/after reference so the script self-reports the fix.
    Returns None if no cached build is found.
    """

    try:
        import rasterio
    except ImportError:
        return None

    for product_dir in cache_dir.iterdir() if cache_dir.exists() else ():
        stac_path = product_dir / "stac-item.json"
        if not stac_path.is_file():
            continue
        try:
            stac = json.loads(stac_path.read_text())
        except Exception:
            continue
        if stac.get("properties", {}).get("surface:composite_period") != period:
            continue
        tif = product_dir / "assets" / "prior" / period / f"{band}.tif"
        if not tif.is_file():
            continue
        with rasterio.open(tif) as src:
            r0, r1 = rows
            c0, c1 = cols
            r1 = min(r1, src.height)
            c1 = min(c1, src.width)
            window = ((r0, r1), (c0, c1))
            data = src.read(1, window=window, masked=True)
            if data.size == 0:
                return None
            return float((~data.mask).mean()) if hasattr(data, "mask") else 1.0
    return None


def _print_period_summary(record: dict) -> None:
    print(f"--- {record['period']} (rebuild took {record['elapsed_s']}s) ---")
    partition = record.get("tile_partition") or {}
    if partition:
        print(f"  tiles used:           {partition.get('tiles_used', [])}")
        print(f"  multi-tile chunks:    {partition.get('multi_tile_chunk_count')}")
        print(f"  unreachable chunks:   {partition.get('unreachable_chunk_count')}")
    else:
        print("  tile_partition attrs missing (source returned no partition)")
    before = record["stripe_fill_before"]
    print(f"  stripe fill before:   {before * 100:.1f}%" if before is not None else "  stripe fill before:   <no cached build>")
    print(f"  stripe fill after:    {record['stripe_fill_after'] * 100:.1f}%")
    print(f"  empty_chunk_count:    {record['empty_chunk_count']}")


if __name__ == "__main__":
    sys.exit(main())
