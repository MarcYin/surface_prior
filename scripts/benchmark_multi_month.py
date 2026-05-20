"""Build N monthly composites against one StacApiSource and report savings.

Demonstrates that sharing a wide temporal_range across multiple Provider
builds amortizes the STAC search cost. Each build sets temporal_filter to
the target month so the source's cached scene list is filtered without
re-querying the catalogue.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from surface_priors.provider import Provider, ProviderConfig  # noqa: E402
from surface_priors.selection import SelectionPolicy  # noqa: E402
from surface_priors.sources.stac_api import StacApiSource  # noqa: E402
from surface_priors.types import DEFAULT_S2_L2A_BANDS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wgs84-bounds", nargs=4, type=float, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--resolution", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--scout-workers", type=int, default=16)
    parser.add_argument("--band-workers", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path(".benchmark-cache"))
    args = parser.parse_args()

    months = sorted(args.months)
    overall_start = f"{args.year}-{months[0]:02d}-01"
    overall_end_month = months[-1] + 1
    overall_end_year = args.year + (overall_end_month > 12)
    overall_end_month = ((overall_end_month - 1) % 12) + 1
    overall_end = f"{overall_end_year}-{overall_end_month:02d}-01"

    source = StacApiSource.earth_search_s2_l2a(
        temporal_ranges=((overall_start, overall_end),),
        chunk_size=args.chunk_size,
        scout_workers=args.scout_workers,
        band_workers=args.band_workers,
    )
    provider = Provider(
        ProviderConfig(
            cache_dir=args.cache_dir,
            source=source,
            chunk_size=args.chunk_size,
            selection_policy=SelectionPolicy(top_k=args.top_k, min_usable_fraction=0.5),
            fetch_workers=args.fetch_workers,
        )
    )

    band_names = tuple(DEFAULT_S2_L2A_BANDS)
    timings = []
    for month in months:
        start = f"{args.year}-{month:02d}-01"
        next_month = month + 1
        next_year = args.year + (next_month > 12)
        next_month_norm = ((next_month - 1) % 12) + 1
        end = f"{next_year}-{next_month_norm:02d}-01"

        t0 = time.perf_counter()
        product = provider.build_prior(
            wgs84_bounds=tuple(args.wgs84_bounds),
            resolution=args.resolution,
            product_id=f"multi-month-{args.year}-{month:02d}",
            band_names=band_names,
            composite_period=f"{args.year}-{month:02d}",
            temporal_filter=(start, end),
        )
        elapsed = time.perf_counter() - t0
        timings.append((month, elapsed, product.composite.observation_count.sum()))

    print("\n=== MULTI-MONTH BENCHMARK ===")
    print(f"AOI:             {args.wgs84_bounds}")
    print(f"Source window:   {overall_start} → {overall_end}")
    print(f"Resolution:      {args.resolution} m")
    print(f"Months:          {months}")
    print()
    total = 0.0
    for month, elapsed, valid_pixels in timings:
        print(f"  month {args.year}-{month:02d}: {elapsed:6.2f} s  valid_pixels={valid_pixels}")
        total += elapsed
    print(f"\n  total:        {total:6.2f} s")
    print(f"  mean/month:   {total / len(timings):6.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
