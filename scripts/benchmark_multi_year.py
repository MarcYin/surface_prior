"""Simulate the user's actual workflow across multiple years.

Per year:
  - one StacApiSource for ``target ± 1 month`` (3-month temporal range)
  - 3 monthly builds using ``temporal_filter`` to slice within
  - same AOI, grid, layout across years (only the temporal range changes)

Runs two passes so we can compare cold (fresh disk cache) vs warm (cache
populated by the cold pass). Useful for understanding how the disk cache
amortises a real 5-year batch.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, default=[30.5, 30.5, 31.6, 31.5])
    parser.add_argument("--target-month", type=int, default=7, help="1..12")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--resolution", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/spx-multi-year"))
    parser.add_argument("--passes", type=int, default=2, help="Number of full passes (1=cold only, 2=cold+warm).")
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Build only the target month per year (skip target-1 and target+1).",
    )
    parser.add_argument(
        "--parallel-years",
        type=int,
        default=1,
        help="How many years to build concurrently in worker threads.",
    )
    parser.add_argument(
        "--per-year-fetch-workers",
        type=int,
        default=None,
        help="Override fetch_workers per year (default: 96 / parallel_years).",
    )
    parser.add_argument(
        "--per-year-scout-workers",
        type=int,
        default=None,
        help="Override scout_workers per year (default: 32).",
    )
    parser.add_argument(
        "--per-year-band-workers",
        type=int,
        default=3,
        help="Override band_workers per year (default 3).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from surface_priors.chunks import ChunkLayout
    from surface_priors.composite import ChunkedCompositor
    from surface_priors.provider import _scene_fetcher_for
    from surface_priors.selection import SelectionPolicy, select
    from surface_priors.sources.stac_api import StacApiSource
    from surface_priors.types import DEFAULT_S2_L2A_BANDS

    bbox = tuple(args.bbox)
    bands = list(DEFAULT_S2_L2A_BANDS)
    target = args.target_month
    if args.target_only:
        months_to_build = [target]
    else:
        months_to_build = [
            _shift_month(target, -1),
            target,
            _shift_month(target, 1),
        ]

    band_workers = args.per_year_band_workers
    # Per-year fetch concurrency × band reads × parallel years must stay
    # under Element84's ~100-concurrent ceiling, otherwise we get 429s and
    # exponential backoff that erases the parallelism win.
    fetch_workers = args.per_year_fetch_workers or max(
        1, 96 // (max(1, args.parallel_years) * max(1, band_workers))
    )
    scout_workers = args.per_year_scout_workers or max(
        4, 96 // max(1, args.parallel_years)
    )

    def build_one(year: int, month: int) -> dict:
        # Source temporal range covers exactly the months we will build.
        first_build = min(months_to_build)
        last_build = max(months_to_build)
        source_start = f"{year}-{first_build:02d}-01"
        source_end = _last_day_of(year, last_build)
        src = StacApiSource.earth_search_s2_l2a(
            temporal_ranges=[(source_start, source_end)],
            chunk_size=args.chunk_size,
            disk_cache=args.cache_dir,
            scout_workers=scout_workers,
            band_workers=band_workers,
        )
        grid = src.resolve_grid(
            wgs84_bounds=bbox,
            native_crs="EPSG:32636",
            resolution=args.resolution,
            band_names=(),
        )
        layout = ChunkLayout.from_grid(grid, chunk_size=args.chunk_size)
        # Sequential per-month builds against the SAME source instance.
        per_month = {}
        list_t = part_t = 0.0
        for build_month in months_to_build:
            month_str = _month_str(year, build_month)
            month_start = f"{month_str}-01"
            month_end = _last_day_of(year, build_month)
            t0 = time.perf_counter()
            scenes = src.list_scenes(grid=grid)
            list_t_local = time.perf_counter() - t0
            t1 = time.perf_counter()
            partition = src.tile_partition(grid=grid, layout=layout)
            part_t_local = time.perf_counter() - t1
            t1 = time.perf_counter()
            stats = src.scout(
                grid=grid, layout=layout, band_names=bands,
                temporal_filter=(month_start, month_end),
            )
            scout_t = time.perf_counter() - t1
            plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=3), partition=partition)
            t1 = time.perf_counter()
            fetch_scene = _scene_fetcher_for(source=src, grid=grid, plan=plan, band_names=bands)
            ChunkedCompositor().compose_pipelined(
                product_id=month_str, grid=grid, band_names=bands, plan=plan,
                fetch_scene=fetch_scene, fetch_workers=fetch_workers,
            )
            fc_t = time.perf_counter() - t1
            total = time.perf_counter() - t0
            per_month[month_str] = {
                "total": total, "list": list_t_local, "part": part_t_local,
                "scout": scout_t, "fetch_compose": fc_t,
                "n_scenes": len(scenes),
            }
        return per_month

    print(f"AOI bbox={bbox}  resolution={args.resolution}  chunk={args.chunk_size}")
    print(f"target month = {target}, building target±1 ({months_to_build}) for years {args.years}")
    print(f"cache dir = {args.cache_dir}")

    if args.passes >= 1:
        # Start cold.
        if args.cache_dir.exists():
            shutil.rmtree(args.cache_dir)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    pass_totals: list[float] = []
    print(
        f"parallel_years={args.parallel_years}, "
        f"per-year: fetch={fetch_workers} scout={scout_workers} band={band_workers}"
    )
    for pass_idx in range(1, args.passes + 1):
        label = "cold" if pass_idx == 1 else f"warm-{pass_idx-1}"
        print(f"\n=== pass {pass_idx} ({label}) ===")

        def run_year(year: int):
            t0 = time.perf_counter()
            results = build_one(year, target)
            return year, time.perf_counter() - t0, results

        wall_start = time.perf_counter()
        per_year: dict[int, tuple] = {}
        if args.parallel_years <= 1:
            for year in args.years:
                y, elapsed, results = run_year(year)
                per_year[y] = (elapsed, results)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_years) as pool:
                futures = [pool.submit(run_year, y) for y in args.years]
                for fut in as_completed(futures):
                    y, elapsed, results = fut.result()
                    per_year[y] = (elapsed, results)
        wall = time.perf_counter() - wall_start

        per_year_cpu_total = 0.0
        for year in args.years:
            elapsed, results = per_year[year]
            month_strs = sorted(results)
            totals = [results[m]["total"] for m in month_strs]
            scenes_first = results[month_strs[0]]["n_scenes"]
            cells = "  ".join(
                f"month{i+1}={t:5.2f}s" for i, t in enumerate(totals)
            )
            print(
                f"  year={year}  {cells}  "
                f"year_elapsed={elapsed:5.2f}s  scenes_in_source={scenes_first}"
            )
            per_year_cpu_total += elapsed
        print(
            f"  pass {label} wall={wall:.2f}s   sum_of_year_elapsed={per_year_cpu_total:.2f}s"
        )
        pass_totals.append(wall)

    if len(pass_totals) >= 2:
        print(
            f"\nCold total: {pass_totals[0]:.2f}s — Warm total: {pass_totals[1]:.2f}s — "
            f"saved by disk cache: {pass_totals[0]-pass_totals[1]:.2f}s "
            f"({(pass_totals[0]-pass_totals[1])/pass_totals[0]*100:.1f}%)"
        )
    return 0


def _shift_month(month: int, delta: int) -> int:
    return ((month - 1 + delta) % 12) + 1


def _month_str(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def _last_day_of(year: int, month: int) -> str:
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    d = days[month]
    if month == 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        d = 29
    return f"{year}-{month:02d}-{d:02d}"


if __name__ == "__main__":
    sys.exit(main())
