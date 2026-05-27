"""Print per-phase timing for a single monthly tile-aware build.

Runs one month against the live STAC source and reports how the elapsed
time splits between list_scenes / scout / partition / fetch / compose,
so we can spot whether the steady-state ~20s is dominated by I/O,
HTTP overhead, or something in-process.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable, Optional

import numpy as np


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, default=[30.5, 30.5, 31.6, 31.5])
    parser.add_argument("--month", default="2024-08")
    parser.add_argument("--resolution", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--scout-workers", type=int, default=16)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--band-workers", type=int, default=12)
    args = parser.parse_args(list(argv) if argv is not None else None)

    from surface_priors.chunks import ChunkLayout
    from surface_priors.composite import ChunkedCompositor
    from surface_priors.provider import _prefetch_chunks
    from surface_priors.selection import SelectionPolicy, select
    from surface_priors.sources.stac_api import StacApiSource
    from surface_priors.types import DEFAULT_S2_L2A_BANDS, Observation

    bbox = tuple(args.bbox)
    start = f"{args.month}-01"
    end = _month_end(args.month)

    source = StacApiSource.earth_search_s2_l2a(
        temporal_ranges=[(start, end)],
        chunk_size=args.chunk_size,
        scout_workers=args.scout_workers,
        band_workers=args.band_workers,
    )
    grid = source.resolve_grid(
        wgs84_bounds=bbox,
        native_crs="EPSG:32636",
        resolution=args.resolution,
        band_names=(),
    )
    layout = ChunkLayout.from_grid(grid, chunk_size=args.chunk_size)
    bands = list(DEFAULT_S2_L2A_BANDS)

    t0 = time.perf_counter()
    scenes = source.list_scenes(grid=grid)
    list_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    partition = source.tile_partition(grid=grid, layout=layout)
    part_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats = source.scout(grid=grid, layout=layout, band_names=bands)
    scout_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=3), partition=partition)
    select_t = time.perf_counter() - t0

    n_scene_chunk = sum(len(s) for s in plan.selected.values())
    unique_scenes = len({s for scenes in plan.selected.values() for s in scenes})

    t0 = time.perf_counter()
    cache = _prefetch_chunks(
        source=source,
        grid=grid,
        plan=plan,
        band_names=bands,
        workers=args.fetch_workers,
    )
    fetch_t = time.perf_counter() - t0

    def chunk_loader(scene_index: int, chunk_id: int) -> Optional[Observation]:
        return cache.get((int(scene_index), int(chunk_id)))

    t0 = time.perf_counter()
    compositor = ChunkedCompositor()
    composite = compositor.compose(
        product_id="profile",
        grid=grid,
        band_names=bands,
        plan=plan,
        chunk_loader=chunk_loader,
    )
    compose_t = time.perf_counter() - t0

    n_filled = int(np.count_nonzero(composite.observation_count))
    total = list_t + part_t + scout_t + select_t + fetch_t + compose_t

    print(f"month: {args.month}")
    print(f"  scenes total:           {len(scenes)}")
    print(f"  plan picks (scene,chunk): {n_scene_chunk}")
    print(f"  unique scenes picked:   {unique_scenes}")
    print(f"  chunks filled (pixels): {n_filled}/{grid.height*grid.width}")
    print()
    print(f"  list_scenes:     {list_t:6.2f}s")
    print(f"  tile_partition:  {part_t:6.2f}s")
    print(f"  scout:           {scout_t:6.2f}s")
    print(f"  select:          {select_t:6.2f}s")
    print(f"  fetch:           {fetch_t:6.2f}s")
    print(f"  compose:         {compose_t:6.2f}s")
    print("  ─────────────────")
    print(f"  TOTAL:           {total:6.2f}s")
    return 0


def _month_end(month: str) -> str:
    year, mm = month.split("-")
    days = {"01": 31, "03": 31, "04": 30, "05": 31, "06": 30, "07": 31, "08": 31, "09": 30, "10": 31, "11": 30, "12": 31}
    if mm == "02":
        return f"{year}-02-29" if int(year) % 4 == 0 else f"{year}-02-28"
    return f"{year}-{mm}-{days[mm]:d}"


if __name__ == "__main__":
    sys.exit(main())
