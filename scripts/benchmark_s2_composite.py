"""Phase-by-phase timing for the chunked S2 monthly-composite pipeline.

Two modes:

  --mode real          Hit Element84 earth-search for the configured AOI and
                       temporal range, time each pipeline phase, and write a
                       monthly composite under --cache-dir.

  --mode synthetic     Skip network entirely. A fake source serves random
                       chunks for an AOI-sized grid so the timing reflects
                       compositor + Provider orchestration only.

Both modes log per-phase wall-clock seconds and a derived per-chunk rate.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from surface_priors.chunks import ChunkLayout  # noqa: E402
from surface_priors.composite import ChunkedCompositor  # noqa: E402
from surface_priors.provider import Provider, ProviderConfig  # noqa: E402
from surface_priors.selection import SceneChunkStats, SelectionPolicy, select  # noqa: E402
from surface_priors.sources.stac_api import StacApiSource  # noqa: E402
from surface_priors.types import DEFAULT_S2_L2A_BANDS, GridSpec, Observation  # noqa: E402


@dataclass
class PhaseTimings:
    list_scenes: float = 0.0
    scout: float = 0.0
    select: float = 0.0
    fetch_and_compose: float = 0.0
    save: float = 0.0
    total: float = 0.0
    n_scenes: int = 0
    n_chunks: int = 0
    n_selected_pairs: int = 0
    grid_shape: tuple = (0, 0)
    chunk_size: int = 0

    def summary(self) -> Dict[str, str]:
        per_chunk = (self.fetch_and_compose / self.n_chunks) if self.n_chunks else 0.0
        per_pair = (
            self.fetch_and_compose / self.n_selected_pairs
            if self.n_selected_pairs
            else 0.0
        )
        return {
            "grid": f"{self.grid_shape[0]}x{self.grid_shape[1]} px",
            "chunk_size": str(self.chunk_size),
            "n_scenes": str(self.n_scenes),
            "n_chunks": str(self.n_chunks),
            "n_selected_pairs": str(self.n_selected_pairs),
            "list_scenes_s": f"{self.list_scenes:.2f}",
            "scout_s": f"{self.scout:.2f}",
            "select_s": f"{self.select:.4f}",
            "fetch_compose_s": f"{self.fetch_and_compose:.2f}",
            "save_s": f"{self.save:.2f}",
            "total_s": f"{self.total:.2f}",
            "per_chunk_s": f"{per_chunk:.3f}",
            "per_selected_pair_s": f"{per_pair:.3f}",
        }


class _SyntheticSource:
    """Deterministic chunked source for compositor-only benchmarking."""

    def __init__(
        self,
        *,
        grid: GridSpec,
        band_names: Sequence[str],
        n_scenes: int,
        chunk_size: int,
    ) -> None:
        self.grid = grid
        self.band_names = tuple(band_names)
        self.n_scenes = int(n_scenes)
        self.chunk_size = int(chunk_size)
        self.name = f"synthetic:{n_scenes}sc:{chunk_size}"
        self._rng = np.random.default_rng(0)

    def block_size(self, *, grid, band_names):
        return None

    def resolve_grid(self, *, wgs84_bounds, native_crs, resolution, band_names):
        return self.grid

    def scout(self, *, grid, layout, band_names):
        stats = []
        for scene_index in range(self.n_scenes):
            for window in layout:
                stats.append(
                    SceneChunkStats(
                        scene_index=scene_index,
                        chunk_id=window.chunk_id,
                        usable_fraction=1.0,
                        mean_clear=0.9 - 0.05 * scene_index,
                    )
                )
        return tuple(stats)

    def fetch_selected(self, *, grid, plan, band_names, scene_index, chunk_id):
        window = plan.layout[chunk_id]
        h, w = window.shape
        data = np.full(
            (len(band_names), h, w),
            0.3,
            dtype="float32",
        )
        # vary by scene so the best-pixel chooser has something to do
        data += scene_index * 0.001
        quality = np.zeros((h, w), dtype="uint16")
        return Observation(
            data=data,
            quality=quality,
            band_names=band_names,
        )


def _utm_zone(wgs84_bounds):
    west, south, east, north = wgs84_bounds
    centre_lon = (west + east) / 2.0
    centre_lat = (south + north) / 2.0
    zone = int(math.floor((centre_lon + 180.0) / 6.0)) + 1
    return f"EPSG:{32600 + zone if centre_lat >= 0 else 32700 + zone}"


def run_real(
    *,
    wgs84_bounds,
    temporal_range,
    resolution: float,
    chunk_size: int,
    bands: Sequence[str],
    fetch_workers: int,
    cache_dir: Path,
    top_k: int,
    min_usable_fraction: float,
    scout_workers: int,
    band_workers: int,
) -> PhaseTimings:
    source = StacApiSource.earth_search_s2_l2a(
        temporal_ranges=(temporal_range,),
        chunk_size=chunk_size,
        scout_workers=scout_workers,
        band_workers=band_workers,
    )
    config = ProviderConfig(
        cache_dir=cache_dir,
        source=source,
        chunk_size=chunk_size,
        selection_policy=SelectionPolicy(
            top_k=top_k,
            min_usable_fraction=min_usable_fraction,
        ),
        fetch_workers=fetch_workers,
    )
    provider = Provider(config)

    timings = PhaseTimings(chunk_size=chunk_size)

    # Phase: build the grid (deterministic) and list scenes (one STAC search).
    grid = source.resolve_grid(
        wgs84_bounds=wgs84_bounds,
        native_crs="ignored",
        resolution=resolution,
        band_names=bands,
    )
    timings.grid_shape = grid.shape
    t0 = time.perf_counter()
    scenes = source.list_scenes(grid=grid)
    timings.list_scenes = time.perf_counter() - t0
    timings.n_scenes = len(scenes)

    layout = ChunkLayout.from_grid(grid, chunk_size=chunk_size)
    timings.n_chunks = len(layout)

    t1 = time.perf_counter()
    stats = source.scout(grid=grid, layout=layout, band_names=bands)
    timings.scout = time.perf_counter() - t1

    t2 = time.perf_counter()
    plan = select(layout=layout, stats=stats, policy=config.selection_policy)
    timings.select = time.perf_counter() - t2
    timings.n_selected_pairs = sum(len(scenes) for scenes in plan.selected.values())

    t3 = time.perf_counter()
    composite = _run_provider_chunked(
        provider=provider,
        product_id=f"benchmark-{int(time.time())}",
        grid=grid,
        band_names=bands,
        plan=plan,
    )
    timings.fetch_and_compose = time.perf_counter() - t3

    # Persistence step
    request = provider._request_payload(
        grid=grid,
        product_id=composite.product_id,
        band_names=bands,
        composite_period=temporal_range[0][:7],
    )
    from surface_priors.persistence import stable_json_hash

    request_hash = stable_json_hash(request)
    t4 = time.perf_counter()
    provider.store.save(request_hash=request_hash, request=request, composite=composite)
    timings.save = time.perf_counter() - t4

    timings.total = (
        timings.list_scenes + timings.scout + timings.select + timings.fetch_and_compose + timings.save
    )
    return timings


def _run_provider_chunked(*, provider, product_id, grid, band_names, plan):
    """Same logic as Provider._build_chunked, exposed here for timing."""
    from concurrent.futures import ThreadPoolExecutor

    source = provider.config.source

    tasks = [
        (int(scene), int(chunk))
        for chunk, scenes in plan.selected.items()
        for scene in scenes
    ]

    def fetch_one(item):
        scene, chunk = item
        return item, source.fetch_selected(
            grid=grid,
            plan=plan,
            band_names=band_names,
            scene_index=scene,
            chunk_id=chunk,
        )

    cache = {}
    if provider.config.fetch_workers <= 1:
        for task in tasks:
            key, value = fetch_one(task)
            cache[key] = value
    else:
        with ThreadPoolExecutor(max_workers=provider.config.fetch_workers) as pool:
            for key, value in pool.map(fetch_one, tasks):
                cache[key] = value

    def chunk_loader(scene_index, chunk_id):
        return cache.get((int(scene_index), int(chunk_id)))

    compositor = ChunkedCompositor(
        quality_rules=provider.config.compositor.quality_rules,
        output_dtype=provider.config.compositor.output_dtype,
    )
    return compositor.compose(
        product_id=product_id,
        grid=grid,
        band_names=band_names,
        plan=plan,
        chunk_loader=chunk_loader,
    )


def run_synthetic(
    *,
    grid_shape: tuple,
    resolution: float,
    chunk_size: int,
    bands: Sequence[str],
    n_scenes: int,
    top_k: int,
    fetch_workers: int,
) -> PhaseTimings:
    height, width = grid_shape
    grid = GridSpec.from_bounds(
        (0.0, 0.0, width * resolution, height * resolution),
        crs="EPSG:32630",
        resolution=resolution,
        wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
    )
    source = _SyntheticSource(
        grid=grid,
        band_names=bands,
        n_scenes=n_scenes,
        chunk_size=chunk_size,
    )
    timings = PhaseTimings(chunk_size=chunk_size)
    timings.grid_shape = grid.shape
    timings.n_scenes = n_scenes

    layout = ChunkLayout.from_grid(grid, chunk_size=chunk_size)
    timings.n_chunks = len(layout)

    t1 = time.perf_counter()
    stats = source.scout(grid=grid, layout=layout, band_names=bands)
    timings.scout = time.perf_counter() - t1

    t2 = time.perf_counter()
    plan = select(
        layout=layout,
        stats=stats,
        policy=SelectionPolicy(top_k=top_k, min_usable_fraction=0.5),
    )
    timings.select = time.perf_counter() - t2
    timings.n_selected_pairs = sum(len(s) for s in plan.selected.values())

    from concurrent.futures import ThreadPoolExecutor

    tasks = [
        (int(scene), int(chunk))
        for chunk, scenes in plan.selected.items()
        for scene in scenes
    ]

    def fetch(item):
        scene, chunk = item
        return item, source.fetch_selected(
            grid=grid,
            plan=plan,
            band_names=bands,
            scene_index=scene,
            chunk_id=chunk,
        )

    cache: Dict[tuple, Optional[Observation]] = {}
    t3 = time.perf_counter()
    if fetch_workers <= 1:
        for task in tasks:
            key, value = fetch(task)
            cache[key] = value
    else:
        with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
            for key, value in pool.map(fetch, tasks):
                cache[key] = value

    def chunk_loader(scene_index, chunk_id):
        return cache.get((int(scene_index), int(chunk_id)))

    compositor = ChunkedCompositor()
    compositor.compose(
        product_id="synthetic",
        grid=grid,
        band_names=bands,
        plan=plan,
        chunk_loader=chunk_loader,
    )
    timings.fetch_and_compose = time.perf_counter() - t3

    timings.total = timings.scout + timings.select + timings.fetch_and_compose
    return timings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "synthetic"), required=True)
    parser.add_argument("--wgs84-bounds", nargs=4, type=float, default=None)
    parser.add_argument(
        "--temporal-range",
        nargs=2,
        default=["2024-07-01", "2024-07-31"],
        metavar=("START", "END"),
    )
    parser.add_argument("--resolution", type=float, default=20.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-usable-fraction", type=float, default=0.5)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--scout-workers", type=int, default=16)
    parser.add_argument("--band-workers", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path(".benchmark-cache"))
    parser.add_argument(
        "--bands",
        nargs="+",
        default=list(DEFAULT_S2_L2A_BANDS),
    )
    # Synthetic-only knobs
    parser.add_argument("--grid-width", type=int, default=5000)
    parser.add_argument("--grid-height", type=int, default=5000)
    parser.add_argument("--n-scenes", type=int, default=5)
    args = parser.parse_args(argv)

    bands = tuple(args.bands)
    if args.mode == "real":
        if args.wgs84_bounds is None:
            parser.error("--wgs84-bounds is required when --mode real")
        timings = run_real(
            wgs84_bounds=tuple(args.wgs84_bounds),
            temporal_range=tuple(args.temporal_range),
            resolution=args.resolution,
            chunk_size=args.chunk_size,
            bands=bands,
            fetch_workers=args.fetch_workers,
            cache_dir=args.cache_dir,
            top_k=args.top_k,
            min_usable_fraction=args.min_usable_fraction,
            scout_workers=args.scout_workers,
            band_workers=args.band_workers,
        )
    else:
        timings = run_synthetic(
            grid_shape=(args.grid_height, args.grid_width),
            resolution=args.resolution,
            chunk_size=args.chunk_size,
            bands=bands,
            n_scenes=args.n_scenes,
            top_k=args.top_k,
            fetch_workers=args.fetch_workers,
        )

    summary = timings.summary()
    width = max(len(key) for key in summary)
    print(f"\n=== {args.mode.upper()} BENCHMARK ===")
    for key, value in summary.items():
        print(f"  {key:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
