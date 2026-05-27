#!/usr/bin/env bash
# Per-phase diagnostic on 1 CPU. Prints Rust + Python phase timings
# side-by-side so we can locate the gap.
set -uo pipefail
cd /home/users/marcyin/surface_prior

CACHE_RS=/tmp/spx-diag-rs
CACHE_PY=/tmp/spx-diag-py
rm -rf "$CACHE_RS" "$CACHE_PY"

echo "node: $(hostname), cpus: $(nproc)"
echo ""

BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build

echo "===== RUST cold (1 cpu) ====="
$BIN --datetime 2024-07-01/2024-07-31 --top-k 3 --concurrency 600 --no-write --disk-cache "$CACHE_RS" 2>&1 | grep -E "list_|scout|partition|fetch|compose|TOTAL"
echo ""

echo "===== RUST warm (1 cpu) ====="
$BIN --datetime 2024-07-01/2024-07-31 --top-k 3 --concurrency 600 --no-write --disk-cache "$CACHE_RS" 2>&1 | grep -E "list_|scout|partition|fetch|compose|TOTAL"
echo ""

echo "===== PYTHON cold + warm (1 cpu) ====="
PYTHONPATH=src python -c "
import time, sys
from surface_priors.sources.stac_api import StacApiSource
from surface_priors.composite import ChunkedCompositor
from surface_priors.provider import _scene_fetcher_for
from surface_priors.chunks import ChunkLayout
from surface_priors.selection import SelectionPolicy, select
from surface_priors.types import DEFAULT_S2_L2A_BANDS

for label in ('cold', 'warm'):
    t0 = time.perf_counter()
    src = StacApiSource.earth_search_s2_l2a(
        temporal_ranges=[('2024-07-01', '2024-07-31')], chunk_size=512,
    )
    grid = src.resolve_grid(
        wgs84_bounds=(30.5, 30.5, 31.6, 31.5),
        native_crs='EPSG:32636', resolution=60.0, band_names=(),
    )
    layout = ChunkLayout.from_grid(grid, chunk_size=512)
    bands = list(DEFAULT_S2_L2A_BANDS)
    t1 = time.perf_counter()
    scenes = src.list_scenes(grid=grid)
    t_list = time.perf_counter() - t1
    t1 = time.perf_counter()
    partition = src.tile_partition(grid=grid, layout=layout)
    t_part = time.perf_counter() - t1
    t1 = time.perf_counter()
    stats = src.scout(grid=grid, layout=layout, band_names=bands)
    t_scout = time.perf_counter() - t1
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=3), partition=partition)
    t1 = time.perf_counter()
    fetch_scene = _scene_fetcher_for(source=src, grid=grid, plan=plan, band_names=bands)
    ChunkedCompositor().compose_pipelined(
        product_id='diag', grid=grid, band_names=bands, plan=plan,
        fetch_scene=fetch_scene, fetch_workers=32,
    )
    t_fc = time.perf_counter() - t1
    total = time.perf_counter() - t0
    sys.stdout.write(f'{label}: total={total:.2f}s  list={t_list:.2f}s  scout={t_scout:.2f}s  fetch+compose={t_fc:.2f}s\n')
    sys.stdout.flush()
"
