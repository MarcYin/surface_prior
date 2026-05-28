"""Benchmark 5-year monthly composites via the Python API.

Matches the CLI bench (bench_pc_full.sh): Nile Delta AOI, July of
each year 2020..2024, PC endpoint, 60 m, top_k=3. Difference: no
GeoTIFF writes — arrays stay in-process as numpy.

Runs sequentially and in parallel (threads). GIL is released inside
build_composite via py.allow_threads(), so the 5 calls truly overlap.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BBOX = (30.5, 30.5, 31.6, 31.5)
YEARS = (2020, 2021, 2022, 2023, 2024)
CACHE = "/tmp/spx-py-5y"

# B1 (coastal), B2 (blue), B3 (green), B4 (red),
# B8 (nir), B11 (swir16), B12 (swir22).
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]


def one(year: int, concurrency: int):
    # Import inside the function so it works under spawn-based
    # multiprocessing (children re-import on fork-spawn boundary).
    import bestpixel as bp
    t0 = time.time()
    out = bp.build_composite(
        bbox=BBOX,
        datetime=f"{year}-07-01/{year}-07-31",
        resolution=60.0,
        top_k=3,
        max_cloud_cover=80.0,
        concurrency=concurrency,
        endpoint="pc",
        disk_cache=CACHE,
        bands=BANDS,
    )
    dt = time.time() - t0
    h, w = out["bands"]["red"].shape
    return year, dt, h, w, out["timings"]


def run_sequential(concurrency):
    print("=== sequential ===")
    t0 = time.time()
    for y in YEARS:
        yr, dt, h, w, tim = one(y, concurrency)
        print(f"  {yr}: {dt:.2f}s   grid={h}x{w}   "
              f"fetch={tim.get('fetch', 0):.2f}  compose={tim.get('compose', 0):.2f}")
    wall = time.time() - t0
    print(f"  wall: {wall:.2f}s")
    return wall


def run_parallel(concurrency, workers=5):
    print(f"=== parallel x{workers} (threads) ===")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, y, concurrency) for y in YEARS]
        results = [f.result() for f in futs]
    wall = time.time() - t0
    for yr, dt, h, w, tim in results:
        print(f"  {yr}: {dt:.2f}s   grid={h}x{w}   "
              f"fetch={tim.get('fetch', 0):.2f}  compose={tim.get('compose', 0):.2f}")
    print(f"  wall: {wall:.2f}s")
    return wall


def run_parallel_proc(concurrency, workers=5):
    # Subprocess-per-year: matches the CLI bench (5 separate
    # `spx-build` processes), with independent tokio runtimes and
    # connection pools.
    print(f"=== parallel x{workers} (processes) ===")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, y, concurrency) for y in YEARS]
        results = [f.result() for f in futs]
    wall = time.time() - t0
    for yr, dt, h, w, tim in results:
        print(f"  {yr}: {dt:.2f}s   grid={h}x{w}   "
              f"fetch={tim.get('fetch', 0):.2f}  compose={tim.get('compose', 0):.2f}")
    print(f"  wall: {wall:.2f}s")
    return wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("seq", "par", "proc", "all"), default="all")
    p.add_argument("--concurrency", type=int, default=120)
    p.add_argument("--iters", type=int, default=2,
                   help="cold + warm passes (cache is wiped between iters)")
    args = p.parse_args()

    import shutil
    for it in range(1, args.iters + 1):
        shutil.rmtree(CACHE, ignore_errors=True)
        print(f"\n### iter {it} (cold) ###")
        if args.mode in ("seq", "all"):
            run_sequential(args.concurrency)
            shutil.rmtree(CACHE, ignore_errors=True)
        if args.mode in ("par", "all"):
            run_parallel(args.concurrency)
            shutil.rmtree(CACHE, ignore_errors=True)
        if args.mode in ("proc", "all"):
            run_parallel_proc(args.concurrency)


if __name__ == "__main__":
    main()
