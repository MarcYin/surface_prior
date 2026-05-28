"""Wall-clock timing: top_k=6 vs adaptive+windowed for the 5-year batch.

The production workload (scripts/gen_egypt_5y_prior.py) is one July composite
per year, 2020-2024, over the Nile Delta, via build_monthly_composites. This
compares the legacy full-coverage selector (top_k=6) against mask-based
adaptive depth + Level-2 windowed fetch on that exact batch, measuring cold
(fresh disk cache) and warm wall time plus fetch volume and coverage.

Scout is selection-independent, so the warm pass isolates the fetch/compose
work that actually differs between the two policies.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import bestpixel as bp
import numpy as np

BBOX = (30.5, 30.5, 31.6, 31.5)
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]

CONFIGS = {
    "top_k=6": {"top_k": 6},
    "adaptive+L2": {"coverage_target": 0.95, "min_k": 2, "max_k": 6, "windowed_fetch": True},
}


def _run(years, month, cache: str, sel: dict) -> tuple[float, list[dict]]:
    t0 = time.perf_counter()
    results = bp.build_monthly_composites(
        bbox=BBOX, years=years, months=[month], resolution=60.0,
        endpoint="pc", disk_cache=cache, bands=BANDS, **sel,
    )
    return time.perf_counter() - t0, results


def _summary(results: list[dict]) -> dict:
    read = sum(r["timings"].get("read_megapixels", 0.0) for r in results)
    fc = sum(r["timings"].get("fetch", 0.0) + r["timings"].get("compose", 0.0) for r in results)
    scout = sum(r["timings"].get("shared_scout", 0.0) for r in results)
    scenes = sum(len(r["source_ids"]) for r in results)
    valids = [float((np.asarray(r["observation_count"]) > 0).mean()) * 100 for r in results]
    return {
        "read": read, "fetch_compose": fc, "scout": scout,
        "scenes": scenes, "valid_min": min(valids), "valid_mean": sum(valids) / len(valids),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--cache-root", default="/tmp/spx-timing")
    args = p.parse_args(argv)

    print(f"AOI={BBOX} res=60m endpoint=pc  batch={args.years} month={args.month}\n")
    rows = []
    for name, sel in CONFIGS.items():
        cache = f"{args.cache_root}-{name.replace('=', '').replace('+', '')}"
        if Path(cache).exists():
            shutil.rmtree(cache)
        print(f"config {name}  ({sel})")
        cold_wall, results = _run(args.years, args.month, cache, sel)
        cold = _summary(results)
        warm_wall, _ = _run(args.years, args.month, cache, sel)
        print(
            f"  cold wall={cold_wall:6.2f}s  warm wall={warm_wall:6.2f}s  "
            f"scenes={cold['scenes']:3d}  read={cold['read']:7.1f}Mpx  "
            f"valid min/mean={cold['valid_min']:.1f}/{cold['valid_mean']:.1f}%"
        )
        rows.append((name, cold_wall, warm_wall, cold))

    print("\n=== summary ===")
    print(f"{'config':>12} {'cold(s)':>8} {'warm(s)':>8} {'scenes':>7} {'read(Mpx)':>10} {'valid%(min)':>11}")
    for name, cw, ww, s in rows:
        print(f"{name:>12} {cw:>8.2f} {ww:>8.2f} {s['scenes']:>7} {s['read']:>10.1f} {s['valid_min']:>10.1f}%")
    if len(rows) == 2:
        (_, cw0, ww0, _), (_, cw1, ww1, _) = rows
        print(f"\nwarm speedup (adaptive+L2 vs top_k=6): {ww0 / ww1:.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
