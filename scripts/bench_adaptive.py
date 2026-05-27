"""Cold-cache benchmark: fixed top_k vs adaptive-depth selection.

Each config runs in its own subprocess (so the per-process COG LRU and
result arrays are released between configs, avoiding tmpfs/RAM exhaustion
on the 2 GB /tmp). Reports wall time, total scenes fetched (the
network-bound cost), and per-period coverage.

  python scripts/bench_adaptive.py                # driver: run all configs
  python scripts/bench_adaptive.py --run <name>   # one config (internal)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

BBOX = (30.5, 30.5, 31.6, 31.5)
YEARS = [2020, 2021, 2022, 2023, 2024]
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]

# name -> kwargs for build_monthly_composites
CONFIGS = {
    "top_k=3": {"top_k": 3},
    "top_k=6": {"top_k": 6},
    "adaptive": {"coverage_target": 0.98, "min_k": 2, "max_k": 8},
    "adaptive+windowed": {"coverage_target": 0.98, "min_k": 2, "max_k": 8, "windowed_fetch": True},
}


def run_one(name: str) -> dict:
    import numpy as np
    import surface_priors_rs as spx
    cache = f"/tmp/bench-adapt-{name.replace('=', '').replace('+', '-')}"
    shutil.rmtree(cache, ignore_errors=True)
    t0 = time.time()
    res = spx.build_monthly_composites(
        bbox=BBOX, years=YEARS, months=[7], resolution=60.0,
        endpoint="pc", disk_cache=cache, bands=BANDS, **CONFIGS[name])
    wall = time.time() - t0
    res = sorted(res, key=lambda r: (r["year"], r["month"]))
    covs = [float((np.asarray(r["observation_count"]) > 0).mean()) for r in res]
    scenes = sum(len(r["source_ids"]) for r in res)
    mpx = sum(float(r["timings"].get("read_megapixels", 0.0)) for r in res)
    shutil.rmtree(cache, ignore_errors=True)
    return {"name": name, "wall": wall, "scenes": scenes, "read_mpx": mpx,
            "min_cov": min(covs), "covs": covs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()

    if args.run:
        print(json.dumps(run_one(args.run)))
        return 0

    print(f"AOI={BBOX}  years={YEARS}  month=7  60m  endpoint=pc  (cold)\n")
    rows = []
    for name in args.configs:
        p = subprocess.run([sys.executable, __file__, "--run", name],
                           capture_output=True, text=True)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        if p.returncode != 0 or not line:
            print(f"  {name:18} FAILED rc={p.returncode}\n{p.stderr[-500:]}")
            continue
        r = json.loads(line)
        rows.append(r)
        covs = " ".join(f"{c:.0%}" for c in r["covs"])
        print(f"  {r['name']:18} wall={r['wall']:6.2f}s  scenes={r['scenes']:3d}  "
              f"read={r['read_mpx']:6.0f}Mpx  min_cov={r['min_cov']:.1%}  [{covs}]")
    if rows:
        base = next((x for x in rows if x["name"] == "top_k=6"), None)
        if base:
            print("\n  vs top_k=6 baseline:")
            for r in rows:
                ds = r["scenes"] / base["scenes"] - 1
                dm = r["read_mpx"] / base["read_mpx"] - 1 if base["read_mpx"] else 0
                print(f"    {r['name']:18} scenes {ds:+.0%}   read-volume {dm:+.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
