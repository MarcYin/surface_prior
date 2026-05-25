"""Smoke-test the new multi-month/multi-year API."""
import time

import surface_priors_rs as spx

print("=== JJA 2018-2020 (9 periods) ===")
t0 = time.time()
out = spx.build_monthly_composites(
    bbox=(30.5, 30.5, 31.6, 31.5),
    years=[2018, 2019, 2020],
    months=[6, 7, 8],
    resolution=60.0,
    top_k=3,
    endpoint="pc",
    disk_cache="/tmp/spx-periods",
)
wall = time.time() - t0
print(f"wall: {wall:.2f}s   periods: {len(out)}")
for r in out:
    t = r["timings"]
    print(
        f"  {r['year']}-{r['month']:02d}: "
        f"fetch={t.get('fetch', 0):.2f}  "
        f"compose={t.get('compose', 0):.2f}  "
        f"total_per_period={t.get('total', 0):.2f}  "
        f"scenes={len(r['source_ids'])}  "
        f"shape={r['bands']['red'].shape}"
    )
