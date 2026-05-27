"""Profile PC S2 endpoint timing: where does the cold call spend its time?"""
import shutil
import time

import bestpixel as spx

CACHE = "/tmp/spx-profile-pc"


def run(label, *, cold):
    if cold:
        shutil.rmtree(CACHE, ignore_errors=True)
    t0 = time.time()
    out = spx.build_composite(
        bbox=(30.5, 30.5, 31.6, 31.5),
        datetime="2024-07-01/2024-07-31",
        resolution=60.0,
        top_k=3,
        endpoint="pc",
        disk_cache=CACHE,
    )
    dt = time.time() - t0
    t = out["timings"]
    rust_total = t.get('total', 0)
    sum_phases = sum(t.get(k, 0) for k in ('list_scenes','sign','scout','partition','fetch','compose'))
    print(
        f"{label:24} pywall={dt:5.2f}s  rust_total={rust_total:.2f}s  sum_phases={sum_phases:.2f}s  | "
        f"list={t.get('list_scenes', 0):.2f}  "
        f"sign={t.get('sign', 0):.2f}  "
        f"scout={t.get('scout', 0):.2f}  "
        f"part={t.get('partition', 0):.2f}  "
        f"fetch={t.get('fetch', 0):.2f}  "
        f"compose={t.get('compose', 0):.2f}  "
        f"scenes={len(out['source_ids'])}"
    )


print("=== fresh process, multiple sequential calls ===")
run("call 1 (cold)", cold=True)
run("call 2 (cache hot)", cold=False)
run("call 3 (cache hot)", cold=False)

# Now also run the same datetime window 5 times in a fresh interpreter is
# the equivalent of the bench's per-year cold-but-shared-process timing.
print()
print("=== fresh process, 5 different datetime windows (mimics 5-year seq) ===")
shutil.rmtree(CACHE, ignore_errors=True)
years = [2020, 2021, 2022, 2023, 2024]
t0 = time.time()
for y in years:
    yt0 = time.time()
    out = spx.build_composite(
        bbox=(30.5, 30.5, 31.6, 31.5),
        datetime=f"{y}-07-01/{y}-07-31",
        resolution=60.0,
        top_k=3,
        endpoint="pc",
        disk_cache=CACHE,
    )
    t = out["timings"]
    print(
        f"  {y}: total={time.time() - yt0:.2f}s  "
        f"list={t.get('list_scenes', 0):.2f}  "
        f"sign={t.get('sign', 0):.2f}  "
        f"scout={t.get('scout', 0):.2f}  "
        f"part={t.get('partition', 0):.2f}  "
        f"fetch={t.get('fetch', 0):.2f}"
    )
print(f"  wall: {time.time() - t0:.2f}s")
