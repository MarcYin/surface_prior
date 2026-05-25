"""Quick regression: build a 1-month composite via each endpoint."""
import time

import surface_priors_rs as spx

for endpoint in ["pc", "hls", "mcd43a4"]:
    res = 500.0 if endpoint == "mcd43a4" else 60.0
    t0 = time.time()
    out = spx.build_composite(
        bbox=(30.5, 30.5, 31.6, 31.5),
        datetime="2024-07-01/2024-07-31",
        resolution=res,
        top_k=3,
        endpoint=endpoint,
        disk_cache=f"/tmp/spx-regress-{endpoint}",
    )
    dt = time.time() - t0
    shape = out["bands"]["red"].shape
    print(
        f"{endpoint:10}: {dt:.2f}s  bands={len(out['band_names'])}  "
        f"scenes={len(out['source_ids'])}  shape={shape}"
    )
