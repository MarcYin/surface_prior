"""Compare adaptive-depth selection against fixed top_k for the Egypt case.

Motivation: the Nile Delta AOI sits on a Sentinel-2 swath edge, so the
bottom-right (SE) corner is under-observed. With ``top_k=3`` that corner
comes out empty in several Julys; ``top_k=6`` fills it but pays for 6
scenes' worth of fetch *everywhere*, not just the thin corner. Adaptive
selection should reach top_k=6 coverage at close to top_k=3 cost by only
deepening the chunks that need it.

For each year this runs build_composite three ways against a warmed disk
cache (so scout is amortised and we compare the fetch/compose work that
actually differs): top_k=3, top_k=6, and adaptive. It reports
fetch+compose time, the number of scenes fetched (the deterministic cost
driver), overall valid-pixel %, and valid-% in the SE quadrant.
"""
from __future__ import annotations

import argparse
import sys
import time

import bestpixel as bp
import numpy as np

BBOX = (30.5, 30.5, 31.6, 31.5)  # Nile Delta, WGS84
RESOLUTION = 60.0
ENDPOINT = "pc"
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]


def _coverage(obs: np.ndarray) -> tuple[float, float]:
    """(overall valid %, SE-quadrant valid %). Rows increase southward,
    cols increase eastward, so the SE corner is the bottom-right block."""
    h, w = obs.shape
    overall = float((obs > 0).mean()) * 100.0
    se = obs[h // 2 :, w // 2 :]
    se_pct = float((se > 0).mean()) * 100.0
    return overall, se_pct


def _build(year: int, month: int, cache: str, **sel) -> dict:
    return bp.build_composite(
        bbox=BBOX,
        datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-28",
        resolution=RESOLUTION,
        endpoint=ENDPOINT,
        disk_cache=cache,
        bands=BANDS,
        **sel,
    )


def _run_config(year: int, month: int, cache: str, label: str, **sel) -> dict:
    t0 = time.perf_counter()
    r = _build(year, month, cache, **sel)
    wall = time.perf_counter() - t0
    obs = np.asarray(r["observation_count"])
    overall, se = _coverage(obs)
    tm = r["timings"]
    fetch_compose = tm.get("fetch", 0.0) + tm.get("compose", 0.0)
    rec = {
        "label": label,
        "wall": wall,
        "fetch_compose": fetch_compose,
        "scout": tm.get("scout", 0.0),
        "read_mpx": tm.get("read_megapixels", 0.0),
        "n_scenes": len(r["source_ids"]),
        "valid_pct": overall,
        "se_valid_pct": se,
    }
    print(
        f"    {label:18s} scenes={rec['n_scenes']:3d}  read={rec['read_mpx']:7.1f}Mpx  "
        f"fetch+compose={fetch_compose:6.2f}s  wall={wall:6.2f}s  "
        f"valid={overall:5.1f}%  SE={se:5.1f}%"
    )
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--cache", default="/tmp/spx-bench-adaptive")
    p.add_argument("--coverage-target", type=float, default=0.95)
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--max-k", type=int, default=6)
    args = p.parse_args(argv)

    print(
        f"AOI={BBOX} res={RESOLUTION}m endpoint={ENDPOINT} month={args.month}\n"
        f"adaptive: coverage_target={args.coverage_target} "
        f"min_k={args.min_k} max_k={args.max_k}\n"
        f"cache={args.cache} (warmed once per year before timing)\n"
    )

    rows = []
    for year in args.years:
        print(f"  year {year}:")
        # Warm the disk cache (scout + COG headers) so the three timed
        # runs compare fetch/compose work, not a one-off cold scout.
        _build(year, args.month, args.cache, top_k=6)

        adaptive_sel = {
            "coverage_target": args.coverage_target, "min_k": args.min_k, "max_k": args.max_k,
        }
        recs = [
            _run_config(year, args.month, args.cache, "top_k=3", top_k=3),
            _run_config(year, args.month, args.cache, "top_k=6", top_k=6),
            _run_config(year, args.month, args.cache, "adaptive L1", **adaptive_sel),
            _run_config(year, args.month, args.cache, "adaptive L2", windowed_fetch=True, **adaptive_sel),
        ]
        rows.append((year, recs))

    print("\n=== summary ===")
    print(f"{'year':>5} {'config':>12} {'scenes':>7} {'read(Mpx)':>10} {'fetch+comp':>11} {'valid%':>7} {'SE%':>7}")
    for year, recs in rows:
        for rec in recs:
            print(
                f"{year:>5} {rec['label']:>12} {rec['n_scenes']:>7} {rec['read_mpx']:>9.1f} "
                f"{rec['fetch_compose']:>10.2f}s {rec['valid_pct']:>6.1f}% {rec['se_valid_pct']:>6.1f}%"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
