"""Absolute-bias test: does Sen2Cor's VISIBLE correction drift with AOD?

The library-residual (shape) test found no AOD effect even at AOD~1. This is the
more sensitive ABSOLUTE test. For each day we predict the clean visible from the
aerosol-free anchor (red+nir+swir) via the global spectral library -- an
estimate that is AOD-robust and adapts to surface change. Then:

  signed_bias(band) = median(observed_Sen2Cor_visible - predicted_clean_visible)

The library carries a constant (AOD-independent) region offset, so any
*correlation* of signed_bias with AOD isolates Sen2Cor's AOD-dependent
correction error (over-correct -> bias falls with AOD; under-correct -> rises).

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/igp_absolute_bias.py \
      --bbox 79.5 26.0 81.0 27.5 --year 2022 --month 11 --label IGP
"""
from __future__ import annotations

import argparse
import calendar

import bestpixel as bp
import numpy as np
from scene_aod_gee import ES_SEARCH, init_ee, scene_aod, search_scenes
from scipy.spatial import cKDTree

CACHE = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3"
BANDS = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
VISN = ["coastal", "blue", "green"]

_Lv = np.load(f"{CACHE}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)
_Ls = np.load(f"{CACHE}/source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float64)
_g = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_Lv, _Ls = _Lv[_g], _Ls[_g]
_ANCHOR = np.column_stack([_Lv[:, 3], _Lv[:, 4], _Ls[:, 0], _Ls[:, 1]])   # red,nir,swir16,swir22
_VIS = _Lv[:, :3]
_TREE = cKDTree(_ANCHOR)


def predict_vis(rows7, k=10):
    q = rows7[:, [3, 4, 5, 6]]
    d, idx = _TREE.query(q, k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (_VIS[idx] * w[:, :, None]).sum(1)


def day_surface(date, bbox):
    r = bp.build_composite(bbox, f"{date}/{date}", resolution=60.0, top_k=8,
                           endpoint="es", bands=BANDS, max_cloud_cover=80.0)
    oc = np.asarray(r["observation_count"]); valid = oc > 0
    Y = np.stack([np.asarray(r["bands"][b]) for b in BANDS], -1)[valid].astype(np.float64) * 1e-4
    return Y[np.all((Y > 0) & (Y < 1.2), axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--label", default="AOI")
    ap.add_argument("--min-valid", type=int, default=30000)
    a = ap.parse_args()
    bbox = tuple(a.bbox); last = calendar.monthrange(a.year, a.month)[1]
    init_ee()
    dates = {}
    for _sid, dt in search_scenes(["sentinel-2-l2a"], bbox,
                                 f"{a.year}-{a.month:02d}-01/{a.year}-{a.month:02d}-{last:02d}",
                                 stac_url=ES_SEARCH):
        dates.setdefault(dt[:10], dt)
    aod = scene_aod(list(dates.items()), bbox)

    print(f"Absolute Sen2Cor visible bias vs AOD — {a.label}, {a.year}-{a.month:02d}")
    print("  signed_bias = median(observed - library-predicted clean visible), DN")
    print(f"  {'date':10} {'AOD':>6} {'nvalid':>8} {'coastal':>8} {'blue':>7} {'green':>7}")
    rows = []
    for d, _t in sorted(dates.items()):
        ad = aod[d]
        if ad["cams"] is None or ad["merra"] is None:
            continue
        A = (ad["cams"] + ad["merra"]) / 2
        try:
            Y = day_surface(d, bbox)
        except Exception as e:
            print(f"  {d} build failed: {str(e)[:45]}"); continue
        if len(Y) < a.min_valid:
            continue
        pred = predict_vis(Y)
        bias = [np.median(Y[:, j] - pred[:, j]) * 1e4 for j in range(3)]
        rows.append((d, A, len(Y), *bias))
    rows.sort(key=lambda r: r[1])
    for d, A, nv, co, bl, gr in rows:
        print(f"  {d:10} {A:6.3f} {nv:8d} {co:8.1f} {bl:7.1f} {gr:7.1f}")

    if len(rows) >= 4:
        A = np.array([r[1] for r in rows])
        print()
        for j, nm in enumerate(VISN):
            v = np.array([r[3 + j] for r in rows])
            r = np.corrcoef(A, v)[0, 1]; sl = np.polyfit(A, v, 1)[0]
            print(f"  corr(AOD, {nm:8} bias) = {r:+.2f}   slope {sl:7.0f} DN per unit AOD")


if __name__ == "__main__":
    main()
