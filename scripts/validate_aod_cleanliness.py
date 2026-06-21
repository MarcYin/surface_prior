"""Validate the premise: do hazier days yield dirtier surface priors?

For each July-2020 S2 acquisition date over the Nile Delta we build a SINGLE-DAY
best-pixel surface, measure how well the spectral library explains it (kNN
reconstruction residual = source_fit_rmse, focus on blue/coastal), and correlate
that cleanliness against the day's CAMS+MERRA AOD. A positive correlation means
low-AOD selection produces a cleaner prior -> the feature is worth accepting.

Per-day (not composited-pool) isolates the AOD effect from compositing
redundancy. Runs entirely in the base env (bestpixel + scipy + library cache).

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/validate_aod_cleanliness.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import bestpixel as bp
from scene_aod_gee import ES_SEARCH, init_ee, scene_aod, search_scenes  # same dir on sys.path

import argparse

CACHE = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3"
BANDS = ["coastal", "blue", "green", "red", "nir"]  # library vnir order

# library kNN (VNIR 5-band), distance-weighted reconstruction residual
_Lv = np.load(f"{CACHE}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)
_g = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_Lv = _Lv[_g]
_TREE = cKDTree(_Lv)


def library_residual(Y, k=10):
    """Y:(n,5) reflectance. Returns (total_rmse_per_px, per_band_abs_resid)."""
    d, idx = _TREE.query(Y, k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    recon = (_Lv[idx] * w[:, :, None]).sum(1)
    resid = Y - recon
    total = np.sqrt((resid ** 2).mean(1))
    return total, np.abs(resid)


def day_surface(date, bbox, endpoint="es"):
    r = bp.build_composite(bbox, f"{date}/{date}", resolution=60.0, top_k=8,
                           endpoint=endpoint, bands=BANDS, max_cloud_cover=80.0)
    oc = np.asarray(r["observation_count"])
    valid = oc > 0
    bands = r["bands"]
    Y = np.stack([np.asarray(bands[b]) for b in BANDS], axis=-1)[valid].astype(np.float64) * 1e-4
    Y = Y[np.all((Y > 0) & (Y < 1.2), axis=1)]
    return Y, int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, default=[30.5, 30.5, 31.6, 31.5])
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--label", default="Nile Delta")
    ap.add_argument("--min-valid", type=int, default=50000)
    a = ap.parse_args()
    BBOX = tuple(a.bbox); YEAR = a.year
    import calendar as _cal
    last = _cal.monthrange(YEAR, a.month)[1]
    init_ee()
    dates = {}
    for sid, dt in search_scenes(["sentinel-2-l2a"], BBOX,
                                 f"{YEAR}-{a.month:02d}-01/{YEAR}-{a.month:02d}-{last:02d}",
                                 stac_url=ES_SEARCH):
        dates.setdefault(dt[:10], dt)
    aod = scene_aod(list(dates.items()), BBOX)

    print(f"Single-day S2 surface cleanliness vs AOD — {a.label}, {YEAR}-{a.month:02d}")
    print(f"  {'date':10} {'AOD':>6} {'nvalid':>8} {'fit_RMSE':>9} {'coastal':>8} {'blue':>7}  (DN)")
    MINV = a.min_valid
    rows = []
    for d, t in sorted(dates.items()):
        ad = aod[d]
        if ad["cams"] is None or ad["merra"] is None:
            continue
        A = (ad["cams"] + ad["merra"]) / 2
        try:
            Y, nv = day_surface(d, BBOX)
        except Exception as e:
            print(f"  {d} build failed: {str(e)[:50]}")
            continue
        if len(Y) < MINV:           # need enough cloud-free pixels for a stable median
            print(f"  {d:10} {A:6.3f} {nv:8d}  (too few valid, skipped)")
            continue
        total, perband = library_residual(Y)
        rows.append((d, A, nv, np.median(total)*1e4,
                     np.median(perband[:, 0])*1e4, np.median(perband[:, 1])*1e4))
    rows.sort(key=lambda r: r[1])
    for d, A, nv, fr, co, bl in rows:
        print(f"  {d:10} {A:6.3f} {nv:8d} {fr:9.1f} {co:8.1f} {bl:7.1f}")

    if len(rows) >= 4:
        A = np.array([r[1] for r in rows])
        for j, nm in [(3, "fit_RMSE"), (4, "coastal resid"), (5, "blue resid")]:
            v = np.array([r[j] for r in rows])
            r = np.corrcoef(A, v)[0, 1]
            sl = np.polyfit(A, v, 1)[0]
            print(f"  corr(AOD, {nm:13}) = {r:+.2f}   slope {sl:6.0f} DN per unit AOD")


if __name__ == "__main__":
    main()
