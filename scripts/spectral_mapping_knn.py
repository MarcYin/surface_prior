"""Spectral-library kNN mapping: explanatory power + aerosol closed-loop.

Uses the `spectral_library` package's kNN SpectralMapper (k=5,
distance_weighted_mean over a scipy cKDTree of 77k real surface spectra) — the
same engine SIAC uses to build its aerosol-solve surface prior. The mapper's
`source_fit_rmse` is the per-pixel residual between the query and its
kNN-reconstruction in source-band space: the proper "how well does the library
explain this pixel" metric (a local-manifold fit, tighter than a global PCA
hull, smoother than 1-NN).

Part 1: source_fit_rmse over the HLS and S2 priors, VNIR(5) vs VNIR+SWIR(7).
Part 2: synthetic closed loop — forward BOA->TOA at known AOT*, then use the
        kNN source_fit_rmse as the cost to recover AOT.

RUN WITH SIAC's env (has the package + rasterio):
  ~/SIAC/.pixi/envs/default/bin/python scripts/spectral_mapping_knn.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from spectral_library import SpectralMapper
from spectral_library.distribution import resolve_prepared_library_root

# Prior GeoTIFF bands 1..7 == S2A schema order [ultra_blue, blue, green, red,
# nir, swir1, swir2]; coastal->ultra_blue, swir16->swir1, swir22->swir2.
NAMES = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])
REFL = 1e-4
SZA, VZA, RAA = 28.0, 5.0, 120.0
RNG = np.random.default_rng(0)
MAPPER = SpectralMapper(resolve_prepared_library_root(), verify_checksums=False)


def knn_fit(rows, valid):
    """Return per-row source_fit_rmse from the kNN distance-weighted mapper."""
    r = MAPPER.map_reflectance_batch_arrays_ndarray(
        source_sensor="sentinel-2a_msi",
        reflectance_rows=np.ascontiguousarray(rows, np.float64),
        valid_mask_rows=np.ascontiguousarray(valid, bool),
        output_mode="target_sensor", target_sensor="sentinel-2a_msi",
        k=5, min_valid_bands=3, neighbor_estimator="distance_weighted_mean",
        knn_backend="scipy_ckdtree", knn_eps=0.0,
    )
    return np.asarray(r.source_fit_rmse, np.float32)


def load(tif, n):
    with rasterio.open(tif) as ds:
        a = np.stack([ds.read(b) for b in range(1, 8)]).astype(np.float32)
    valid = np.all((a > 0) & (a != 65535), axis=0)
    Y = (a[:, valid].T * REFL).astype(np.float64)
    idx = RNG.choice(len(Y), size=min(n, len(Y)), replace=False)
    return Y[idx]


# ---- compact 6S-style RT (same as aerosol_closed_loop; forward==inverse) ----
def rt(wl, aot, alpha=1.2, g=0.65, w0=0.95):
    lam = np.asarray(wl) / 1000.0
    tr = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4)
    ta = aot * (lam / 0.55) ** (-alpha)
    mus, muv = np.cos(np.radians(SZA)), np.cos(np.radians(VZA))
    cosT = -mus*muv + np.sin(np.radians(SZA))*np.sin(np.radians(VZA))*np.cos(np.radians(RAA))
    Pr = 0.75*(1+cosT**2); Pa = (1-g**2)/(1+g**2-2*g*cosT)**1.5
    path = (tr*Pr + w0*ta*Pa)/(4*mus*muv)
    T = np.exp(-(0.5*tr + ta*(1-w0*g))*(1/mus+1/muv))
    S = 0.92*tr + 0.33*ta
    return path, T, S


def forward(rho, aot, wl):
    p, T, S = rt(wl, aot); return p + T*rho/(1-S*rho)


def correct(toa, aot, wl):
    p, T, S = rt(wl, aot); u = toa - p; return u/(T+S*u)


def part1():
    print("=== PART 1: library kNN explanatory power (source_fit_rmse, DN) ===")
    print(f"{'prior':4} {'bands':9} {'median':>7} {'mean':>7} {'p90':>7} {'p99':>7}")
    out = {}
    for nm, d in [("S2", "egypt_5y_prior"), ("HLS", "egypt_5y_prior_hls")]:
        Y = load(Path(d) / "egypt_2022-07_prior.tif", 100000)
        out[nm] = Y
        for label, vmask in [("VNIR(5)", [1,1,1,1,1,0,0]), ("VNIR+SWIR", [1,1,1,1,1,1,1])]:
            V = np.tile(np.array(vmask, bool), (len(Y), 1))
            fr = knn_fit(Y, V) * 1e4
            print(f"{nm:4} {label:9} {np.median(fr):7.1f} {fr.mean():7.1f} "
                  f"{np.percentile(fr,90):7.1f} {np.percentile(fr,99):7.1f}")
    return out


def part2(truth):
    print("\n=== PART 2: closed-loop AOT recovery with kNN cost ===")
    print("truth=HLS 2022 BOA, RT=compact 6S-style SS, noise=0.003 refl, k=5 dwm")
    wl = WL
    rho = np.clip(truth["HLS"][RNG.choice(len(truth["HLS"]), 4000, replace=False)], 1e-4, 0.95)
    grid = np.round(np.arange(0.02, 0.81, 0.04), 3)
    for label, vmask in [("VNIR(5)", [1,1,1,1,1,0,0]), ("VNIR+SWIR(7)", [1,1,1,1,1,1,1])]:
        V = np.tile(np.array(vmask, bool), (len(rho), 1))
        print(f"\n  --- cost = kNN source_fit_rmse on {label} ---")
        print(f"  {'AOT*':>5} {'scene_AOT':>9} {'px_med':>7} {'px_MAD':>7}")
        for aT in (0.10, 0.30, 0.50):
            toa = forward(rho, aT, wl) + RNG.normal(0, 0.003, rho.shape)
            R = np.empty((len(grid), len(rho)), np.float32)
            for i, ac in enumerate(grid):
                R[i] = knn_fit(correct(toa, ac, wl), V)
            scene = grid[np.argmin(R.mean(1))]
            px = grid[np.argmin(R, 0)]
            line = (f"  {aT:5.2f} {scene:9.2f} {np.median(px):7.2f} "
                    f"{np.median(np.abs(px-np.median(px))):7.2f}")
            print(line)
            if abs(aT-0.30) < 1e-6:
                probe=[0.02,0.1,0.2,0.3,0.4,0.5,0.7,0.8]
                c=np.array([R.mean(1)[int(np.argmin(np.abs(grid-p)))] for p in probe]); c/=c.min()
                print("     scene-cost (AOT*=.30,/min): "+" ".join(f"{p}:{v:.2f}" for p,v in zip(probe,c)))


if __name__ == "__main__":
    truth = part1()
    part2(truth)
