"""Definitive closed-loop AOD accuracy with the best configuration found.

Ingredients established across the thread:
  anchor   = red+nir+swir (joint kNN; ~2x better than nir+swir, AOD-robust)
  predict  = coastal,blue,green clean surface
  solve    = coastal(443)+blue(490)  (best aerosol SNR)
  debias   = per-band gain calibrated vs truth (the AERONET-tuned-surface analog)
  pooling  = coarse-cell AOD (averages the random surface-prediction error)

Closed loop on a real clean composite block forwarded to TOA at a KNOWN AOD
(constant levels + a spatial gradient). Reports per-pixel and cell-pooled AOD
accuracy -> can it estimate AOD, and how well?

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/aod_retrieval_final.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

CACHE = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3"
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])
SOLVE = [0, 1]                      # coastal, blue
SZA, VZA, RAA = 28.0, 5.0, 120.0
RNG = np.random.default_rng(0)
HLS = "egypt_monthly_5y_hls/egypt_2022-07_hls.tif"

_Lv = np.load(f"{CACHE}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)
_Ls = np.load(f"{CACHE}/source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float64)
_g = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_Lv, _Ls = _Lv[_g], _Ls[_g]
_ANCHOR = np.column_stack([_Lv[:, 3], _Lv[:, 4], _Ls[:, 0], _Ls[:, 1]])   # red,nir,swir16,swir22
_VIS = _Lv[:, :3]                                                          # coastal,blue,green
_TREE = cKDTree(_ANCHOR)


def predict_vis(rows7, k=10):
    q = rows7[:, [3, 4, 5, 6]]
    d, idx = _TREE.query(q, k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (_VIS[idx] * w[:, :, None]).sum(1)


def rt(wl, aot, alpha=1.2, g=0.65, w0=0.95):
    lam = np.asarray(wl) / 1000.0
    tr = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4)
    ta = np.asarray(aot) * (lam / 0.55) ** (-alpha)
    mus, muv = np.cos(np.radians(SZA)), np.cos(np.radians(VZA))
    cosT = -mus*muv + np.sin(np.radians(SZA))*np.sin(np.radians(VZA))*np.cos(np.radians(RAA))
    Pr = 0.75*(1+cosT**2); Pa = (1-g**2)/(1+g**2-2*g*cosT)**1.5
    path = (tr*Pr + w0*ta*Pa)/(4*mus*muv)
    T = np.exp(-(0.5*tr + ta*(1-w0*g))*(1/mus+1/muv))
    return path, T, 0.92*tr + 0.33*ta


def forward(rho, aot, wl):
    p, T, S = rt(wl, aot); return p + T*rho/(1-S*rho)


def fit_debias(cal7):
    pred = predict_vis(cal7)
    a = np.empty(3); c = np.empty(3)
    for j in range(3):
        A = np.vstack([pred[:, j], np.ones(len(pred))]).T
        (a[j], c[j]), *_ = np.linalg.lstsq(A, cal7[:, j], rcond=None)
    return a, c


def solve_aod(toa, vis, grid):
    R = np.empty((len(grid), len(toa)))
    for i, ac in enumerate(grid):
        R[i] = ((forward(vis[:, SOLVE], ac, WL[SOLVE]) - toa[:, SOLVE]) ** 2).sum(1)
    return R


def load_block(r0, c0, H, W):
    with rasterio.open(HLS) as ds:
        a = ds.read(list(range(1, 8)), window=Window(c0, r0, W, H)).astype(np.float32)
    return np.clip(a.reshape(7, -1).T * 1e-4, 1e-4, 0.95).astype(np.float64), (H, W)


def main():
    grid = np.round(np.arange(0.02, 1.001, 0.02), 3)
    # calibration sample (clean) for the debias gain
    cal, _ = load_block(300, 300, 200, 200)
    a_g, c_g = fit_debias(cal[RNG.choice(len(cal), 8000, replace=False)])
    print(f"debias gain coastal x{a_g[0]:.3f}+{c_g[0]*1e4:.0f}  blue x{a_g[1]:.3f}+{c_g[1]*1e4:.0f}")

    rho, (H, W) = load_block(800, 800, 360, 360)   # disjoint test block
    # in-scene debias (optimistic ceiling: surface model perfectly tuned to scene)
    a_i, c_i = fit_debias(rho[RNG.choice(len(rho), 8000, replace=False)])

    print("\nA. CONSTANT AOD, 60-cell pooled (cross-scene debias / in-scene debias):")
    print(f"  {'AOT*':>5} {'pooled med (x-scene)':>20} {'RMSE':>7} {'pooled med (in-scene)':>22} {'RMSE':>7}")
    C = 60
    def pooled(toa, vis):
        R = solve_aod(toa, vis, grid)
        Rc = R.reshape(len(grid), H, W); cell = np.empty((H, W))
        for i in range(0, H, C):
            for j in range(0, W, C):
                cell[i:i+C, j:j+C] = grid[np.argmin(Rc[:, i:i+C, j:j+C].reshape(len(grid), -1).mean(1))]
        return cell.ravel()
    for aT in (0.10, 0.30, 0.50, 0.80):
        toa = forward(rho, aT, WL) + RNG.normal(0, 0.003, rho.shape)
        cx = pooled(toa, predict_vis(toa) * a_g + c_g)
        ci = pooled(toa, predict_vis(toa) * a_i + c_i)
        print(f"  {aT:5.2f} {np.median(cx):20.2f} {np.sqrt(np.mean((cx-aT)**2)):7.3f} "
              f"{np.median(ci):22.2f} {np.sqrt(np.mean((ci-aT)**2)):7.3f}")

    print("\nB. SPATIAL GRADIENT AOD 0.10 -> 0.60 across the block:")
    col = np.tile(np.arange(W), H)
    at = 0.10 + 0.50 * col / (W - 1)
    toa = forward(rho, at[:, None], WL) + RNG.normal(0, 0.003, rho.shape)
    vis = predict_vis(toa) * a_g + c_g
    R = solve_aod(toa, vis, grid)
    px = grid[np.argmin(R, 0)]
    Rc = R.reshape(len(grid), H, W); cell = np.empty((H, W))
    for i in range(0, H, C):
        for j in range(0, W, C):
            cell[i:i+C, j:j+C] = grid[np.argmin(Rc[:, i:i+C, j:j+C].reshape(len(grid), -1).mean(1))]
    cell = cell.ravel()
    def stat(x): return np.sqrt(np.mean((x-at)**2)), np.mean(x-at), np.corrcoef(x, at)[0, 1]
    pr, pb, pc = stat(px); cr, cb, cc = stat(cell)
    print(f"  per-pixel : RMSE {pr:.3f}  bias {pb:+.3f}  corr {pc:.2f}")
    print(f"  {C}-cell   : RMSE {cr:.3f}  bias {cb:+.3f}  corr {cc:.2f}")
    lo = cell[col < W/4].mean(); hi = cell[col > 3*W/4].mean()
    print(f"  recovered gradient: left {lo:.2f} (truth ~0.16)  right {hi:.2f} (truth ~0.54)")


if __name__ == "__main__":
    main()
