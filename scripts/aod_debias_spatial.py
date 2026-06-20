"""Close the gap: debias the surface prediction + spatial pooling, then solve
AOD from the high-SNR 443 (deep-blue analog) band alone.

Builds on aod_from_anchor: the 443 band has the best aerosol SNR but raw
retrieval is biased low because the kNN-predicted surface is systematically
bright (scene darker than library mean). Operational algos remove this with a
calibrated surface model (Deep Blue's AERONET-tuned DB; SIAC's gain). Here:

  A. DEBIAS  - fit a per-band linear surface correction on calibration pixels
               (true vs predicted), apply to held-out test pixels. Does 443-only
               AOD then land on truth?
  B. POOLING - on a contiguous image block, solve one AOD per coarse cell by
               pooling the cost (AOD is spatially smooth; per-pixel surface-
               prediction error is random -> averages out).

  ~/SIAC/.pixi/envs/default/bin/python scripts/aod_debias_spatial.py
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window

from aod_from_anchor import WL, forward, joint_predict_vis  # shared RT + joint kNN

REFL = 1e-4
RNG = np.random.default_rng(1)
HLS = "egypt_5y_prior_hls/egypt_2022-07_prior.tif"


def fit_debias(cal_true):
    """Per-band linear map true ~ a*pred + c on calibration pixels (surface-
    model calibration; in practice fit vs AERONET-corrected truth)."""
    pred = joint_predict_vis(cal_true)
    a = np.empty(3); c = np.empty(3)
    for j in range(3):
        A = np.vstack([pred[:, j], np.ones(len(pred))]).T
        (a[j], c[j]), *_ = np.linalg.lstsq(A, cal_true[:, j], rcond=None)
    return a, c


def solve(toa, vis, grid, bands):
    wl_s = WL[bands]
    R = np.empty((len(grid), len(toa)))
    for i, ac in enumerate(grid):
        R[i] = ((forward(vis[:, bands], ac, wl_s) - toa[:, bands]) ** 2).sum(1)
    return R


def part_a_debias():
    with rasterio.open(HLS) as ds:
        a = np.stack([ds.read(b) for b in range(1, 8)]).astype(np.float32)
    v = np.all((a > 0) & (a != 65535), axis=0)
    Y = np.clip((a[:, v].T * REFL).astype(np.float64), 1e-4, 0.95)
    idx = RNG.choice(len(Y), 8000, replace=False)
    cal, test = Y[idx[:4000]], Y[idx[4000:]]
    a_g, c_g = fit_debias(cal)
    print(f"A. DEBIAS (443-only solve, JOINT anchor)   gain coastal: x{a_g[0]:.3f} +{c_g[0]*1e4:.0f}DN")
    grid = np.round(np.arange(0.02, 1.001, 0.02), 3)
    print(f"  {'AOT*':>5} {'raw scene/px':>14} {'debiased scene/px':>18}")
    for aT in (0.10, 0.30, 0.50):
        toa = forward(test, aT, WL) + RNG.normal(0, 0.003, test.shape)
        vis = joint_predict_vis(toa)
        visd = vis * a_g + c_g
        out = []
        for V in (vis, visd):
            R = solve(toa, V, grid, [0])           # 443 only
            out.append((grid[np.argmin(R.mean(1))], np.median(grid[np.argmin(R, 0)])))
        print(f"  {aT:5.2f} {out[0][0]:6.2f}/{out[0][1]:.2f}    {out[1][0]:11.2f}/{out[1][1]:.2f}")
    return a_g, c_g


def part_b_spatial(a_g, c_g):
    # contiguous central (vegetated) block, fully valid
    r0, c0, H, W = 700, 700, 420, 420
    with rasterio.open(HLS) as ds:
        blk = ds.read(list(range(1, 8)), window=Window(c0, r0, W, H)).astype(np.float32)
    rho = np.clip(blk.reshape(7, -1).T * REFL, 1e-4, 0.95).astype(np.float64)   # (N,7)
    n = len(rho)
    # known AOD: smooth left->right gradient 0.15 -> 0.45
    col = np.tile(np.arange(W), H)
    aod_true = 0.15 + 0.30 * col / (W - 1)
    toa = forward(rho, aod_true[:, None], WL) + RNG.normal(0, 0.003, rho.shape)
    vis = joint_predict_vis(toa) * a_g + c_g                                    # debiased
    grid = np.round(np.arange(0.02, 1.001, 0.02), 3)
    R = solve(toa, vis, grid, [0])                                             # (ncand,N), 443-only
    px = grid[np.argmin(R, 0)]
    # coarse-cell pooling: one AOD per CxC cell
    C = 60
    Rc = R.reshape(len(grid), H, W)
    cell = np.empty((H, W))
    for i in range(0, H, C):
        for j in range(0, W, C):
            sub = Rc[:, i:i+C, j:j+C].reshape(len(grid), -1).mean(1)
            cell[i:i+C, j:j+C] = grid[np.argmin(sub)]
    cell = cell.ravel()
    at = aod_true
    def stat(x): return np.sqrt(np.mean((x-at)**2)), np.mean(x-at)
    pr, pb = stat(px); cr, cb = stat(cell)
    print(f"\nB. SPATIAL POOLING (debiased 443-only, AOD gradient 0.15->0.45, {H}x{W} block)")
    print(f"  per-pixel AOD:        RMSE {pr:.3f}  bias {pb:+.3f}")
    print(f"  {C}x{C}-cell pooled:   RMSE {cr:.3f}  bias {cb:+.3f}")
    # does the pooled field recover the gradient? check left vs right thirds
    lt = cell[col < W/3].mean(); rt = cell[col > 2*W/3].mean()
    print(f"  recovered gradient: left-third {lt:.2f} (truth ~0.20)  right-third {rt:.2f} (truth ~0.40)")


if __name__ == "__main__":
    a_g, c_g = part_a_debias()
    part_b_spatial(a_g, c_g)
