"""AOD retrieval with a SCENE-LOCAL surface model (the real lever).

The global spectral library predicts the clean visible only to ~114 DN, which
caps AOD accuracy. Training the surface model on the scene's own clean composite
(a different date, same AOI) drops that to ~41 DN -> below the aerosol signal.
Re-run the closed-loop AOD solve with that surface model and see if AOD becomes
accurately retrievable.

train surface model = 2021-07 clean composite (anchor red+nir+swir -> coastal,blue,green)
test = 2022-07 clean surface forwarded to TOA at known AOD; solve coastal+blue, pooled.

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/aod_scene_local.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

DIR = Path("egypt_monthly_5y_hls")
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])
SOLVE = [0, 1]                      # coastal, blue
SZA, VZA, RAA = 28.0, 5.0, 120.0
RNG = np.random.default_rng(0)


def rt(wl, aot, alpha=1.2, g=0.65, w0=0.95):
    lam = np.asarray(wl) / 1000.0
    tr = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4)
    ta = np.asarray(aot) * (lam / 0.55) ** (-alpha)
    mus, muv = np.cos(np.radians(SZA)), np.cos(np.radians(VZA))
    cosT = -mus*muv + np.sin(np.radians(SZA))*np.sin(np.radians(VZA))*np.cos(np.radians(RAA))
    Pr = 0.75*(1+cosT**2); Pa = (1-g**2)/(1+g**2-2*g*cosT)**1.5
    return (tr*Pr + w0*ta*Pa)/(4*mus*muv), np.exp(-(0.5*tr+ta*(1-w0*g))*(1/mus+1/muv)), 0.92*tr+0.33*ta


def forward(rho, aot, wl):
    p, T, S = rt(wl, aot); return p + T*rho/(1-S*rho)


def load_full(month):
    with rasterio.open(DIR / f"egypt_{month}_hls.tif") as ds:
        a = ds.read(list(range(1, 8))).astype(np.float32)
    refl = a.reshape(7, -1).T.astype(np.float64) * 1e-4
    v = np.all((refl > 0) & (refl < 1.2), axis=1)
    return refl, v


def load_block(month, r0, c0, H, W):
    with rasterio.open(DIR / f"egypt_{month}_hls.tif") as ds:
        a = ds.read(list(range(1, 8)), window=Window(c0, r0, W, H)).astype(np.float32)
    return np.clip(a.reshape(7, -1).T * 1e-4, 1e-4, 0.95).astype(np.float64), (H, W)


def main():
    # train scene-local surface model on 2021-07
    ref, refv = load_full("2021-07")
    ti = RNG.choice(np.where(refv)[0], 200000, replace=False)
    Ra = ref[ti][:, [3, 4, 5, 6]]                       # red,nir,swir16,swir22
    am, asd = Ra.mean(0), Ra.std(0)
    tree = cKDTree((Ra - am) / asd)
    vis_lib = ref[ti][:, :3]                            # coastal,blue,green

    def predict(rows7, k=10):
        d, idx = tree.query((rows7[:, [3, 4, 5, 6]] - am) / asd, k=k)
        w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
        return (vis_lib[idx] * w[:, :, None]).sum(1)

    rho, (H, W) = load_block("2022-07", 800, 800, 360, 360)
    grid = np.round(np.arange(0.02, 1.001, 0.02), 3)
    C = 60

    def pooled(toa, vis):
        Rr = np.empty((len(grid), len(toa)))
        for i, ac in enumerate(grid):
            Rr[i] = ((forward(vis[:, SOLVE], ac, WL[SOLVE]) - toa[:, SOLVE])**2).sum(1)
        px = grid[np.argmin(Rr, 0)]
        Rc = Rr.reshape(len(grid), H, W); cell = np.empty((H, W))
        for i in range(0, H, C):
            for j in range(0, W, C):
                cell[i:i+C, j:j+C] = grid[np.argmin(Rc[:, i:i+C, j:j+C].reshape(len(grid), -1).mean(1))]
        return px, cell.ravel()

    print("AOD with SCENE-LOCAL surface model (train 2021-07 -> test 2022-07)")
    print(f"  {'AOT*':>5} {'pixel med':>10} {'pixel RMSE':>11} {'pooled med':>11} {'pooled RMSE':>12}")
    for aT in (0.10, 0.30, 0.50, 0.80):
        toa = forward(rho, aT, WL) + RNG.normal(0, 0.003, rho.shape)
        vis = predict(toa)
        px, cell = pooled(toa, vis)
        print(f"  {aT:5.2f} {np.median(px):10.2f} {np.sqrt(np.mean((px-aT)**2)):11.3f} "
              f"{np.median(cell):11.2f} {np.sqrt(np.mean((cell-aT)**2)):12.3f}")

    # spatial gradient
    col = np.tile(np.arange(W), H); at = 0.10 + 0.50 * col / (W - 1)
    toa = forward(rho, at[:, None], WL) + RNG.normal(0, 0.003, rho.shape)
    px, cell = pooled(toa, predict(toa))
    cr = np.sqrt(np.mean((cell-at)**2)); cb = np.mean(cell-at); cc = np.corrcoef(cell, at)[0, 1]
    print(f"\n  gradient 0.10->0.60, {C}-cell: RMSE {cr:.3f}  bias {cb:+.3f}  corr {cc:.2f}")
    print(f"    left {cell[col<W/4].mean():.2f} (truth ~0.16)  right {cell[col>3*W/4].mean():.2f} (truth ~0.54)")


if __name__ == "__main__":
    main()
