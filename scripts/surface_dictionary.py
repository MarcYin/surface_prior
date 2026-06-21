"""Reusable scene-local surface predictor + AOD solver with anchor iteration.

SurfaceDictionary: a kNN surface model built once from an AOI's clean composites
(any years/months -- LOYO showed they tie), mapping the aerosol-robust anchor
(red+nir+swir) -> clean visible (coastal,blue,green). Generic across years for a
stable landscape.

AOD solve: the systematic low bias came from predicting off the haze-elevated
red. Fix = ANCHOR ITERATION: at each candidate AOD, atmospherically-correct the
anchor first, predict the clean visible from the *corrected* anchor, and match
it to the *corrected* observed visible. Plus an optional bias calibration (the
AERONET-anchored residual map).

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/surface_dictionary.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

DIR = Path("egypt_monthly_5y_hls")
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])
ANCHOR = [3, 4, 5, 6]    # red, nir, swir16, swir22
SOLVE = [0, 1]           # coastal, blue (within the visible target)
SZA, VZA, RAA = 28.0, 5.0, 120.0
RNG = np.random.default_rng(0)


class SurfaceDictionary:
    """kNN red+nir+swir -> coastal,blue,green, trained on AOI clean composites."""

    def __init__(self, cap=300000):
        self.cap = cap

    def fit(self, refl_arrays):
        pool = np.vstack(refl_arrays)
        if len(pool) > self.cap:
            pool = pool[RNG.choice(len(pool), self.cap, replace=False)]
        A = pool[:, ANCHOR]
        self.mean_, self.std_ = A.mean(0), A.std(0)
        self.tree_ = cKDTree((A - self.mean_) / self.std_)
        self.vis_ = pool[:, :3]
        return self

    def predict(self, rows7, k=10):
        q = (rows7[:, ANCHOR] - self.mean_) / self.std_
        d, idx = self.tree_.query(q, k=k)
        w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
        return (self.vis_[idx] * w[:, :, None]).sum(1)


def rt(wl, aot, alpha=1.2, g=0.65, w0=0.95):
    lam = np.asarray(wl) / 1000.0
    tr = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4)
    ta = np.asarray(aot) * (lam / 0.55) ** (-alpha)
    mus, muv = np.cos(np.radians(SZA)), np.cos(np.radians(VZA))
    cosT = -mus*muv + np.sin(np.radians(SZA))*np.sin(np.radians(VZA))*np.cos(np.radians(RAA))
    Pr = 0.75*(1+cosT**2); Pa = (1-g**2)/(1+g**2-2*g*cosT)**1.5
    return (tr*Pr+w0*ta*Pa)/(4*mus*muv), np.exp(-(0.5*tr+ta*(1-w0*g))*(1/mus+1/muv)), 0.92*tr+0.33*ta


def forward(rho, aot, wl):
    p, T, S = rt(wl, aot); return p + T*rho/(1-S*rho)


def correct(toa, aot, wl):
    p, T, S = rt(wl, aot); u = toa - p; return u / (T + S*u)


def predict_visible(toa, dic, aod_prior=None):
    """Predict the clean visible from the anchor, optionally pre-correcting it
    with a rough AOD prior to remove the TOA->surface domain shift.

    aod_prior : None (raw TOA anchor; biased), a scalar (per-scene prior, e.g.
                CAMS / MERRA-2 scene value), or a per-pixel array of length
                len(toa) (gridded prior, e.g. MCD19A2/MAIAC or MOD04). The
                solve is robust to ~+/-0.1 error in this prior.
    """
    rows = toa.copy()
    if aod_prior is not None:
        aa = aod_prior if np.isscalar(aod_prior) else np.asarray(aod_prior)[:, None]
        rows[:, ANCHOR] = correct(toa[:, ANCHOR], aa, WL[ANCHOR])
    return dic.predict(rows)


def _pool(R, grid, HW, C):
    H, W = HW
    px = grid[np.argmin(R, 0)]
    Rc = R.reshape(len(grid), H, W); cell = np.empty((H, W))
    for a in range(0, H, C):
        for b in range(0, W, C):
            cell[a:a+C, b:b+C] = grid[np.argmin(Rc[:, a:a+C, b:b+C].reshape(len(grid), -1).mean(1))]
    return px, cell.ravel()


def solve(toa, dic, grid, HW, C=60, mode="prior", aod_prior=None):
    """Closed-loop AOD solve over `grid`, per-pixel and `C`-cell pooled.

    mode:
      'prior'    (recommended) pre-correct the anchor ONCE with `aod_prior`
                 (any source: CAMS, MERRA-2, MOD04, MCD19A2/MAIAC; scalar or
                 grid), predict the surface, then forward-match. Cheap (one kNN
                 prediction) and unbiased for a prior within ~+/-0.1. Falls back
                 to 'iterate' if aod_prior is None.
      'iterate'  correct the anchor at EVERY candidate AOD (no prior needed;
                 ~Ngrid x more kNN queries). Unbiased.
      'raw'      predict from the uncorrected TOA anchor (biased low; baseline).
    """
    H, W = HW
    if mode == "prior" and aod_prior is None:
        mode = "iterate"
    R = np.empty((len(grid), len(toa)))
    if mode in ("prior", "raw"):
        vis = predict_visible(toa, dic, aod_prior if mode == "prior" else None)
    for i, ac in enumerate(grid):
        if mode == "iterate":
            rows = toa.copy()
            rows[:, ANCHOR] = correct(toa[:, ANCHOR], ac, WL[ANCHOR])
            vp = dic.predict(rows); vo = correct(toa[:, SOLVE], ac, WL[SOLVE])
            R[i] = ((vp[:, SOLVE] - vo) ** 2).sum(1)
        else:                                   # 'prior' / 'raw': fixed surface, forward-match
            R[i] = ((forward(vis[:, SOLVE], ac, WL[SOLVE]) - toa[:, SOLVE]) ** 2).sum(1)
    return _pool(R, grid, HW, C)


def load_full(month_year):
    with rasterio.open(DIR / f"egypt_{month_year}_hls.tif") as ds:
        a = ds.read(list(range(1, 8))).astype(np.float32)
    refl = a.reshape(7, -1).T.astype(np.float64) * 1e-4
    v = np.where(np.all((refl > 0) & (refl < 1.2), axis=1))[0]
    return refl[RNG.choice(v, min(40000, len(v)), replace=False)]


def load_block(month_year, r0, c0, H, W):
    with rasterio.open(DIR / f"egypt_{month_year}_hls.tif") as ds:
        a = ds.read(list(range(1, 8)), window=Window(c0, r0, W, H)).astype(np.float32)
    return np.clip(a.reshape(7, -1).T * 1e-4, 1e-4, 0.95).astype(np.float64)


def main():
    YEARS = [2020, 2021, 2022, 2023, 2024]
    TY = 2022                                   # target year (held out)
    # generic dictionary: all clean composites EXCEPT the target July (LOYO)
    paths = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13) if not (y == TY and m == 7)]
    dic = SurfaceDictionary().fit([load_full(p) for p in paths])
    print(f"SurfaceDictionary fit on {len(paths)} composites (LOYO, target {TY}-07)")

    H = W = 240
    rho = load_block(f"{TY}-07", 800, 800, H, W)
    grid = np.round(np.arange(0.02, 1.001, 0.04), 3)
    levels = [0.10, 0.30, 0.50, 0.80]

    print("\nConstant AOD, 60-cell pooled median (raw / prior=CAMS-like / iterate):")
    print(f"  {'AOT*':>5} {'raw(TOA)':>9} {'prior':>7} {'iterate':>8}")
    for aT in levels:
        toa = forward(rho, aT, WL) + RNG.normal(0, 0.003, rho.shape)
        _, c_raw = solve(toa, dic, grid, (H, W), mode="raw")
        _, c_pri = solve(toa, dic, grid, (H, W), mode="prior", aod_prior=aT + 0.05)  # CAMS-like (+err)
        _, c_itr = solve(toa, dic, grid, (H, W), mode="iterate")
        print(f"  {aT:5.2f} {np.median(c_raw):9.2f} {np.median(c_pri):7.2f} {np.median(c_itr):8.2f}")

    # gradient test: prior path (scene-mean prior, as a coarse model product would give)
    col = np.tile(np.arange(W), H); at = 0.10 + 0.50 * col / (W - 1)
    toa = forward(rho, at[:, None], WL) + RNG.normal(0, 0.003, rho.shape)
    _, cell = solve(toa, dic, grid, (H, W), mode="prior", aod_prior=float(at.mean()))
    def stat(x): return np.sqrt(np.mean((x-at)**2)), np.mean(x-at), np.corrcoef(x, at)[0, 1]
    r0, b0, cc0 = stat(cell)
    print("\nGradient 0.10->0.60 (60-cell), prior path (scene-mean AOD):")
    print(f"  RMSE {r0:.3f}  bias {b0:+.3f}  corr {cc0:.2f}")
    print(f"  recovered: left {cell[col<W/4].mean():.2f} (~0.16)  right {cell[col>3*W/4].mean():.2f} (~0.54)")


if __name__ == "__main__":
    main()
