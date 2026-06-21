"""Does the AOD-prior SOURCE/resolution matter for anchor pre-correction?

The anchor pre-correction accepts any rough AOD prior (CAMS, MERRA-2, MOD04,
MCD19A2/MAIAC...). Given the ~+/-0.1 tolerance, the practical question is
resolution: a coarse model prior (CAMS/MERRA, ~40km, scene-constant) vs a 1km
retrieved prior (MAIAC, per-pixel) on a spatially-VARYING AOD scene.

Test: true AOD = spatial gradient 0.10->0.60. Pre-correct the anchor with:
  TOA           - none (baseline)
  coarse        - scene-mean constant (CAMS/MERRA-like, resolves nothing)
  coarse+bias   - scene-mean + 0.1 (biased model prior)
  highres       - true + N(0,0.03) per pixel (MAIAC-like)
  oracle        - true per pixel
Report blue surface-prediction error and AOD retrieval (forward-match solve).

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/aod_prior_source.py
"""
from __future__ import annotations

import numpy as np
from surface_dictionary import (
    ANCHOR,
    SOLVE,
    WL,
    SurfaceDictionary,
    correct,
    forward,
    load_block,
    load_full,
)

RNG = np.random.default_rng(0)
YEARS = [2020, 2021, 2022, 2023, 2024]
TY = 2022
GRID = np.round(np.arange(0.02, 1.001, 0.04), 3)


def predict(toa, dic, aod_anchor):
    rows = toa.copy()
    if aod_anchor is not None:
        # aod_anchor may be scalar or per-pixel (n,) -> needs (n,1) for the RT
        aa = aod_anchor if np.isscalar(aod_anchor) else np.asarray(aod_anchor)[:, None]
        rows[:, ANCHOR] = correct(toa[:, ANCHOR], aa, WL[ANCHOR])
    return dic.predict(rows)


def solve_fwdmatch(toa, vis, HW, C=60):
    H, W = HW
    R = np.empty((len(GRID), len(toa)))
    for i, ac in enumerate(GRID):
        R[i] = ((forward(vis[:, SOLVE], ac, WL[SOLVE]) - toa[:, SOLVE]) ** 2).sum(1)
    px = GRID[np.argmin(R, 0)]
    Rc = R.reshape(len(GRID), H, W); cell = np.empty((H, W))
    for a in range(0, H, C):
        for b in range(0, W, C):
            cell[a:a+C, b:b+C] = GRID[np.argmin(Rc[:, a:a+C, b:b+C].reshape(len(GRID), -1).mean(1))]
    return px, cell.ravel()


def main():
    paths = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13) if not (y == TY and m == 7)]
    dic = SurfaceDictionary().fit([load_full(p) for p in paths])
    H = W = 240
    rho = load_block(f"{TY}-07", 800, 800, H, W)
    col = np.tile(np.arange(W), H)
    at = 0.10 + 0.50 * col / (W - 1)                  # true AOD gradient (per pixel)
    toa = forward(rho, at[:, None], WL) + RNG.normal(0, 0.003, rho.shape)
    print(f"Gradient AOD 0.10->0.60, block {H}x{W} (target {TY}-07)\n")

    priors = {
        "TOA (none)":     None,
        "coarse mean":    float(at.mean()),
        "coarse+0.1":     float(at.mean()) + 0.1,
        "highres (MAIAC)": at + RNG.normal(0, 0.03, len(at)),
        "oracle":         at,
    }
    print(f"  {'prior':16} {'blue_predRMSE':>13} {'AOD pooled RMSE':>16} {'bias':>7} {'corr':>6}")
    for name, p in priors.items():
        vis = predict(toa, dic, p)
        blue = np.sqrt(np.mean((vis[:, 1] - rho[:, 1])**2)) * 1e4
        _, cell = solve_fwdmatch(toa, vis, (H, W))
        rmse = np.sqrt(np.mean((cell - at)**2)); bias = np.mean(cell - at); cc = np.corrcoef(cell, at)[0, 1]
        print(f"  {name:16} {blue:13.0f} {rmse:16.3f} {bias:+7.3f} {cc:6.2f}")


if __name__ == "__main__":
    main()
