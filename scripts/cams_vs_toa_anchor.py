"""Is a rough CAMS pre-correction of the anchor better than raw TOA?

The dictionary is trained on SURFACE reflectance, so feeding it a raw TOA anchor
is a domain mismatch (haze-elevated red -> biased prediction -> AOD low). Full
anchor iteration fixes it but costs a kNN prediction per candidate AOD. A cheap
alternative: correct the anchor ONCE with the scene's CAMS AOD, then solve.

Compares anchor states for (1) clean-visible prediction error and (2) AOD
retrieval:
  TOA         : raw, no correction
  CAMS(d)     : corrected once with CAMS = true_AOD + d  (d = CAMS error)
  oracle      : corrected at the true AOD (= full iteration's fixed point)

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/cams_vs_toa_anchor.py
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
        rows[:, ANCHOR] = correct(toa[:, ANCHOR], aod_anchor, WL[ANCHOR])
    return dic.predict(rows)


def solve_fwdmatch(toa, vis):
    """AOD per pixel by forward-matching a fixed predicted surface to TOA."""
    R = np.empty((len(GRID), len(toa)))
    for i, ac in enumerate(GRID):
        R[i] = ((forward(vis[:, SOLVE], ac, WL[SOLVE]) - toa[:, SOLVE]) ** 2).sum(1)
    return GRID[np.argmin(R, 0)]


def solve_iter(toa, dic):
    R = np.empty((len(GRID), len(toa)))
    for i, ac in enumerate(GRID):
        rows = toa.copy(); rows[:, ANCHOR] = correct(toa[:, ANCHOR], ac, WL[ANCHOR])
        vp = dic.predict(rows); vo = correct(toa[:, SOLVE], ac, WL[SOLVE])
        R[i] = ((vp[:, SOLVE] - vo) ** 2).sum(1)
    return GRID[np.argmin(R, 0)]


def main():
    paths = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13) if not (y == TY and m == 7)]
    dic = SurfaceDictionary().fit([load_full(p) for p in paths])
    rho = load_block(f"{TY}-07", 800, 800, 200, 200)
    n = len(rho)
    print(f"dictionary on {len(paths)} composites; test block {n} px (target {TY}-07)\n")

    # ---- (1) clean-visible (blue) prediction error vs anchor state ----
    print("(1) blue prediction RMSE vs clean truth (DN), by anchor state:")
    print(f"  {'AOT*':>5} {'TOA':>6} {'CAMS d=0':>9} {'CAMS+0.1':>9} {'CAMS-0.1':>9} {'CAMS+0.2':>9} {'oracle':>7}")
    for aT in (0.10, 0.30, 0.50, 0.80):
        toa = forward(rho, aT, WL) + RNG.normal(0, 0.003, rho.shape)
        row = [f"{aT:5.2f}"]
        for _lab, aa in [("TOA", None), ("c0", aT), ("c+", aT+0.1), ("c-", max(aT-0.1, 0.01)),
                        ("c2", aT+0.2), ("or", aT)]:
            p = predict(toa, dic, aa)
            row.append(f"{np.sqrt(np.mean((p[:,1]-rho[:,1])**2))*1e4:6.0f}")
        # reorder to header (TOA, c0, c+, c-, c2, oracle==c0)
        print(f"  {row[0]} {row[1]:>6} {row[2]:>9} {row[3]:>9} {row[4]:>9} {row[5]:>9} {row[2]:>7}")

    # ---- (2) AOD retrieval (per-pixel median) by method ----
    print("\n(2) retrieved AOD (per-pixel median), CAMS = true + error d:")
    print(f"  {'AOT*':>5} {'non-iter(TOA)':>14} {'CAMS d=0':>9} {'CAMS+0.1':>9} {'CAMS-0.1':>9} {'full-iter':>10}")
    for aT in (0.10, 0.30, 0.50, 0.80):
        toa = forward(rho, aT, WL) + RNG.normal(0, 0.003, rho.shape)
        m_toa = np.median(solve_fwdmatch(toa, predict(toa, dic, None)))
        m_c0 = np.median(solve_fwdmatch(toa, predict(toa, dic, aT)))
        m_cp = np.median(solve_fwdmatch(toa, predict(toa, dic, aT+0.1)))
        m_cm = np.median(solve_fwdmatch(toa, predict(toa, dic, max(aT-0.1, 0.01))))
        m_it = np.median(solve_iter(toa, dic))
        print(f"  {aT:5.2f} {m_toa:14.2f} {m_c0:9.2f} {m_cp:9.2f} {m_cm:9.2f} {m_it:10.2f}")


if __name__ == "__main__":
    main()
