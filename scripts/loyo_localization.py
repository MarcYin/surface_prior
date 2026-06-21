"""Find the best GENERIC localization for the SWIR+NIR -> visible surface model.

Leave-one-year-out: predict each target year's July clean visible from training
pools that EXCLUDE that target, comparing how to build the local "library":

  global         : 77k generic spectra (no localization)
  LOYO-Jul       : the 4 OTHER years' July only (seasonal match, few surfaces)
  LOYO-allmonth  : the 4 OTHER years, all 12 months (rich surface-state dict)
  same-year-rest : the target year's OWN other 11 months (no July) -- most local
  all-but-target : every composite except the target July (59) -- richest

Anchor = red+nir+swir; clean->clean so this isolates the surface-prediction
floor (localization quality), not aerosol. Lower blue/coastal RMSE = better.

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/loyo_localization.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

CACHE = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3"
DIR = Path("egypt_monthly_5y_hls")
YEARS = [2020, 2021, 2022, 2023, 2024]
PER = 20000              # sampled pixels per composite
RNG = np.random.default_rng(0)

# global library anchor red+nir+swir -> visible
_Lv = np.load(f"{CACHE}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)
_Ls = np.load(f"{CACHE}/source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float64)
_g = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_LIB_A = np.column_stack([_Lv[_g, 3], _Lv[_g, 4], _Ls[_g, 0], _Ls[_g, 1]])
_LIB_V = _Lv[_g, :3]


def load_sample(year, month):
    with rasterio.open(DIR / f"egypt_{year}-{month:02d}_hls.tif") as ds:
        a = ds.read(list(range(1, 8))).astype(np.float32)
    refl = a.reshape(7, -1).T.astype(np.float64) * 1e-4
    v = np.where(np.all((refl > 0) & (refl < 1.2), axis=1))[0]
    idx = RNG.choice(v, min(PER, len(v)), replace=False)
    return refl[idx]


def knn(train_anchor, train_vis, q_anchor, k=10, standardize=True):
    if standardize:
        m, s = train_anchor.mean(0), train_anchor.std(0)
        tA, qA = (train_anchor - m) / s, (q_anchor - m) / s
    else:
        tA, qA = train_anchor, q_anchor
    tree = cKDTree(tA)
    d, idx = tree.query(qA, k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (train_vis[idx] * w[:, :, None]).sum(1)


def main():
    # pre-sample every composite once: {(year,month): (N,7)}
    print("sampling 60 composites ...")
    S = {(y, m): load_sample(y, m) for y in YEARS for m in range(1, 13)}
    A = lambda arr: arr[:, [3, 4, 5, 6]]     # anchor cols
    V = lambda arr: arr[:, :3]               # visible targets

    configs = ["global", "LOYO-Jul", "LOYO-allmonth", "same-year-rest", "all-but-target"]
    res = {c: {"coastal": [], "blue": []} for c in configs}

    for ty in YEARS:
        test = S[(ty, 7)]
        truth = V(test); qa = A(test)
        pools = {
            "LOYO-Jul":       np.vstack([S[(y, 7)] for y in YEARS if y != ty]),
            "LOYO-allmonth":  np.vstack([S[(y, m)] for y in YEARS if y != ty for m in range(1, 13)]),
            "same-year-rest": np.vstack([S[(ty, m)] for m in range(1, 13) if m != 7]),
            "all-but-target": np.vstack([S[(y, m)] for y in YEARS for m in range(1, 13)
                                         if not (y == ty and m == 7)]),
        }
        for c in configs:
            if c == "global":
                pred = knn(_LIB_A, _LIB_V, qa, standardize=False)
            else:
                tr = pools[c]
                if len(tr) > 300000:
                    tr = tr[RNG.choice(len(tr), 300000, replace=False)]
                pred = knn(A(tr), V(tr), qa)
            res[c]["coastal"].append(np.sqrt(np.mean((pred[:, 0] - truth[:, 0])**2)) * 1e4)
            res[c]["blue"].append(np.sqrt(np.mean((pred[:, 1] - truth[:, 1])**2)) * 1e4)

    print(f"\nLOYO visible-prediction RMSE (DN), mean over 5 target years (+per-year blue):")
    print(f"  {'config':16} {'coastal':>8} {'blue':>7}   blue per target year")
    for c in configs:
        co = np.mean(res[c]["coastal"]); bl = np.mean(res[c]["blue"])
        peryr = " ".join(f"{v:3.0f}" for v in res[c]["blue"])
        print(f"  {c:16} {co:8.0f} {bl:7.0f}   [{peryr}]  ({', '.join(map(str, YEARS))})")


if __name__ == "__main__":
    main()
