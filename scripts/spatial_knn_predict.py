"""Does adding (x,y) to the kNN shrink the visible-prediction floor?

Three predictors of a clean month's visible (coastal/blue/green) from its
aerosol-free anchor (red+nir+swir):
  A. GLOBAL library      : 77k generic spectra, anchor only (what we had)
  B. SCENE-local         : trained on a *different-date* clean composite of the
                           same AOI, anchor only (scene-specific surfaces)
  C. SCENE-local + (x,y) : same, but the kNN also matches geographic location
                           -> reflectance "from the area"

Train on a reference clean composite, predict a held-out date's clean visible
(no self-lookup). Lower error -> tighter surface prior -> better AOD ceiling.

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/spatial_knn_predict.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

CACHE = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3"
DIR = Path("egypt_monthly_5y_hls")
REF_MONTH = "2021-07"     # surface-model reference (clean)
TEST_MONTH = "2022-07"    # held-out clean truth (same season, 1 yr later)
VISN = ["coastal", "blue", "green"]
RNG = np.random.default_rng(0)

# global library anchor red+nir+swir -> visible
_Lv = np.load(f"{CACHE}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)
_Ls = np.load(f"{CACHE}/source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float64)
_g = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_Lv, _Ls = _Lv[_g], _Ls[_g]
_LIB_ANCHOR = np.column_stack([_Lv[:, 3], _Lv[:, 4], _Ls[:, 0], _Ls[:, 1]])
_LIB_VIS = _Lv[:, :3]
_LIB_TREE = cKDTree(_LIB_ANCHOR)


def load(month):
    with rasterio.open(DIR / f"egypt_{month}_hls.tif") as ds:
        a = ds.read(list(range(1, 8))).astype(np.float32)        # (7,H,W)
        tr = ds.transform
        H, W = ds.height, ds.width
    refl = a.reshape(7, -1).T.astype(np.float64) * 1e-4           # (N,7)
    rows, cols = np.divmod(np.arange(H * W), W)
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    valid = np.all((refl > 0) & (refl < 1.2), axis=1)
    return refl, np.column_stack([xs, ys]), valid


def knn_pred(tree, vis_lib, query, k=10):
    d, idx = tree.query(query, k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (vis_lib[idx] * w[:, :, None]).sum(1)


def main():
    ref, refxy, refv = load(REF_MONTH)
    tst, tstxy, tstv = load(TEST_MONTH)
    ref_i = np.where(refv)[0]; ref_i = RNG.choice(ref_i, 200000, replace=False)
    tst_i = np.where(tstv)[0]; tst_i = RNG.choice(tst_i, 50000, replace=False)
    R, Rxy = ref[ref_i], refxy[ref_i]
    T, Txy, Ttruth = tst[tst_i], tstxy[tst_i], tst[tst_i, :3]

    # anchors (red,nir,swir16,swir22)
    Ra = R[:, [3, 4, 5, 6]]; Ta = T[:, [3, 4, 5, 6]]
    # standardize anchor (for B/C) using ref stats; xy standardized too
    am, asd = Ra.mean(0), Ra.std(0)
    xym, xysd = Rxy.mean(0), Rxy.std(0)
    Ra_s, Ta_s = (Ra - am) / asd, (Ta - am) / asd
    Rxy_s, Txy_s = (Rxy - xym) / xysd, (Txy - xym) / xysd

    preds = {}
    preds["A global library"] = knn_pred(_LIB_TREE, _LIB_VIS, Ta)             # raw refl anchor
    treeB = cKDTree(Ra_s)
    preds["B scene-local"] = knn_pred(treeB, R[:, :3], Ta_s)
    treeC = cKDTree(np.column_stack([Rxy_s, Ra_s]))
    preds["C scene-local + xy"] = knn_pred(treeC, R[:, :3],
                                           np.column_stack([Txy_s, Ta_s]))

    print(f"Predicting {TEST_MONTH} clean visible from {REF_MONTH} surface model")
    print("(red+nir+swir anchor; n_train=200k n_test=50k) — RMSE vs truth (DN)")
    print(f"  {'method':22} {'coastal':>8} {'blue':>7} {'green':>7}")
    for name, P in preds.items():
        e = [np.sqrt(np.mean((P[:, j] - Ttruth[:, j])**2)) * 1e4 for j in range(3)]
        print(f"  {name:22} {e[0]:8.0f} {e[1]:7.0f} {e[2]:7.0f}")


if __name__ == "__main__":
    main()
