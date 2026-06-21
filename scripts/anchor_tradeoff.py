"""Anchor tradeoff: accuracy vs aerosol-robustness for visible prediction.

  nir+swir       : SWIR/NIR ~aerosol-free -> prediction error flat with AOD
                   (robust) but weaker (red excluded).
  red+nir+swir   : red is a strong visible predictor -> accurate at low AOD,
                   but red is aerosol-sensitive, so on TOA its query drifts and
                   the error GROWS with AOD.

Compared on the bands both predict (coastal, blue, green). Closed loop: clean
monthly composites forwarded to TOA at a range of AOD, predict from the
contaminated anchor, error vs clean truth. A crossover AOD shows where
robustness overtakes accuracy.

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/anchor_tradeoff.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

CACHE = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3"
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])
TGT = [0, 1, 2]                       # coastal, blue, green (predicted by both)
TGTN = ["coastal", "blue", "green"]
SZA, VZA, RAA = 28.0, 5.0, 120.0
RNG = np.random.default_rng(0)

_Lv = np.load(f"{CACHE}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)  # ub,blue,green,red,nir
_Ls = np.load(f"{CACHE}/source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float64)
_g = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_Lv, _Ls = _Lv[_g], _Ls[_g]
_TGTLIB = _Lv[:, :3]
# anchor libraries + their column indices within a 7-band pixel row
_ANCHORS = {
    "nir+swir":      (cKDTree(np.column_stack([_Lv[:, 4], _Ls[:, 0], _Ls[:, 1]])),       [4, 5, 6]),
    "red+nir+swir":  (cKDTree(np.column_stack([_Lv[:, 3], _Lv[:, 4], _Ls[:, 0], _Ls[:, 1]])), [3, 4, 5, 6]),
}


def predict(rows7, tree, cols, k=10):
    d, idx = tree.query(rows7[:, cols], k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (_TGTLIB[idx] * w[:, :, None]).sum(1)


def rt(wl, aot, alpha=1.2, g=0.65, w0=0.95):
    lam = np.asarray(wl) / 1000.0
    tr = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4)
    ta = aot * (lam / 0.55) ** (-alpha)
    mus, muv = np.cos(np.radians(SZA)), np.cos(np.radians(VZA))
    cosT = -mus*muv + np.sin(np.radians(SZA))*np.sin(np.radians(VZA))*np.cos(np.radians(RAA))
    Pr = 0.75*(1+cosT**2); Pa = (1-g**2)/(1+g**2-2*g*cosT)**1.5
    path = (tr*Pr + w0*ta*Pa)/(4*mus*muv)
    T = np.exp(-(0.5*tr + ta*(1-w0*g))*(1/mus+1/muv))
    return path + 0*tr, T, 0.92*tr + 0.33*ta


def forward(rho, aot):
    p, T, S = rt(WL, aot); return p + T*rho/(1-S*rho)


def load_pixels(n=60000):
    Ys = []
    for mm in ["2022-01", "2022-04", "2022-07", "2022-10"]:
        with rasterio.open(Path("egypt_monthly_5y_hls") / f"egypt_{mm}_hls.tif") as ds:
            a = np.stack([ds.read(b) for b in range(1, 8)]).astype(np.float32)
        v = np.all((a > 0) & (a != 65535), axis=0)
        Y = (a[:, v].T * 1e-4).astype(np.float64)
        Ys.append(Y[RNG.choice(len(Y), n // 4, replace=False)])
    return np.clip(np.vstack(Ys), 1e-4, 0.95)


def main():
    rho = load_pixels()
    aods = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
    curves = {a: {t: [] for t in TGTN} for a in _ANCHORS}
    print(f"Anchor tradeoff (n={len(rho)}): visible prediction RMSE (DN) vs AOD")
    for aT in aods:
        toa = forward(rho, aT) + RNG.normal(0, 0.003, rho.shape)
        line = [f"AOD {aT:.2f}"]
        for aname, (tree, cols) in _ANCHORS.items():
            pred = predict(toa, tree, cols)
            for j, b in enumerate(TGT):
                e = np.sqrt(np.mean((pred[:, j] - rho[:, b])**2)) * 1e4
                curves[aname][TGTN[j]].append(e)
            line.append(f"{aname}: blue={curves[aname]['blue'][-1]:.0f}")
        print("  " + "  ".join(line))
    # crossover AOD per band (where red+nir+swir error exceeds nir+swir)
    print("\ncrossover AOD (red+nir+swir loses its accuracy edge to robust nir+swir):")
    for t in TGTN:
        a = np.array(curves["nir+swir"][t]); b = np.array(curves["red+nir+swir"][t])
        diff = b - a
        xo = next((aods[i] for i in range(1, len(aods)) if diff[i] >= 0), None)
        print(f"  {t:8}: nir+swir~{a.mean():.0f} flat;  red+nir+swir {b[0]:.0f}@0 -> {b[-1]:.0f}@0.8;  "
              f"crossover {'AOD '+str(xo) if xo is not None else '>0.8 (red+swir always better)'}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        for k, t in enumerate(TGTN):
            ax[k].plot(aods, curves["red+nir+swir"][t], "g-o", label="red+nir+swir (accurate, AOD-sensitive)")
            ax[k].plot(aods, curves["nir+swir"][t], "b-s", label="nir+swir (robust, weaker)")
            ax[k].set_title(t); ax[k].set_xlabel("AOD550"); ax[k].set_ylabel("predict RMSE vs clean (DN)")
            ax[k].grid(alpha=0.3); ax[k].legend(fontsize=8)
        fig.suptitle("Anchor tradeoff (result): red+nir+swir wins at ALL tested AOD\n"
                     "red's mild haze-contamination is buffered by nir+swir -> ~2x more accurate, no crossover <=0.8")
        fig.tight_layout()
        out = "/home/users/marcyin/surface_prior/anchor_tradeoff.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
