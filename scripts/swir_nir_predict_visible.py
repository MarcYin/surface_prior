"""Can SWIR+NIR predict the visible, and is it robust to aerosol loading?

SWIR is ~aerosol-free and NIR only weakly affected, so they should anchor the
surface identity even under haze. We test: take the clean monthly composites as
truth, forward to TOA at a range of AOD, then predict the visible (coastal,
blue, green, red) from the *haze-contaminated* SWIR+NIR via the spectral-library
joint kNN. Two errors per AOD level:

  predict-vs-CLEAN  : does the anchor still recover the clean surface visible?
                      (robustness — should grow only slowly with AOD)
  observed-vs-CLEAN : how contaminated the raw TOA visible is (grows fast)

If the first stays far below the second, SWIR+NIR is a usable aerosol-robust
surface predictor (the dark-target / SIAC anchor).

  /home/users/marcyin/.pixi/envs/base/bin/python scripts/swir_nir_predict_visible.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

CACHE = "/home/users/marcyin/.cache/spectral-library/prepared-runtime/v0.6.3"
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])  # 7-band centres
VIS = [0, 1, 2, 3]                  # coastal, blue, green, red (predicted)
VISNAMES = ["coastal", "blue", "green", "red"]
SZA, VZA, RAA = 28.0, 5.0, 120.0
RNG = np.random.default_rng(0)

# library: anchor = [nir, swir16, swir22], predict = [coastal, blue, green, red]
_Lv = np.load(f"{CACHE}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)   # ub,blue,green,red,nir
_Ls = np.load(f"{CACHE}/source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float64)  # swir1,2
_g = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_Lv, _Ls = _Lv[_g], _Ls[_g]
_ANCHOR = np.column_stack([_Lv[:, 4], _Ls[:, 0], _Ls[:, 1]])   # nir, swir16, swir22
_VISLIB = _Lv[:, :4]                                            # coastal,blue,green,red
_TREE = cKDTree(_ANCHOR)


def predict_visible(rows7, k=10):
    """rows7:(n,7) reflectance -> predicted (n,4) visible from nir+swir anchor."""
    q = np.column_stack([rows7[:, 4], rows7[:, 5], rows7[:, 6]])   # nir,swir16,swir22
    d, idx = _TREE.query(q, k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (_VISLIB[idx] * w[:, :, None]).sum(1)


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


def forward(rho, aot):
    p, T, S = rt(WL, aot); return p + T*rho/(1-S*rho)


def load_pixels(n=60000):
    # sample across seasons for surface variety
    months = ["2022-01", "2022-04", "2022-07", "2022-10"]
    Ys = []
    for mm in months:
        with rasterio.open(Path("egypt_monthly_5y_hls") / f"egypt_{mm}_hls.tif") as ds:
            a = np.stack([ds.read(b) for b in range(1, 8)]).astype(np.float32)
        v = np.all((a > 0) & (a != 65535), axis=0)
        Y = (a[:, v].T * 1e-4).astype(np.float64)
        Ys.append(Y[RNG.choice(len(Y), n // len(months), replace=False)])
    return np.clip(np.vstack(Ys), 1e-4, 0.95)


def main():
    rho = load_pixels()
    print(f"SWIR+NIR -> visible prediction across AOD (n={len(rho)} clean pixels, 4 seasons)")
    print("  per-band RMSE (DN): P=predict-vs-clean (robustness), O=observed-vs-clean (contamination)")
    print(f"  {'AOD':>5} " + " ".join(f"{n[:4]:>13}" for n in VISNAMES))
    curves = {n: {"P": [], "O": []} for n in VISNAMES}
    aods = [0.0, 0.1, 0.2, 0.4, 0.8]
    for aT in aods:
        toa = forward(rho, aT) + RNG.normal(0, 0.003, rho.shape)
        pred = predict_visible(toa)                       # from contaminated nir+swir
        cells = []
        for j, b in enumerate(VIS):
            P = np.sqrt(np.mean((pred[:, j] - rho[:, b])**2)) * 1e4    # vs CLEAN truth
            O = np.sqrt(np.mean((toa[:, b] - rho[:, b])**2)) * 1e4     # raw contamination
            curves[VISNAMES[j]]["P"].append(P); curves[VISNAMES[j]]["O"].append(O)
            cells.append(f"P{P:4.0f}/O{O:4.0f}")
        print(f"  {aT:5.2f} " + " ".join(f"{c:>13}" for c in cells))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for k, n in enumerate(VISNAMES):
            a = ax[k // 2, k % 2]
            a.plot(aods, curves[n]["O"], "r-o", label="observed TOA vs clean (contamination)")
            a.plot(aods, curves[n]["P"], "b-s", label="SWIR+NIR predicted vs clean (robustness)")
            a.set_title(f"{n}"); a.set_xlabel("AOD550"); a.set_ylabel("RMSE vs clean surface (DN)")
            a.grid(alpha=0.3); a.legend(fontsize=7)
        fig.suptitle("SWIR+NIR predicts the clean visible under haze\n"
                     "blue (predicted) stays low while red (raw TOA) climbs with AOD")
        fig.tight_layout()
        out = "/home/users/marcyin/surface_prior/swir_nir_predict_visible.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
