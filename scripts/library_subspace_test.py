"""How well does the spectral library span the image's VNIR spectral variation?

Step 1 of the SIAC-style plan: the library defines a low-dimensional surface
reflectance manifold. We measure (a) its intrinsic dimensionality, and (b) how
much of the image's per-pixel spectral variation lives inside that manifold,
for the HLS and S2 priors. Whatever the manifold cannot explain is the residual
that, on TOA data, would carry the aerosol signal.

Tests
  A. Library intrinsic dim at full VNIR hyperspectral (601 bands, 1 nm).
  B. Library PCA at the 5 S2A VNIR bands -> eigen-spectra + explained variance.
  C. Project every image pixel onto the library's k-dim variance subspace
     (image-mean-centered, library directions): explained variance vs k and
     per-band residual. Also the mean offset (image - library) per band.
  D. The leftover (5th) direction's spectral shape vs a Rayleigh/aerosol-like
     path-signal shape -> how much leverage VNIR leaves to solve aerosol.

Uses the prepared-runtime cache of the `spectral-library` package (v0.6.3,
sentinel-2a_msi). Library cols: vnir [ultra_blue, blue, green, red, nir].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

CACHE = Path.home() / ".cache/spectral-library/prepared-runtime/v0.6.3"
VNIR_BANDS = [1, 2, 3, 4, 5]   # coastal blue green red nir
NAMES = ["coastal", "blue", "green", "red", "nir"]
REFL = 1e-4
# Canonical hyperspectral VNIR grid is 400..1000 nm @ 1 nm (601 pts).
HYPER_WL = np.arange(400, 1001)


def clean_lib(L):
    g = np.all(np.isfinite(L), 1) & (np.nanmin(L, 1) >= -0.05) & (np.nanmax(L, 1) <= 1.2)
    return L[g]


def pca(X):
    """Mean-centered PCA. Returns mean, eigvecs (cols, desc), eigvals (desc)."""
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    w, V = np.linalg.eigh(cov)              # ascending
    order = np.argsort(w)[::-1]
    return mu, V[:, order], w[order]


def load_img(tif):
    with rasterio.open(tif) as ds:
        a = np.stack([ds.read(b) for b in VNIR_BANDS]).astype(np.float32)
    valid = np.all((a > 0) & (a != 65535), axis=0)
    return a * REFL, valid


def explained_vs_k(Y, Vlib):
    """Image pixels Y:(n,5). Project (image-centered) onto library directions
    Vlib[:, :k]. Return EV(k) and per-band residual RMSE at each k."""
    mu = Y.mean(0)
    Yc = Y - mu
    tot = (Yc ** 2).sum()
    ev, band_rmse = [], []
    for k in range(1, Y.shape[1] + 1):
        Vk = Vlib[:, :k]
        proj = Yc @ Vk @ Vk.T            # reconstruction of the centered signal
        resid = Yc - proj
        ev.append(1 - (resid ** 2).sum() / tot)
        band_rmse.append(np.sqrt((resid ** 2).mean(0)) * 1e4)
    return np.array(ev), np.array(band_rmse), mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--s2", type=Path, default=Path("egypt_5y_prior"))
    ap.add_argument("--hls", type=Path, default=Path("egypt_5y_prior_hls"))
    a = ap.parse_args()

    # ---- A. hyperspectral intrinsic dimensionality ----
    H = clean_lib(np.load(CACHE / "hyperspectral_vnir.npy").astype(np.float32))
    _, _, wH = pca(H)
    cumH = np.cumsum(wH) / wH.sum()
    print(f"A. Library intrinsic dim (hyperspectral VNIR, {H.shape[0]} spectra, 601 bands)")
    for k in (1, 2, 3, 4, 5, 6, 8):
        print(f"   {k} PCs -> {cumH[k-1]*100:.3f}% surface variance")

    # ---- B. 5-band library PCA ----
    L = clean_lib(np.load(CACHE / "source_sentinel-2a_msi_vnir.npy").astype(np.float32))
    muL, VL, wL = pca(L)
    cumL = np.cumsum(wL) / wL.sum()
    print(f"\nB. Library PCA at 5 S2A VNIR bands ({L.shape[0]} spectra)")
    print("   cum explained var: " + "  ".join(f"k{k}={cumL[k-1]*100:.3f}%" for k in range(1, 6)))
    print(f"   library mean refl (DN): " + " ".join(f"{n}={muL[i]*1e4:.0f}" for i, n in enumerate(NAMES)))
    np.set_printoptions(precision=3, suppress=True)
    for k in range(5):
        print(f"   eigvec{k+1} ({wL[k]/wL.sum()*100:5.2f}%): {VL[:,k]}")

    # ---- C. image explained by library subspace ----
    print(f"\nC. Image VNIR variation explained by library subspace ({a.year}-07)")
    for nm, d in [("S2", a.s2), ("HLS", a.hls)]:
        img, valid = load_img(d / f"egypt_{a.year}-07_prior.tif")
        Y = img[:, valid].T.copy()                       # (n,5) ALL valid pixels
        ev, brmse, mu = explained_vs_k(Y, VL)
        print(f"\n  --- {nm}  (n={Y.shape[0]}) ---")
        print("   EV vs k:  " + "  ".join(f"k{k}={ev[k-1]*100:.3f}%" for k in range(1, 6)))
        print("   image-vs-library mean offset (DN): " +
              " ".join(f"{NAMES[i]}={(mu[i]-muL[i])*1e4:+.0f}" for i in range(5)))
        for k in (3, 4):
            print(f"   residual RMSE @k={k} (DN): " +
                  " ".join(f"{NAMES[i]}={brmse[k-1][i]:.0f}" for i in range(5)))

    # ---- D. leftover direction vs aerosol-like shape ----
    leftover = VL[:, 4]                                   # 5th library direction
    # crude Rayleigh-ish path signal ~ lambda^-4, normalized over band centres
    wl = np.array([443, 490, 560, 665, 842.0])           # S2 VNIR band centres
    ray = wl ** -4.0
    ray = ray / np.linalg.norm(ray)
    cos = abs(float(leftover @ ray))
    print(f"\nD. 5th (least-explained) library direction: {leftover}")
    print(f"   |cos| with Rayleigh lambda^-4 path shape: {cos:.3f}  "
          f"(higher => VNIR residual aligns with atmosphere => more aerosol leverage)")


if __name__ == "__main__":
    raise SystemExit(main())
