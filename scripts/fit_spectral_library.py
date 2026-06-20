"""Fit a spectral library to every pixel of a prior (VIS-NIR) and score it.

For each valid pixel of an Egypt-case composite we find the brightness-scaled
library spectrum that best matches the 5 visible-to-NIR bands
(coastal/ultra_blue, blue, green, red, nir), using the prepared-runtime data
of the `spectral-library` package (v0.6.3, sentinel-2a_msi convolution, 77125
signatures). We then:

  1. report the VIS-NIR fit residual per band (how close the prior sits to the
     library manifold of physically-real surface spectra), and
  2. PREDICT swir16/swir22 from that same library match (bands the fit never
     used) and compare to the prior's actual SWIR — an independent consistency
     test.

Run for HLS and S2 over identical pixels to see which prior is better explained
by real-world spectra and where (which band) they diverge.

The package module isn't importable in this env, so we read its prepared-runtime
cache directly (that is the library data). Library cols:
  vnir.npy: [ultra_blue, blue, green, red, nir]   (== prior bands 1..5)
  swir.npy: [nir, swir1, swir2]                   (cols 1,2 == swir16, swir22)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

CACHE = Path.home() / ".cache/spectral-library/prepared-runtime/v0.6.3"
# Prior GeoTIFF band order (1-indexed): coastal blue green red nir swir16 swir22
VNIR_BANDS = [1, 2, 3, 4, 5]   # -> library vnir cols 0..4
SWIR_BANDS = [6, 7]            # swir16, swir22 -> library swir cols 1,2
REFL = 1e-4                    # DN -> reflectance


def load_prior(tif: Path):
    with rasterio.open(tif) as ds:
        vnir = np.stack([ds.read(b) for b in VNIR_BANDS]).astype(np.float32)  # (5,H,W)
        swir = np.stack([ds.read(b) for b in SWIR_BANDS]).astype(np.float32)  # (2,H,W)
    valid = np.all((vnir > 0) & (vnir != 65535), axis=0) & np.all(
        (swir > 0) & (swir != 65535), axis=0
    )
    return vnir * REFL, swir * REFL, valid


def fit(Y: np.ndarray, L: np.ndarray, Lswir: np.ndarray, chunk: int = 4000):
    """1-NN brightness-scaled spectral fit. Y:(m,5) refl; L:(K,5); Lswir:(K,2).
    Returns fitted VNIR (m,5), predicted SWIR (m,2), gain (m,)."""
    Ln = np.linalg.norm(L, axis=1)               # |l_i|
    Lsq = (L * L).sum(1)                          # l_i . l_i
    LT = np.ascontiguousarray(L.T)               # (5,K)
    m = Y.shape[0]
    fit_v = np.empty((m, 5), np.float32)
    pred_s = np.empty((m, 2), np.float32)
    gain = np.empty(m, np.float32)
    for s in range(0, m, chunk):
        y = Y[s:s + chunk]                        # (c,5)
        score = (y @ LT) / Ln[None, :]           # argmax cosine == argmax this
        idx = np.argmax(score, axis=1)           # (c,)
        lstar = L[idx]                            # (c,5)
        g = (y * lstar).sum(1) / Lsq[idx]        # optimal brightness
        fit_v[s:s + chunk] = g[:, None] * lstar
        pred_s[s:s + chunk] = g[:, None] * Lswir[idx]
        gain[s:s + chunk] = g
    return fit_v, pred_s, gain


def angle_deg(a, b):
    cs = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def report(name, Yv, Ys, fit_v, pred_s):
    bands = ["coastal", "blue", "green", "red", "nir"]
    sa = angle_deg(Yv, fit_v)
    rmse = np.sqrt(((Yv - fit_v) ** 2).mean())
    print(f"\n=== {name}  (n={Yv.shape[0]}) ===")
    print(f"  VIS-NIR fit: spectral angle  mean {sa.mean():.2f}deg  median {np.median(sa):.2f}deg"
          f"   overall RMSE {rmse*1e4:.0f} DN ({rmse:.4f} refl)")
    print(f"  {'band':8} {'resid_bias':>10} {'resid_RMSE':>10}   (obs - libfit, DN)")
    for i, bn in enumerate(bands):
        r = (Yv[:, i] - fit_v[:, i]) * 1e4
        print(f"  {bn:8} {r.mean():10.1f} {np.sqrt((r**2).mean()):10.1f}")
    print(f"  SWIR PREDICTION (from VIS-NIR fit, bands unseen):")
    print(f"  {'band':8} {'pred_bias':>10} {'pred_RMSE':>10} {'corr':>6}   (obs - pred, DN)")
    for i, bn in enumerate(["swir16", "swir22"]):
        r = (Ys[:, i] - pred_s[:, i]) * 1e4
        c = np.corrcoef(Ys[:, i], pred_s[:, i])[0, 1]
        print(f"  {bn:8} {r.mean():10.1f} {np.sqrt((r**2).mean()):10.1f} {c:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--s2", type=Path, default=Path("egypt_5y_prior"))
    ap.add_argument("--hls", type=Path, default=Path("egypt_5y_prior_hls"))
    a = ap.parse_args()

    L = np.load(CACHE / "source_sentinel-2a_msi_vnir.npy")            # (K,5)
    Lswir = np.load(CACHE / "source_sentinel-2a_msi_swir.npy")[:, [1, 2]]  # swir1,swir2
    # Library quality filter: keep finite, non-negative, plausible spectra.
    good = np.all(np.isfinite(L), 1) & (L.min(1) >= -0.02) & (L.max(1) <= 1.2)
    L, Lswir = L[good].astype(np.float32), Lswir[good].astype(np.float32)
    print(f"library: {L.shape[0]} S2A-convolved VIS-NIR spectra (of 77125)")

    res = {}
    for nm, d in [("S2", a.s2), ("HLS", a.hls)]:
        tif = d / f"egypt_{a.year}-07_prior.tif"
        res[nm] = load_prior(tif)
    # identical pixels: intersect both valid masks
    both = res["S2"][2] & res["HLS"][2]
    print(f"common valid pixels ({a.year}-07): {int(both.sum())}")

    out = {}
    for nm in ("S2", "HLS"):
        vnir, swir, _ = res[nm]
        Yv = vnir[:, both].T.copy()   # (n,5)
        Ys = swir[:, both].T.copy()   # (n,2)
        fit_v, pred_s, _ = fit(Yv, L, Lswir)
        report(nm, Yv, Ys, fit_v, pred_s)
        out[nm] = (Yv, Ys, fit_v, pred_s)

    # head-to-head: which prior is closer to the library manifold
    saS = angle_deg(out["S2"][0], out["S2"][2])
    saH = angle_deg(out["HLS"][0], out["HLS"][2])
    print(f"\n=== head-to-head (lower = closer to real-spectra manifold) ===")
    print(f"  median VIS-NIR spectral angle:  S2 {np.median(saS):.2f}deg   HLS {np.median(saH):.2f}deg")
    for i, bn in enumerate(["swir16", "swir22"]):
        rS = np.sqrt((((out['S2'][1][:, i]-out['S2'][3][:, i])*1e4)**2).mean())
        rH = np.sqrt((((out['HLS'][1][:, i]-out['HLS'][3][:, i])*1e4)**2).mean())
        print(f"  SWIR-pred RMSE {bn}:  S2 {rS:.0f} DN   HLS {rH:.0f} DN")


if __name__ == "__main__":
    raise SystemExit(main())
