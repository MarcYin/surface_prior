"""Synthetic closed-loop: can the library-subspace residual recover a KNOWN AOT?

Step 2 (validation) of the SIAC-style plan. We take a BOA surface prior as
truth, forward-model it to TOA at a known AOT* with a compact 6S-style
single-scattering atmosphere, then invert by searching the AOT that minimises
the surface's distance to the spectral-library manifold (the part-1 residual).
If the library residual is a good aerosol estimator the recovered AOT matches
AOT*; if VNIR leverage is weak (part-1 finding) the per-pixel cost is flat and
only pooling pixels / adding SWIR sharpens it.

The RT is a self-contained single-scattering model (Bodhaine Rayleigh +
Angstrom aerosol + the standard rho_path / T / S coupling). For a closed loop
its absolute fidelity is irrelevant — forward and inverse share it — so the
test isolates the inversion, not the radiometry. On real L1C this RT is
swapped for a SIAC 6S (xap,xb,xc) lookup table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

CACHE = Path.home() / ".cache/spectral-library/prepared-runtime/v0.6.3"
# S2 band centres (nm): coastal blue green red nir(B8A) swir16 swir22
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])
NAMES = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
REFL = 1e-4
RNG = np.random.default_rng(0)

# Fixed acquisition geometry (July Nile Delta, ~10:30 local).
SZA, VZA, RAA = 28.0, 5.0, 120.0


def rt_coeffs(wl_nm, aot550, alpha=1.2, g=0.65, w0=0.95):
    """Compact 6S-style single-scattering coefficients per band.
    Returns path reflectance, two-way transmittance, spherical albedo."""
    lam = np.asarray(wl_nm) / 1000.0  # um
    tr = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4)  # Rayleigh OD
    ta = aot550 * (lam / 0.55) ** (-alpha)                                # aerosol OD
    mus, muv = np.cos(np.radians(SZA)), np.cos(np.radians(VZA))
    cosT = -mus * muv + np.sin(np.radians(SZA)) * np.sin(np.radians(VZA)) * np.cos(np.radians(RAA))
    Pr = 0.75 * (1 + cosT**2)                                             # Rayleigh phase
    Pa = (1 - g**2) / (1 + g**2 - 2 * g * cosT) ** 1.5                    # HG aerosol phase
    path = (tr * Pr + w0 * ta * Pa) / (4 * mus * muv)                     # single-scatter path refl
    # two-way transmittance: scattering forward-peaked -> effective tau < extinction
    tau_t = 0.5 * tr + ta * (1 - w0 * g)
    T = np.exp(-tau_t * (1 / mus + 1 / muv))
    S = 0.92 * tr + 0.33 * ta                                            # atmospheric spherical albedo
    return path, T, S


def forward(rho_s, aot, wl):
    p, T, S = rt_coeffs(wl, aot)
    return p + T * rho_s / (1 - S * rho_s)


def correct(rho_toa, aot, wl):
    p, T, S = rt_coeffs(wl, aot)
    u = rho_toa - p
    return u / (T + S * u)


def pca_basis(X, k):
    mu = X.mean(0)
    Xc = X - mu
    w, V = np.linalg.eigh((Xc.T @ Xc) / (len(X) - 1))
    V = V[:, np.argsort(w)[::-1]]
    return mu, V[:, :k]


def manifold_resid(Y, mu, Vk):
    """Per-pixel distance to the affine library manifold mu + span(Vk)."""
    Yc = Y - mu
    proj = Yc @ Vk @ Vk.T
    return np.sqrt(((Yc - proj) ** 2).sum(1))


def load_truth(tif, n, bands):
    with rasterio.open(tif) as ds:
        a = np.stack([ds.read(b) for b in bands]).astype(np.float32)
    valid = np.all((a > 0) & (a != 65535), axis=0)
    Y = (a[:, valid].T * REFL).astype(np.float32)
    idx = RNG.choice(len(Y), size=min(n, len(Y)), replace=False)
    return Y[idx]


def run(bandset, tif, aot_stars, n=50000, noise=0.003):
    bidx = {"VNIR(5)": [1, 2, 3, 4, 5], "VNIR+SWIR(7)": [1, 2, 3, 4, 5, 6, 7]}[bandset]
    wl = WL[: len(bidx)]
    # library in the matching band space
    Lv = np.load(CACHE / "source_sentinel-2a_msi_vnir.npy").astype(np.float32)
    Ls = np.load(CACHE / "source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float32)
    L = Lv if len(bidx) == 5 else np.hstack([Lv, Ls])
    good = np.all(np.isfinite(L), 1) & (L.min(1) >= -0.05) & (L.max(1) <= 1.2)
    L = L[good]
    # surface intrinsic dim -> k leaves the rest of the space for atmosphere
    k = 4 if len(bidx) == 5 else 5
    mu, Vk = pca_basis(L, k)

    rho_s = load_truth(tif, n, bidx)                      # truth surface (n,B)
    rho_s = np.clip(rho_s, 1e-4, 0.95)
    grid = np.round(np.arange(0.02, 1.001, 0.02), 3)     # candidate AOT

    print(f"\n=== {bandset}  (library {L.shape[0]} spectra, k={k}, n={len(rho_s)} px) ===")
    print(f"  {'AOT*':>5} {'scene_AOT':>9} {'px_AOT med':>11} {'px_AOT MAD':>11} {'sharpness':>9}")
    for aT in aot_stars:
        toa = forward(rho_s, aT, wl)
        toa = toa + RNG.normal(0, noise, toa.shape).astype(np.float32)
        # cost(AOT) per pixel
        R = np.empty((len(grid), len(rho_s)), np.float32)
        for i, ac in enumerate(grid):
            R[i] = manifold_resid(correct(toa, ac, wl), mu, Vk)
        scene_cost = R.mean(1)
        scene_aot = grid[np.argmin(scene_cost)]
        px_aot = grid[np.argmin(R, axis=0)]
        # sharpness: curvature of scene cost near AOT* (higher = better posed)
        j = int(np.argmin(np.abs(grid - aT)))
        lo, hi = max(j - 5, 0), min(j + 5, len(grid) - 1)
        sharp = (scene_cost[lo] + scene_cost[hi] - 2 * scene_cost[j]) / max(scene_cost[j], 1e-6)
        print(f"  {aT:5.2f} {scene_aot:9.2f} {np.median(px_aot):11.2f} "
              f"{np.median(np.abs(px_aot-np.median(px_aot))):11.2f} {sharp:9.2f}")
        if abs(aT - 0.30) < 1e-6:
            probe = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
            cc = [scene_cost[int(np.argmin(np.abs(grid - p)))] for p in probe]
            cc = np.array(cc) / min(cc)
            print("     scene-cost curve (AOT*=0.30, /min): " +
                  " ".join(f"{p}:{c:.2f}" for p, c in zip(probe, cc)))


def nn_residual(Y, L, Ln, Lsq, chunk=2000):
    """Brightness-scaled 1-NN reconstruction residual against the DISCRETE
    library (tighter prior than the linear PCA hull). Y:(n,B) L:(K,B)."""
    LT = np.ascontiguousarray(L.T)
    out = np.empty(len(Y), np.float32)
    for s in range(0, len(Y), chunk):
        y = Y[s:s + chunk]
        idx = np.argmax((y @ LT) / Ln[None, :], axis=1)
        lstar = L[idx]
        g = (y * lstar).sum(1) / Lsq[idx]
        out[s:s + chunk] = np.linalg.norm(y - g[:, None] * lstar, axis=1)
    return out


def run_discrete(tif, aot_stars, n=3000):
    """Same closed loop but cost = discrete-library 1-NN residual on 7 bands,
    plus a hard non-negativity feasibility gate (over-correction -> negative
    reflectance is infeasible). Tests whether a tighter prior is better posed."""
    wl = WL
    Lv = np.load(CACHE / "source_sentinel-2a_msi_vnir.npy").astype(np.float32)
    Ls = np.load(CACHE / "source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float32)
    L = np.hstack([Lv, Ls])
    good = np.all(np.isfinite(L), 1) & (L.min(1) >= -0.05) & (L.max(1) <= 1.2)
    L = L[good]
    Ln, Lsq = np.linalg.norm(L, axis=1), (L * L).sum(1)
    rho_s = np.clip(load_truth(tif, n, [1, 2, 3, 4, 5, 6, 7]), 1e-4, 0.95)
    grid = np.round(np.arange(0.02, 0.81, 0.04), 3)
    print(f"\n=== DISCRETE 1-NN cost + non-negativity  (7 band, lib {L.shape[0]}, n={len(rho_s)}) ===")
    print(f"  {'AOT*':>5} {'scene_AOT':>9} {'px_AOT med':>11} {'px_AOT MAD':>11}")
    for aT in aot_stars:
        toa = forward(rho_s, aT, wl) + RNG.normal(0, 0.003, (len(rho_s), len(wl))).astype(np.float32)
        R = np.empty((len(grid), len(rho_s)), np.float32)
        for i, ac in enumerate(grid):
            sh = correct(toa, ac, wl)
            res = nn_residual(sh, L, Ln, Lsq)
            res[(sh < -0.02).any(1)] = np.inf      # non-negativity feasibility gate
            R[i] = res
        scene_aot = grid[np.argmin(np.nansum(np.where(np.isinf(R), 1e3, R), 1))]
        px_aot = grid[np.argmin(R, axis=0)]
        print(f"  {aT:5.2f} {scene_aot:9.2f} {np.median(px_aot):11.2f} "
              f"{np.median(np.abs(px_aot-np.median(px_aot))):11.2f}")


def main():
    tif = Path("egypt_5y_prior_hls/egypt_2022-07_prior.tif")
    aot_stars = [0.10, 0.30, 0.50]
    print("Synthetic closed-loop AOT recovery (truth = 2022-07 HLS BOA prior)")
    print(f"geometry: SZA={SZA} VZA={VZA} RAA={RAA};  noise=0.003 refl;  RT=compact 6S-style SS")
    for bs in ("VNIR(5)", "VNIR+SWIR(7)"):
        run(bs, tif, aot_stars)
    run_discrete(tif, aot_stars)
    print("\nscene_AOT = AOT from pooling all pixels (scene-constant assumption)")
    print("px_AOT = per-pixel retrieval; sharpness = relative cost curvature at AOT* (bigger=better posed)")


if __name__ == "__main__":
    raise SystemExit(main())
