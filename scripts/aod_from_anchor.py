"""Solve AOD by anchoring the surface in aerosol-insensitive bands.

Idea (dark-target / MAIAC / SIAC style): red+NIR are largely aerosol-free, so
use them (uncorrected TOA ~ surface) to identify the pixel's surface via the
kNN spectral library, which then PREDICTS the clean blue/green surface
reflectance. Aerosol is whatever explains the gap between observed TOA blue/
green and that predicted surface -> a strong, well-posed signal (aerosol path
reflectance is large in the blue).

Closed loop: forward a known surface to TOA at known AOT*, anchor-predict the
surface from TOA red+NIR, then solve AOD by matching predicted->TOA in the
aerosol-sensitive bands. Compare retrieved AOD to AOT*.

  ~/SIAC/.pixi/envs/default/bin/python scripts/aod_from_anchor.py
"""
from __future__ import annotations

import numpy as np
import rasterio
from spectral_library import SpectralMapper
from spectral_library.distribution import resolve_prepared_library_root

NAMES = ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
WL = np.array([443.0, 490.0, 560.0, 665.0, 865.0, 1610.0, 2190.0])
REFL = 1e-4
SZA, VZA, RAA = 28.0, 5.0, 120.0
RNG = np.random.default_rng(0)
MAPPER = SpectralMapper(resolve_prepared_library_root(), verify_checksums=False)

ANCHOR = [3, 4]              # red, nir  (aerosol-insensitive identity anchor)
SOLVE = [0, 1, 2]           # coastal, blue, green (aerosol-sensitive)


def predict_surface(rows, anchor_bands, k=10):
    """Package kNN-predict the full 7-band surface from only `anchor_bands`.
    NB the package maps VNIR and SWIR as *independent* segments, so SWIR
    anchor bands do not inform the visible prediction."""
    valid = np.zeros((len(rows), 7), bool)
    valid[:, anchor_bands] = True
    r = MAPPER.map_reflectance_batch_arrays_ndarray(
        source_sensor="sentinel-2a_msi",
        reflectance_rows=np.ascontiguousarray(rows, np.float64),
        valid_mask_rows=valid, output_mode="target_sensor",
        target_sensor="sentinel-2a_msi", k=k, min_valid_bands=2,
        neighbor_estimator="distance_weighted_mean", knn_backend="scipy_ckdtree", knn_eps=0.0)
    return np.asarray(r.reflectance, np.float64)


# ---- custom JOINT aerosol-free kNN (red+nir+swir together inform visible) ----
from scipy.spatial import cKDTree  # noqa: E402

_C = resolve_prepared_library_root()
_Lv = np.load(f"{_C}/source_sentinel-2a_msi_vnir.npy").astype(np.float64)   # ub,blue,green,red,nir
_Ls = np.load(f"{_C}/source_sentinel-2a_msi_swir.npy")[:, [1, 2]].astype(np.float64)  # swir1,swir2
_good = np.all(np.isfinite(_Lv), 1) & (_Lv.min(1) >= -0.05) & (_Lv.max(1) <= 1.2)
_Lv, _Ls = _Lv[_good], _Ls[_good]
_ANCHOR_LIB = np.column_stack([_Lv[:, 3], _Lv[:, 4], _Ls[:, 0], _Ls[:, 1]])  # red,nir,swir1,swir2
_VIS_LIB = _Lv[:, :3]                                                          # coastal,blue,green
_TREE = cKDTree(_ANCHOR_LIB)


def joint_predict_vis(rows, k=10):
    """Predict coastal/blue/green from a JOINT kNN over red+nir+swir1+swir2."""
    q = np.column_stack([rows[:, 3], rows[:, 4], rows[:, 5], rows[:, 6]])
    d, idx = _TREE.query(q, k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (_VIS_LIB[idx] * w[:, :, None]).sum(1)                              # (n,3)


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


def forward(rho, aot, wl):
    p, T, S = rt(wl, aot); return p + T*rho/(1-S*rho)


def load(d, n):
    with rasterio.open(f"{d}/egypt_2022-07_prior.tif") as ds:
        a = np.stack([ds.read(b) for b in range(1, 8)]).astype(np.float32)
    v = np.all((a > 0) & (a != 65535), axis=0)
    Y = (a[:, v].T * REFL).astype(np.float64)
    return np.clip(Y[RNG.choice(len(Y), n, replace=False)], 1e-4, 0.95)


def run_method(rho_s, grid, predict_vis, tag):
    """predict_vis: (rows7)->(n,3) predicted coastal/blue/green surface."""
    floor = predict_vis(rho_s)
    fl = {NAMES[b]: np.sqrt(np.mean((floor[:, j]-rho_s[:, b])**2))*1e4 for j, b in enumerate(SOLVE)}
    print(f"\n### {tag}")
    print("  surface-prediction floor vs TRUE (DN RMSE): " +
          " ".join(f"{k}={v:.1f}" for k, v in fl.items()))
    print(f"  {'AOT*':>5} {'scene_AOD':>9} {'px_med':>7} {'px_MAD':>7} {'pred_err_blue':>13}")
    wl_solve = WL[SOLVE]
    for aT in (0.10, 0.30, 0.50):
        toa = forward(rho_s, aT, WL) + RNG.normal(0, 0.003, rho_s.shape)
        vis_pred = predict_vis(toa)                       # TOA anchor (~surface premise)
        pe_blue = np.sqrt(np.mean((vis_pred[:, 1]-rho_s[:, 1])**2))*1e4
        R = np.empty((len(grid), len(rho_s)))
        for i, ac in enumerate(grid):
            R[i] = ((forward(vis_pred, ac, wl_solve)-toa[:, SOLVE])**2).sum(1)
        scene = grid[np.argmin(R.mean(1))]
        px = grid[np.argmin(R, 0)]
        print(f"  {aT:5.2f} {scene:9.2f} {np.median(px):7.2f} "
              f"{np.median(np.abs(px-np.median(px))):7.2f} {pe_blue:13.1f}")


def run_solve_sweep(rho_s, grid):
    """Does the 443 'coastal' (deep-blue analog) band help? Signal vs noise +
    AOD retrieval as a function of which visible bands drive the solve.
    Surface predicted by the JOINT aerosol-free anchor (best predictor)."""
    vis_names = ["coastal(443)", "blue(490)", "green(560)"]
    pred_true = joint_predict_vis(rho_s)                          # (n,3)
    floor = [np.sqrt(np.mean((pred_true[:, j]-rho_s[:, j])**2))*1e4 for j in range(3)]
    toa03 = forward(rho_s, 0.30, WL)
    signal = [np.mean(np.abs(toa03[:, j]-rho_s[:, j]))*1e4 for j in range(3)]
    print("\nper-band signal vs prediction-noise (DN, at AOT*=0.30):")
    print(f"  {'band':12} {'aer.signal':>10} {'pred.noise':>10} {'SNR':>6}")
    for j in range(3):
        print(f"  {vis_names[j]:12} {signal[j]:10.1f} {floor[j]:10.1f} {signal[j]/floor[j]:6.2f}")

    sets = {"443 only": [0], "490 only": [1], "560 only": [2],
            "443+490": [0, 1], "443+490+560": [0, 1, 2]}
    print("\nAOD retrieval by solve-band set (scene / px-median):")
    print(f"  {'solve set':14} {'AOT*=.10':>12} {'AOT*=.30':>12} {'AOT*=.50':>12}")
    for tag, S in sets.items():
        row = [f"  {tag:14}"]
        for aT in (0.10, 0.30, 0.50):
            toa = forward(rho_s, aT, WL) + RNG.normal(0, 0.003, rho_s.shape)
            vis = joint_predict_vis(toa)                          # (n,3)
            R = np.empty((len(grid), len(rho_s)))
            wl_s = WL[S]
            for i, ac in enumerate(grid):
                R[i] = ((forward(vis[:, S], ac, wl_s)-toa[:, S])**2).sum(1)
            scene = grid[np.argmin(R.mean(1))]
            pxm = np.median(grid[np.argmin(R, 0)])
            row.append(f"{scene:.2f}/{pxm:.2f}".rjust(12))
        print("".join(row))


def main():
    rho_s = load("egypt_5y_prior_hls", 4000)
    grid = np.round(np.arange(0.02, 1.001, 0.02), 3)
    print("Does the 443 coastal (deep-blue analog) band help the AOD solve?")
    print("anchor = JOINT red+NIR+SWIR;  truth = HLS 2022 BOA;  RT=compact 6S-style, noise=0.003")
    run_solve_sweep(rho_s, grid)


if __name__ == "__main__":
    main()
