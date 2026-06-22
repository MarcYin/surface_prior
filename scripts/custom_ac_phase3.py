"""MAIAC-select + 6S-correct composite — Phase 3: correct, composite, compare.

Interpolates the 6S coefficients per pixel by its L2A water vapour, corrects each
day's L1C TOA (custom-MAIAC and custom-AERONET surfaces), best-pixel composites
by Cloud Score+, and compares the strategy (custom-MAIAC) and Sen2Cor-L2A
against the AERONET-truth-AOD surface.

  python scripts/custom_ac_phase3.py --meta /tmp/cac_meta.json \
      --pixels /tmp/cac_pixels.npz --coeffs /tmp/cac_coeffs.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def correct(toa, c):                              # c: (...,3,7) = xap,xbp,xcp
    xap, xbp, xcp = c[..., 0, :], c[..., 1, :], c[..., 2, :]
    y = xap * toa - xbp
    return y / (1 + xcp * y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pixels", required=True)
    ap.add_argument("--coeffs", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.meta)); sel = meta["selected"]; B = meta["bands"]
    cube = np.load(a.pixels)["cube"]              # (D, P, 16): toa7, sr7, wvp, cs
    Z = np.load(a.coeffs); co = Z["coeffs"]; tg = Z["tcwv"]   # co: (D,2,ntcwv,3,7)
    toa = cube[:, :, 0:7]; sr = cube[:, :, 7:14]; wvp = cube[:, :, 14]; cs = cube[:, :, 15]
    D, P, _ = toa.shape

    # best-pixel composite: per point, clearest day with finite TOA
    clear = (cs > 0.6) & np.isfinite(toa).all(2)
    win = np.argmax(np.where(clear, cs, -np.inf), axis=0); idx = np.arange(P)
    toa_w = toa[win, idx]; sr_w = sr[win, idx]; wvp_w = wvp[win, idx]
    co_w = co[win]                                # (P,2,ntcwv,3,7)

    # per-pixel WVP interpolation of coefficients
    wc = np.clip(np.nan_to_num(wvp_w, nan=float(tg.mean())), tg[0], tg[-1])
    j = np.clip(np.searchsorted(tg, wc) - 1, 0, len(tg) - 2)
    frac = (wc - tg[j]) / (tg[j + 1] - tg[j])
    lo = co_w[idx, :, j]; hi = co_w[idx, :, j + 1]          # (P,2,3,7)
    cf = lo + (hi - lo) * frac[:, None, None, None]
    cm = correct(toa_w, cf[:, 0]); ca = correct(toa_w, cf[:, 1])

    valid = clear.any(0) & np.isfinite(cm).all(1) & np.isfinite(sr_w).all(1) & np.isfinite(ca).all(1)
    cm, ca, sr_w = cm[valid], ca[valid], sr_w[valid]
    maoa = np.array([s["maiac"] for s in sel]); aeoa = np.array([s["aeronet_op"] or s["maiac"] for s in sel])
    print(f"composite over {D} low-MAIAC days, {valid.sum()} valid points")
    print(f"winning-pixel AOD: MAIAC {maoa[win[valid]].mean():.3f}  AERONET {aeoa[win[valid]].mean():.3f}")
    print("\n  per-band vs AERONET-truth surface (DN):")
    print(f"  {'band':6} {'truth':>6} {'MAIAC':>6} {'L2A':>6} | {'MAIAC-tru':>9} {'L2A-tru':>8}")
    for i, b in enumerate(B):
        print(f"  {b:6} {ca[:, i].mean()*1e4:6.0f} {cm[:, i].mean()*1e4:6.0f} {sr_w[:, i].mean()*1e4:6.0f} | "
              f"{(cm[:, i]-ca[:, i]).mean()*1e4:9.0f} {(sr_w[:, i]-ca[:, i]).mean()*1e4:8.0f}")
    print("\n  RMSE vs truth (DN):")
    for i, b in enumerate(B):
        rm = np.sqrt(((cm[:, i]-ca[:, i])**2).mean())*1e4; rl = np.sqrt(((sr_w[:, i]-ca[:, i])**2).mean())*1e4
        print(f"  {b:6} custom-MAIAC {rm:5.0f}   Sen2Cor-L2A {rl:5.0f}   {'MAIAC' if rm < rl else 'L2A'} closer")


if __name__ == "__main__":
    main()
