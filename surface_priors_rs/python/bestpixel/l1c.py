"""End-to-end Sentinel-2 **L1C custom-AC** monthly composite.

MAIAC-selects low-aerosol days, fetches raw L1C TOA + Cloud Score+ from GEE
(scout-first, only the patches that win), 6S-corrects each scene to surface
reflectance via the Rust core (:func:`bestpixel.correct_toa`), and best-pixel
composites preferring the lowest-AOD clear observation.

L1C is JP2-on-s3 (not an HTTP COG), so the fetch is GEE-side (Python); the
correction + the rest of the package's compositing stay Rust-backed.

    import bestpixel as bp
    out = bp.build_l1c_composite(
        bbox=(31.0, 29.9, 31.5, 30.3), datetime=("2022-07-01", "2022-08-01"),
        sidecar="cairo_sidecar.json", resolution=60, out="cairo_l1c.tif")
    blue = out["bands"]["blue"]            # (H, W) surface reflectance
"""
from __future__ import annotations

import concurrent.futures as cf
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ._gee import get_patch, init_ee, resolve_assets, scout_clear, utm_epsg_from_bbox, utm_grid
from .atmosphere import AtmoSidecar
from .bestpixel import correct_toa

# sidecar/6S band name -> GEE L1C band id
SIDECAR_TO_GEE = {
    "coastal": "B1", "blue": "B2", "green": "B3", "red": "B4",
    "nir08": "B8A", "swir16": "B11", "swir22": "B12",
}
TOA_SCALE = 10000.0


def _chunk_windows(w: int, h: int, chunk: int):
    return [(c0, r0, min(chunk, w - c0), min(chunk, h - r0))
            for r0 in range(0, h, chunk) for c0 in range(0, w, chunk)]


def _select_chunk(fracs, target, min_k, max_k, min_frac=0.02):
    """Greedily stack scenes by clear fraction until coverage 1-prod(1-f) hits
    target (min_k..max_k); only fetch the patches a chunk needs."""
    ranked = sorted((sf for sf in fracs if sf[1] > min_frac), key=lambda x: -x[1])
    chosen, cov = [], 0.0
    for sid, f in ranked:
        if len(chosen) >= max_k:
            break
        chosen.append(sid)
        cov = 1.0 - (1.0 - cov) * (1.0 - f)
        if len(chosen) >= min_k and cov >= target:
            break
    return chosen


def build_l1c_composite(
    bbox: Sequence[float],
    datetime: Tuple[str, str],
    sidecar: str,
    *,
    resolution: float = 60.0,
    epsg: Optional[int] = None,
    bands: Optional[Sequence[str]] = None,
    low_aod_frac: float = 0.6,
    cs_thresh: float = 0.6,
    rank: str = "aod",
    chunk: int = 1024,
    scout_factor: int = 8,
    coverage_target: float = 0.98,
    min_k: int = 2,
    max_k: int = 8,
    workers: int = 16,
    out: Optional[str] = None,
    ee_module: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build an L1C custom-AC surface-reflectance composite.

    Returns ``{"bands": {name: (H,W) float32}, "grid": {...}, "count": (H,W),
    "scenes": [...]}`` and, if ``out`` is given, writes a scaled int16 GeoTIFF.
    """
    ee = init_ee(ee_module=ee_module)
    sc = AtmoSidecar.load(sidecar)
    band_names = list(bands) if bands else list(sc.bands)
    gee_bands = [SIDECAR_TO_GEE[b] for b in band_names]
    band_idx = [sc.band_index(b) for b in band_names]
    nb = len(band_names)

    # clean-day gate (lowest-AOD frac), bounded to the requested window
    d0, d1 = str(datetime[0])[:10], str(datetime[1])[:10]

    def _in_window(sid: str) -> bool:
        ymd = sid.split("_")[2]
        return d0 <= f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}" <= d1

    sel = [s for s in sc.select_low_aod(low_aod_frac) if _in_window(s)]
    epsg = epsg or utm_epsg_from_bbox(bbox)
    grid = utm_grid(bbox, epsg, resolution)
    geom = ee.Geometry.Rectangle(list(bbox))
    assets = resolve_assets(ee, sel, geom)
    sel = [s for s in sel if s in assets]
    if not sel:
        raise RuntimeError("no scenes resolved from sidecar over the AOI/period")

    w, h, f = grid["W"], grid["H"], scout_factor

    # scout: coarse Cloud Score+ clear bitmap per scene
    def _scout(sid):
        try:
            return scout_clear(ee, assets[sid][1], grid, f, cs_thresh)
        except Exception:
            return None
    with cf.ThreadPoolExecutor(max_workers=min(len(sel), workers)) as ex:
        coarse = {s: c for s, c in zip(sel, ex.map(_scout, sel)) if c is not None}

    # per-chunk selection
    chunks = _chunk_windows(w, h, chunk)
    win_of, plan = {}, []
    for ci, (c0, r0, cw, ch) in enumerate(chunks):
        win_of[ci] = (c0, r0, cw, ch)
        sc0, sr0 = c0 // f, r0 // f
        sc1, sr1 = math.ceil((c0 + cw) / f), math.ceil((r0 + ch) / f)
        fracs = [(s, float(coarse[s][sr0:sr1, sc0:sc1].mean())) for s in coarse]
        plan.append((ci, _select_chunk(fracs, coverage_target, min_k, max_k)))

    # per-scene 6S coeffs at scene-mean WVP (one set per band)
    coeffs = {}
    for sid in coarse:
        atm = sc.scenes[sid]
        triples = [atm.coeffs(bi, atm.wvp) for bi in band_idx]
        xap = [t[0] for t in triples]
        xbp = [t[1] for t in triples]
        xcp = [t[2] for t in triples]
        coeffs[sid] = (xap, xbp, xcp, float(atm.maiac_aod))

    # windowed fetch: every needed (scene, chunk) x {l1c, cs} in one pool
    units = [(ci, sid, kind) for ci, pick in plan for sid in pick for kind in ("l1c", "cs")]

    def _fetch(unit):
        ci, sid, kind = unit
        c0, r0, cw, ch = win_of[ci]
        try:
            if kind == "l1c":
                return get_patch(ee, assets[sid][0], gee_bands, grid=grid, c0=c0, r0=r0, cw=cw, ch=ch)
            return get_patch(ee, assets[sid][1], ["cs"], grid=grid, c0=c0, r0=r0, cw=cw, ch=ch)
        except Exception:
            return None
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fetched = list(zip(units, ex.map(_fetch, units)))

    pieces: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
    for (ci, sid, kind), arr in fetched:
        if arr is not None:
            pieces.setdefault((ci, sid), {})[kind] = arr

    # composite: gate cs, rank by lowest AOD (cs tie-break) or highest cs
    best = np.full((nb, h, w), np.nan, np.float32)
    best_cs = np.full((h, w), -1.0, np.float32)
    best_aod = np.full((h, w), np.inf, np.float32)
    count = np.zeros((h, w), np.int16)
    for (ci, sid), p in pieces.items():
        if "l1c" not in p or "cs" not in p:
            continue
        c0, r0, cw, ch = win_of[ci]
        toa = (p["l1c"] / TOA_SCALE).astype(np.float32)
        cs = p["cs"][0]
        xap, xbp, xcp, aod = coeffs[sid]
        surf = np.asarray(correct_toa(np.ascontiguousarray(toa), xap, xbp, xcp))  # Rust core
        clear = cs > cs_thresh
        sl = (slice(r0, r0 + ch), slice(c0, c0 + cw))
        count[sl] += clear.astype(np.int16)
        bcs, baod = best_cs[sl], best_aod[sl]
        if rank == "aod":
            win = clear & ((aod < baod) | ((aod == baod) & (cs > bcs)))
        else:
            win = clear & (cs > bcs)
        for bi in range(nb):
            best[bi][sl][win] = surf[bi][win]
        bcs[win] = cs[win]
        baod[win] = aod

    result = {
        "bands": {name: best[i] for i, name in enumerate(band_names)},
        "grid": grid,
        "count": count,
        "scenes": list(coarse),
    }
    if out is not None:
        _write_geotiff(out, best, band_names, grid)
        result["path"] = out
    return result


def _write_geotiff(path: str, surf: np.ndarray, band_names: Sequence[str], grid: dict) -> None:
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    nb, h, w = surf.shape
    dn = np.where(np.isfinite(surf), np.clip(surf * 1e4, 0, 32767), 0).astype(np.int16)
    transform = Affine(grid["res"], 0, grid["x0"], 0, -grid["res"], grid["y1"])
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=nb, dtype="int16",
                       crs=CRS.from_epsg(grid["epsg"]), transform=transform, compress="deflate",
                       predictor=2, tiled=True) as dst:
        for i, name in enumerate(band_names):
            dst.write(dn[i], i + 1)
            dst.set_band_description(i + 1, name)
        dst.scales = [1e-4] * nb
