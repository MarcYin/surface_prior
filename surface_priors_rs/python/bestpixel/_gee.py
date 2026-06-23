"""Google Earth Engine fetch helpers for the bestpixel L1C pipeline.

Pulls AOI patches directly as numpy via ``ee.data.getPixels`` (raw assets, no
server-side compute graph — ~2.5x faster than ``computePixels``) for L1C TOA and
Cloud Score+, plus a coarse ``computePixels`` scout. earthengine-api is imported
lazily so importing :mod:`bestpixel` never requires the ``gee`` extra.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
from typing import Any, Optional, Sequence, Tuple

import numpy as np

L1C_COLLECTION = "COPERNICUS/S2_HARMONIZED"
CS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"


def init_ee(service_account: Optional[str] = None, key: Optional[str] = None,
            ee_module: Optional[Any] = None) -> Any:
    """Return an initialised ``ee`` module. Uses an existing session if live,
    else a service account from args or GEE_SERVICE_ACCOUNT[_KEY], else the
    default credentials."""
    if ee_module is not None:
        return ee_module
    try:
        import ee
    except ImportError as exc:  # pragma: no cover
        raise ImportError("bestpixel L1C requires the 'gee' extra: "
                          "pip install 'bestpixel[gee]'") from exc
    try:
        ee.Number(1).getInfo()
        return ee
    except Exception:
        pass
    sa = service_account or os.environ.get("GEE_SERVICE_ACCOUNT")
    kp = key or os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if sa and kp:
        ee.Initialize(ee.ServiceAccountCredentials(sa, kp))
    else:
        ee.Initialize()
    return ee


def utm_epsg_from_bbox(bbox: Sequence[float]) -> int:
    west, south, east, north = (float(v) for v in bbox)
    lon, lat = (west + east) / 2.0, (south + north) / 2.0
    zone = max(1, min(60, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def utm_grid(bbox: Sequence[float], epsg: int, res: float) -> dict:
    """Fixed north-up UTM grid (crs/res/origin/W/H) from a WGS84 bbox."""
    from pyproj import Transformer
    crs = f"EPSG:{epsg}"
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]],
                          [bbox[1], bbox[3], bbox[3], bbox[1]])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w = int(round((x1 - x0) / res))
    h = int(round((y1 - y0) / res))
    return {"crs": crs, "res": float(res), "x0": x0, "y1": y1, "W": w, "H": h, "epsg": int(epsg)}


def _day_bounds(ymd: str) -> Tuple[str, str]:
    d = _dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
    return d.isoformat(), (d + _dt.timedelta(days=1)).isoformat()


def resolve_assets(ee: Any, scene_ids: Sequence[str], geom: Any) -> dict:
    """Map sidecar STAC ids (``S2A_36RTU_20220720_0_L1C``) -> (l1c_asset,
    cs_asset) via the GEE system:index, resolved in one batched getInfo."""
    exprs = []
    for sid in scene_ids:
        tile, ymd = sid.split("_")[1], sid.split("_")[2]
        d0, d1 = _day_bounds(ymd)
        img = ee.Image(ee.ImageCollection(L1C_COLLECTION).filterBounds(geom)
                       .filterDate(d0, d1).filter(ee.Filter.eq("MGRS_TILE", tile)).first())
        exprs.append(img.get("system:index"))
    indices = ee.List(exprs).getInfo()
    return {sid: (f"{L1C_COLLECTION}/{idx}", f"{CS_COLLECTION}/{idx}")
            for sid, idx in zip(scene_ids, indices) if idx}


def _structured(raw: Any, band_ids: Sequence[str]) -> Tuple[np.ndarray, ...]:
    arr = np.asarray(raw)
    if arr.dtype.names is None:
        return (arr.astype("float32", copy=False),)
    return tuple(np.asarray(arr[b], dtype="float32") for b in band_ids)


def get_patch(ee: Any, source: Any, band_ids: Sequence[str], *, grid: dict,
              c0: int = 0, r0: int = 0, cw: Optional[int] = None, ch: Optional[int] = None,
              is_asset: bool = True) -> np.ndarray:
    """Fetch ``band_ids`` over a window of ``grid`` as (bands, ch, cw).

    ``source`` is a raw asset id (``getPixels``) or an ee.Image (``computePixels``).
    """
    cw = grid["W"] if cw is None else cw
    ch = grid["H"] if ch is None else ch
    res = grid["res"]
    request = {
        "fileFormat": "NUMPY_NDARRAY", "bandIds": list(band_ids),
        "grid": {"dimensions": {"width": int(cw), "height": int(ch)}, "crsCode": grid["crs"],
                 "affineTransform": {"scaleX": res, "shearX": 0, "translateX": grid["x0"] + c0 * res,
                                     "shearY": 0, "scaleY": -res, "translateY": grid["y1"] - r0 * res}}}
    if is_asset:
        request["assetId"] = source
        raw = ee.data.getPixels(request)
    else:
        request["expression"] = source
        raw = ee.data.computePixels(request)
    return np.stack(_structured(raw, band_ids), axis=0)


def scout_clear(ee: Any, cs_asset: str, grid: dict, factor: int, thresh: float) -> np.ndarray:
    """Coarse (decimated ``factor``x) Cloud Score+ clear bitmap for one scene."""
    ws = max(1, math.ceil(grid["W"] / factor))
    hs = max(1, math.ceil(grid["H"] / factor))
    coarse = {**grid, "res": grid["res"] * factor, "W": ws, "H": hs}
    arr = get_patch(ee, cs_asset, ["cs"], grid=coarse, is_asset=True)
    return arr[0] > thresh
