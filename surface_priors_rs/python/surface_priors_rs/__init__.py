"""Rust-backed Sentinel-2 / HLS / MCD43A4 composite builder.

Thin package over the compiled extension: it re-exports everything from
the native module unchanged and ships a type stub (``__init__.pyi``) so
``build_composite`` / ``build_monthly_composites`` — including the
adaptive-depth (``coverage_target`` / ``min_k`` / ``max_k``) and
windowed-fetch (``windowed_fetch``) parameters — are typed and
discoverable from Python.

It also pins ``PROJ_DATA`` before the native module loads — see
``_ensure_proj_data`` — so the WGS84->UTM transform is robust no matter
when a GDAL/PROJ-using library (rasterio, fiona, pyproj) is imported.
"""

import os as _os
import sys as _sys


def _ensure_proj_data() -> None:
    """Pin libproj's data dir so the native transform survives any import order.

    The extension links libproj and uses its *default* PROJ context. If a
    GDAL/PROJ-using library (e.g. rasterio) initialises PROJ first in the
    same process, that default context can lose its data path and the
    WGS84->UTM transform aborts with "proj_create: ... no database context".
    Setting ``PROJ_DATA`` to this environment's ``proj.db`` directory (the
    same data GDAL/rasterio use in a conda/pixi env, so it's harmless to
    them) makes the transform resolve correctly regardless of order.

    Respects an existing ``PROJ_DATA``/``PROJ_LIB`` — only sets it when the
    user/tooling hasn't already chosen one.
    """
    if _os.environ.get("PROJ_DATA") or _os.environ.get("PROJ_LIB"):
        return
    candidates = [
        _os.path.join(_sys.prefix, "share", "proj"),
        _os.path.join(_os.environ.get("CONDA_PREFIX", ""), "share", "proj"),
    ]
    for cand in candidates:
        if cand and _os.path.isfile(_os.path.join(cand, "proj.db")):
            _os.environ["PROJ_DATA"] = cand
            _os.environ["PROJ_LIB"] = cand  # older libproj reads PROJ_LIB
            return


_ensure_proj_data()

from .surface_priors_rs import *  # noqa: E402,F401,F403
from . import surface_priors_rs as _core  # noqa: E402

__doc__ = _core.__doc__
if hasattr(_core, "__all__"):
    __all__ = _core.__all__
