"""Per-scene atmosphere sidecar for the Sentinel-2 L1C custom-AC composite.

L1C is top-of-atmosphere. To best-pixel composite *surface* reflectance we
correct each scene's TOA with precomputed 6S coefficients (``xap``, ``xbp``,
``xcp``) carried in a per-scene sidecar produced by the Python pre-step (GEE for
MAIAC AOD / water vapour / geometry, SIAC native 6S for the coefficients). The
6S surface-reflectance relation is::

    y       = xap * rho_toa - xbp
    rho_boa = y / (1 + xcp * y)

Coefficients depend on (AOD, water vapour, geometry, band). AOD and geometry are
fixed per scene; water vapour varies per pixel, so the sidecar ships the
coefficients over a small TCWV LUT and we interpolate by the pixel's (or
scene-mean) water vapour.

This is the Python loader for the per-scene sidecar; the actual 6S
correction runs in Rust via `bestpixel.correct_toa`. The sidecar lets the GEE-based :class:`~surface_priors.sources.s2_l1c_gee`
source correct fetched patches without the native extension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

NODATA = 65535


@dataclass(frozen=True)
class SceneAtmosphere:
    """6S coefficients for one scene over a TCWV LUT, in the sidecar band order.

    ``xap``/``xbp``/``xcp`` are indexed ``[tcwv_node][band]``.
    """

    maiac_aod: float
    wvp: float
    tcwv_nodes: Tuple[float, ...]
    xap: Tuple[Tuple[float, ...], ...]
    xbp: Tuple[Tuple[float, ...], ...]
    xcp: Tuple[Tuple[float, ...], ...]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SceneAtmosphere":
        def grid(key: str) -> Tuple[Tuple[float, ...], ...]:
            return tuple(tuple(float(v) for v in row) for row in d[key])  # type: ignore[index]

        return cls(
            maiac_aod=float(d["maiac_aod"]),  # type: ignore[arg-type]
            wvp=float(d["wvp"]),  # type: ignore[arg-type]
            tcwv_nodes=tuple(float(v) for v in d["tcwv_nodes"]),  # type: ignore[index]
            xap=grid("xap"),
            xbp=grid("xbp"),
            xcp=grid("xcp"),
        )

    def coeffs(self, band: int, wvp: float) -> Tuple[float, float, float]:
        """Linearly interpolate ``(xap, xbp, xcp)`` for one band at water vapour
        ``wvp`` (clamped to the LUT range)."""
        nodes = self.tcwv_nodes
        n = len(nodes)
        if n == 0:
            return 1.0, 0.0, 0.0
        if n == 1:
            return self.xap[0][band], self.xbp[0][band], self.xcp[0][band]
        w = min(max(wvp, nodes[0]), nodes[-1])
        hi = next((i for i, t in enumerate(nodes) if t >= w), n - 1) or 1
        lo = hi - 1
        t0, t1 = nodes[lo], nodes[hi]
        f = 0.0 if abs(t1 - t0) < 1e-6 else (w - t0) / (t1 - t0)
        lerp = lambda a: a[lo][band] + (a[hi][band] - a[lo][band]) * f  # noqa: E731
        return lerp(self.xap), lerp(self.xbp), lerp(self.xcp)


def correct_reflectance(rho_toa, xap: float, xbp: float, xcp: float):
    """6S TOA->BOA for a scalar or numpy array of TOA reflectance (0..1)."""
    y = xap * rho_toa - xbp
    return y / (1.0 + xcp * y)


def _parse_tile_date(scene_id: str) -> Optional[Tuple[str, str]]:
    """Best-effort (MGRS tile, yyyymmdd) from common S2 id forms.

    Handles earth-search STAC ids ``S2A_36RTU_20220720_0_L1C`` and GEE
    ``system:index`` ``20220720T082611_20220720T083855_T36RUU``.
    """
    parts = scene_id.split("_")
    # STAC: S2A_<tile>_<yyyymmdd>_...
    if len(parts) >= 3 and len(parts[1]) == 5 and len(parts[2]) == 8 and parts[2].isdigit():
        return parts[1], parts[2]
    # GEE: <yyyymmddT...>_<...>_T<tile>
    tile = next((p[1:] for p in parts if p.startswith("T") and len(p) == 6), None)
    date = parts[0][:8] if parts and parts[0][:8].isdigit() else None
    if tile and date:
        return tile, date
    return None


@dataclass(frozen=True)
class AtmoSidecar:
    """Per-scene atmosphere keyed by scene id, plus the band order the
    coefficients are in (must match the fetched band order)."""

    bands: Tuple[str, ...]
    scenes: Mapping[str, SceneAtmosphere]

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AtmoSidecar":
        payload = json.loads(Path(path).read_text())
        scenes = {
            str(k): SceneAtmosphere.from_dict(v) for k, v in payload["scenes"].items()
        }
        return cls(bands=tuple(payload["bands"]), scenes=scenes)

    def band_index(self, band: str) -> int:
        return self.bands.index(band)

    def select_low_aod(self, frac: float) -> List[str]:
        """Scene ids ranked by MAIAC AOD ascending, keeping the lowest ``frac``
        (clamped to (0, 1]); the "select clean days" step."""
        ranked = sorted(self.scenes, key=lambda k: self.scenes[k].maiac_aod)
        frac = min(max(frac, 1e-3), 1.0)
        k = max(1, min(len(ranked), round(frac * len(ranked))))
        return ranked[:k]

    def _by_tile_date(self) -> Dict[Tuple[str, str], SceneAtmosphere]:
        index: Dict[Tuple[str, str], SceneAtmosphere] = {}
        for sid, atm in self.scenes.items():
            key = _parse_tile_date(sid)
            if key is not None:
                index.setdefault(key, atm)
        return index

    def lookup(self, scene_id: str) -> Optional[SceneAtmosphere]:
        """Resolve a scene's atmosphere by exact id, else by (tile, date) so a
        GEE ``system:index`` matches a sidecar keyed by earth-search STAC id."""
        atm = self.scenes.get(scene_id)
        if atm is not None:
            return atm
        key = _parse_tile_date(scene_id)
        if key is None:
            return None
        return self._by_tile_date().get(key)

    def correct(
        self,
        scene: SceneAtmosphere,
        band_names: Sequence[str],
        toa: np.ndarray,
        wvp: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """6S-correct a TOA reflectance stack ``toa[band, H, W]`` (0..1) to
        surface reflectance. ``wvp`` (cm) is per-pixel if given, else the
        scene-mean is used. NaN TOA stays NaN."""
        toa = np.asarray(toa, dtype="float32")
        out = np.empty_like(toa)
        per_pixel = wvp is not None
        for i, name in enumerate(band_names):
            b = self.band_index(name)
            if per_pixel:
                w = np.where(np.isfinite(wvp), wvp, scene.wvp).astype("float32")
                xap, xbp, xcp = _coeffs_array(scene, b, w)
            else:
                xap, xbp, xcp = scene.coeffs(b, scene.wvp)
            out[i] = correct_reflectance(toa[i], xap, xbp, xcp)
        return out


def _coeffs_array(
    scene: SceneAtmosphere, band: int, wvp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised per-pixel coefficient interpolation over the TCWV LUT."""
    nodes = np.asarray(scene.tcwv_nodes, dtype="float32")
    if nodes.size == 1:
        shape = wvp.shape
        return (
            np.full(shape, scene.xap[0][band], "float32"),
            np.full(shape, scene.xbp[0][band], "float32"),
            np.full(shape, scene.xcp[0][band], "float32"),
        )
    w = np.clip(wvp, nodes[0], nodes[-1])
    hi = np.clip(np.searchsorted(nodes, w, side="left"), 1, nodes.size - 1)
    lo = hi - 1
    t0, t1 = nodes[lo], nodes[hi]
    f = np.where(np.abs(t1 - t0) < 1e-6, 0.0, (w - t0) / (t1 - t0)).astype("float32")
    xap = np.asarray([row[band] for row in scene.xap], "float32")
    xbp = np.asarray([row[band] for row in scene.xbp], "float32")
    xcp = np.asarray([row[band] for row in scene.xcp], "float32")
    return (xap[lo] + (xap[hi] - xap[lo]) * f,
            xbp[lo] + (xbp[hi] - xbp[lo]) * f,
            xcp[lo] + (xcp[hi] - xcp[lo]) * f)
