"""Google Earth Engine source for Sentinel-2 **L1C** with custom atmospheric
correction (6S sidecar) and Cloud Score+ quality.

L1C is top-of-atmosphere, so unlike :mod:`surface_priors.sources.s2_gee` (which
reads ready-made L2A surface reflectance) this source fetches raw TOA via
``ee.data.getPixels`` — no server-side compute graph, ~2.5x faster than
``computePixels`` for raw assets — and corrects each chunk to surface
reflectance with the scene's 6S coefficients from an
:class:`~surface_priors.atmosphere.AtmoSidecar`.

The clean-day gate, scout, per-chunk selection, and best-pixel compositing are
all inherited from the package's source/provider machinery. Two things are
L1C-specific: (1) candidates are restricted to the lowest-AOD ``low_aod_frac``
of the sidecar's scenes; (2) quality ranks clear pixels by *lowest MAIAC AOD*
(most reliable correction) via :func:`aod_cloud_score_to_quality`, not by bare
Cloud Score+ which saturates near 1.0 on clear pixels.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from surface_priors.atmosphere import AtmoSidecar
from surface_priors.chunks import ChunkLayout, ChunkWindow, chunk_grid
from surface_priors.selection import SceneChunkStats, SelectionPlan
from surface_priors.sources.s2 import (
    CLOUD_SCORE_PLUS_COLLECTION_ID,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CLEAR_THRESHOLD,
    DEFAULT_SCORE_BAND,
    DEFAULT_SCOUT_FACTOR,
    S2_L1C_BANDS,
    S2_L1C_COLLECTION_ID,
    S2_TOA_SCALE,
    aggregate_chunk_stats,
    aod_cloud_score_to_quality,
    cloud_score_valid_mask,
)
from surface_priors.temporal import sample_temporal_ranges, temporal_ranges_name
from surface_priors.types import GridSpec, Observation

DEFAULT_LOW_AOD_FRAC = 0.6


@dataclass(frozen=True)
class S2L1CScene:
    """One candidate Sentinel-2 L1C scene with its matched atmosphere AOD."""

    scene_index: int
    l1c_image_id: str
    cs_image_id: str
    system_index: str
    timestamp_ms: int
    maiac_aod: float

    @property
    def short_id(self) -> str:
        return self.system_index


class S2L1CGeeSource:
    """Chunked Sentinel-2 L1C custom-AC source over GEE + 6S sidecar.

    Authentication, listing, scout, and fetch run lazily so the module imports
    without earthengine-api configured. Pass ``ee_module`` to drive it in tests.
    """

    def __init__(
        self,
        *,
        temporal_ranges: Sequence[Tuple[str, str]],
        atmosphere: AtmoSidecar,
        low_aod_frac: float = DEFAULT_LOW_AOD_FRAC,
        sample_every_days: Optional[int] = None,
        score_band: str = DEFAULT_SCORE_BAND,
        clear_threshold: float = DEFAULT_CLEAR_THRESHOLD,
        scout_factor: int = DEFAULT_SCOUT_FACTOR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_scenes: Optional[int] = None,
        name: Optional[str] = None,
        ee_module: Optional[Any] = None,
    ) -> None:
        if not temporal_ranges:
            raise ValueError("S2L1CGeeSource requires explicit temporal_ranges")
        if scout_factor <= 0:
            raise ValueError("scout_factor must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.temporal_ranges = tuple((str(start), str(end)) for start, end in temporal_ranges)
        self.atmosphere = atmosphere
        self.low_aod_frac = float(low_aod_frac)
        self.sample_every_days = None if sample_every_days is None else int(sample_every_days)
        self.query_temporal_ranges = sample_temporal_ranges(
            self.temporal_ranges, sample_every_days=self.sample_every_days
        )
        self.score_band = str(score_band)
        self.clear_threshold = float(clear_threshold)
        self.scout_factor = int(scout_factor)
        self.chunk_size = int(chunk_size)
        self.max_scenes = None if max_scenes is None else int(max_scenes)
        self._ee_module = ee_module
        temporal_key = temporal_ranges_name(
            self.temporal_ranges, sample_every_days=self.sample_every_days
        )
        self._name = name or (
            f"s2-l1c-gee:cs:{self.score_band}:aod{self.low_aod_frac:g}:"
            f"{self.scout_factor}x:chunk{self.chunk_size}:{temporal_key}"
        )
        self._scenes: Optional[Tuple[S2L1CScene, ...]] = None
        self._scene_cache_key: Optional[tuple] = None

    @property
    def name(self) -> str:
        return self._name

    def block_size(self, *, grid: GridSpec, band_names: Sequence[str]) -> Optional[int]:
        del grid, band_names
        return None

    def resolve_grid(
        self,
        *,
        wgs84_bounds: Sequence[float],
        native_crs: str,
        resolution: float,
        band_names: Sequence[str],
    ) -> GridSpec:
        del native_crs, band_names
        utm_crs = _utm_crs_from_wgs84_bounds(wgs84_bounds)
        utm_bounds = _transform_wgs84_to_crs(wgs84_bounds, utm_crs)
        snapped = _snap_bounds_to_resolution(utm_bounds, resolution=resolution)
        return GridSpec.from_bounds(
            bounds=snapped, crs=utm_crs, resolution=resolution, wgs84_bounds=wgs84_bounds
        )

    def list_scenes(self, *, grid: GridSpec) -> Tuple[S2L1CScene, ...]:
        if grid.wgs84_bounds is None:
            raise ValueError("S2L1CGeeSource requires GridSpec with WGS84 bounds")
        key = (
            tuple(float(v) for v in grid.wgs84_bounds),
            self.query_temporal_ranges,
            self.max_scenes,
            self.low_aod_frac,
        )
        if self._scenes is not None and self._scene_cache_key == key:
            return self._scenes
        ee = self._ee()
        bbox = ee.Geometry.BBox(*grid.wgs84_bounds)
        # clean-day gate: AOD cut from the sidecar's lowest-frac selection
        selected = self.atmosphere.select_low_aod(self.low_aod_frac)
        aod_cut = max(self.atmosphere.scenes[s].maiac_aod for s in selected)
        items: list[dict[str, Any]] = []
        for start, end in self.query_temporal_ranges:
            collection = (
                ee.ImageCollection(S2_L1C_COLLECTION_ID)
                .filterBounds(bbox)
                .filterDate(start, end)
                .sort("system:time_start")
            )
            info = collection.aggregate_array("system:index").getInfo() or []
            times = collection.aggregate_array("system:time_start").getInfo() or []
            for system_index, ts in zip(info, times):
                items.append({"system_index": str(system_index), "timestamp_ms": int(ts)})
        items.sort(key=lambda entry: entry["timestamp_ms"])
        scenes: list[S2L1CScene] = []
        for item in items:
            atm = self.atmosphere.lookup(item["system_index"])
            if atm is None or atm.maiac_aod > aod_cut:
                continue
            scenes.append(
                S2L1CScene(
                    scene_index=len(scenes),
                    l1c_image_id=f"{S2_L1C_COLLECTION_ID}/{item['system_index']}",
                    cs_image_id=f"{CLOUD_SCORE_PLUS_COLLECTION_ID}/{item['system_index']}",
                    system_index=item["system_index"],
                    timestamp_ms=item["timestamp_ms"],
                    maiac_aod=float(atm.maiac_aod),
                )
            )
            if self.max_scenes is not None and len(scenes) >= self.max_scenes:
                break
        self._scenes = tuple(scenes)
        self._scene_cache_key = key
        return self._scenes

    def scout(
        self,
        *,
        grid: GridSpec,
        layout: ChunkLayout,
        band_names: Sequence[str],
        temporal_filter: Optional[Tuple[str, str]] = None,
    ) -> Sequence[SceneChunkStats]:
        del band_names
        scenes = self.list_scenes(grid=grid)
        if temporal_filter is not None:
            scenes = _filter_scenes_by_datetime(scenes, *temporal_filter)
        if not scenes:
            return ()
        ee = self._ee()
        scale = float(grid.resolution) * self.scout_factor
        coarse_grid = _coarse_grid(grid=grid, scout_factor=self.scout_factor)
        stats: list[SceneChunkStats] = []
        for scene in scenes:
            score = self._fetch_coarse_score(ee=ee, scene=scene, coarse_grid=coarse_grid, scale=scale)
            if score is None:
                stats.extend(
                    SceneChunkStats(
                        scene_index=scene.scene_index,
                        chunk_id=window.chunk_id,
                        usable_fraction=0.0,
                        mean_clear=float("nan"),
                    )
                    for window in layout
                )
                continue
            valid = cloud_score_valid_mask(score)
            stats.extend(
                aggregate_chunk_stats(
                    scene_index=scene.scene_index,
                    coarse_score=score,
                    coarse_valid=valid,
                    layout=layout,
                    scout_factor=self.scout_factor,
                )
            )
        return tuple(stats)

    def fetch_selected(
        self,
        *,
        grid: GridSpec,
        plan: SelectionPlan,
        band_names: Sequence[str],
        scene_index: int,
        chunk_id: int,
    ) -> Optional[Observation]:
        scenes = self.list_scenes(grid=grid)
        scene = next((s for s in scenes if s.scene_index == scene_index), None)
        if scene is None:
            return None
        return self._fetch_chunk(
            grid=grid,
            window=plan.layout[chunk_id],
            scene=scene,
            band_names=tuple(str(b) for b in band_names),
        )

    def _fetch_coarse_score(
        self, *, ee: Any, scene: S2L1CScene, coarse_grid: GridSpec, scale: float
    ) -> Optional[np.ndarray]:
        cs_image = ee.Image(scene.cs_image_id).select(self.score_band)
        image = cs_image.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=4096, bestEffort=True
        ).reproject(crs=coarse_grid.crs, scale=scale)
        request = _pixels_request(grid=coarse_grid, band_ids=(self.score_band,), expression=image)
        try:
            raw = ee.data.computePixels(request)
        except Exception:
            return None
        return _structured_to_2d(raw, band=self.score_band)

    def _fetch_chunk(
        self, *, grid: GridSpec, window: ChunkWindow, scene: S2L1CScene, band_names: Sequence[str]
    ) -> Optional[Observation]:
        ee = self._ee()
        atm = self.atmosphere.lookup(scene.system_index)
        if atm is None:
            return None
        sub_grid = chunk_grid(grid, window)
        toa_band_ids = tuple(S2_L1C_BANDS[b][0] for b in band_names)
        sidecar_bands = tuple(S2_L1C_BANDS[b][1] for b in band_names)
        # raw TOA via getPixels (no compute graph), Cloud Score+ via getPixels too
        toa_raw = _get_pixels(
            ee, asset_id=scene.l1c_image_id, grid=sub_grid, band_ids=toa_band_ids
        )
        score = _get_pixels(
            ee, asset_id=scene.cs_image_id, grid=sub_grid, band_ids=(self.score_band,)
        )
        if toa_raw is None or score is None:
            return None
        toa = np.stack(toa_raw, axis=0).astype("float32") * S2_TOA_SCALE
        boa = self.atmosphere.correct(atm, sidecar_bands, toa)
        boa = np.clip(boa, 0.0, None).astype("float32", copy=False)
        score_2d = score[0]
        quality = aod_cloud_score_to_quality(
            score_2d, aod=atm.maiac_aod, clear_threshold=self.clear_threshold
        )
        return Observation(
            data=boa,
            quality=quality,
            band_names=band_names,
            source_id=scene.system_index,
            metadata={
                "l1c_image_id": scene.l1c_image_id,
                "cs_image_id": scene.cs_image_id,
                "system_index": scene.system_index,
                "timestamp_ms": scene.timestamp_ms,
                "maiac_aod": scene.maiac_aod,
                "chunk_id": int(window.chunk_id),
            },
        )

    def _ee(self) -> Any:
        if self._ee_module is not None:
            return self._ee_module
        try:
            import ee  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "S2L1CGeeSource requires the 'gee' extra: pip install 'surface-priors[gee]'"
            ) from exc
        _ensure_initialized(ee)
        return ee


def _get_pixels(
    ee: Any, *, asset_id: str, grid: GridSpec, band_ids: Sequence[str]
) -> Optional[Tuple[np.ndarray, ...]]:
    request = _pixels_request(grid=grid, band_ids=band_ids, asset_id=asset_id)
    try:
        raw = ee.data.getPixels(request)
    except Exception:
        return None
    return _structured_to_band_stack(raw, band_ids=band_ids)


def _pixels_request(
    *,
    grid: GridSpec,
    band_ids: Sequence[str],
    expression: Any = None,
    asset_id: Optional[str] = None,
) -> Mapping[str, Any]:
    xmin, _ymin, _xmax, ymax = grid.bounds
    request: dict[str, Any] = {
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": list(band_ids),
        "grid": {
            "dimensions": {"width": int(grid.width), "height": int(grid.height)},
            "crsCode": grid.crs,
            "affineTransform": {
                "scaleX": float(grid.resolution), "shearX": 0.0, "translateX": float(xmin),
                "shearY": 0.0, "scaleY": -float(grid.resolution), "translateY": float(ymax),
            },
        },
    }
    if asset_id is not None:
        request["assetId"] = asset_id
    else:
        request["expression"] = expression
    return request


def _structured_to_band_stack(raw: Any, *, band_ids: Sequence[str]) -> Tuple[np.ndarray, ...]:
    array = np.asarray(raw)
    if array.dtype.names is None:
        if len(band_ids) != 1:
            raise ValueError("getPixels returned a non-structured array for multiple bands")
        return (array.astype("float32", copy=False),)
    return tuple(np.asarray(array[name], dtype="float32") for name in band_ids)


def _structured_to_2d(raw: Any, *, band: str) -> np.ndarray:
    array = np.asarray(raw)
    if array.dtype.names is None:
        return array.astype("float32", copy=False)
    return np.asarray(array[band], dtype="float32")


def _coarse_grid(*, grid: GridSpec, scout_factor: int) -> GridSpec:
    coarse_resolution = float(grid.resolution) * int(scout_factor)
    coarse_width = max(1, int(math.ceil(grid.width / scout_factor)))
    coarse_height = max(1, int(math.ceil(grid.height / scout_factor)))
    xmin, _ymin, _xmax, ymax = grid.bounds
    return GridSpec(
        bounds=(xmin, ymax - coarse_height * coarse_resolution,
                xmin + coarse_width * coarse_resolution, ymax),
        crs=grid.crs,
        resolution=coarse_resolution,
        width=coarse_width,
        height=coarse_height,
        wgs84_bounds=grid.wgs84_bounds,
    )


def _filter_scenes_by_datetime(
    scenes: Sequence[S2L1CScene], start: str, end: str
) -> Tuple[S2L1CScene, ...]:
    from datetime import date as _date

    start_prefix, end_prefix = str(start)[:10], str(end)[:10]
    out = [
        s for s in scenes
        if start_prefix <= _date.fromtimestamp(s.timestamp_ms / 1000.0).isoformat() <= end_prefix
    ]
    return tuple(out)


def _ensure_initialized(ee: Any) -> None:
    try:
        ee.Number(1).getInfo()
        return
    except Exception:
        pass
    service_account = os.environ.get("GEE_SERVICE_ACCOUNT")
    key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if service_account and key_path:
        ee.Initialize(ee.ServiceAccountCredentials(service_account, key_path))
        return
    ee.Initialize()


def _utm_crs_from_wgs84_bounds(wgs84_bounds: Sequence[float]) -> str:
    west, south, east, north = (float(v) for v in wgs84_bounds)
    centre_lon = (west + east) / 2.0
    centre_lat = (south + north) / 2.0
    zone = max(1, min(60, int(math.floor((centre_lon + 180.0) / 6.0)) + 1))
    epsg = 32600 + zone if centre_lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _transform_wgs84_to_crs(
    wgs84_bounds: Sequence[float], dst_crs: str
) -> Tuple[float, float, float, float]:
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("S2L1CGeeSource requires pyproj for grid alignment.") from exc
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    return transformer.transform_bounds(*[float(v) for v in wgs84_bounds], densify_pts=21)


def _snap_bounds_to_resolution(
    bounds: Sequence[float], *, resolution: float
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    return (
        math.floor(xmin / resolution) * resolution,
        math.floor(ymin / resolution) * resolution,
        math.ceil(xmax / resolution) * resolution,
        math.ceil(ymax / resolution) * resolution,
    )
