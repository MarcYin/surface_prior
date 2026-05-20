"""Google Earth Engine source for Sentinel-2 L2A with Cloud Score+ quality.

Uses `ee.data.computePixels` directly so we can fetch a coarse cloud raster
for the AOI per scene (scout) and then materialise only the chunks the
selection plan asks for (fetch). The S2 surface reflectance band and the
Cloud Score+ `cs` band are joined per image via `linkCollection`, so each
chunk fetch returns SR + quality in one round-trip.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from surface_priors.chunks import ChunkLayout, ChunkWindow, chunk_grid
from surface_priors.selection import SceneChunkStats, SelectionPlan
from surface_priors.sources.s2 import (
    CLOUD_SCORE_PLUS_COLLECTION_ID,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CLEAR_THRESHOLD,
    DEFAULT_SCORE_BAND,
    DEFAULT_SCOUT_FACTOR,
    S2_L2A_BAND_MAP,
    S2_L2A_COLLECTION_ID,
    aggregate_chunk_stats,
    apply_zero_as_nodata,
    cloud_score_to_quality,
    cloud_score_valid_mask,
)
from surface_priors.temporal import sample_temporal_ranges, temporal_ranges_name
from surface_priors.types import GridSpec, Observation

S2_SR_SCALE = 0.0001


@dataclass(frozen=True)
class S2Scene:
    """One candidate Sentinel-2 L2A scene returned by the listing step."""

    scene_index: int
    s2_image_id: str
    cs_image_id: str
    system_index: str
    timestamp_ms: int

    @property
    def short_id(self) -> str:
        return self.system_index


class S2L2AGeeSource:
    """Chunked Sentinel-2 L2A source over Google Earth Engine + Cloud Score+.

    Authentication, image listing, scout, and fetch all run lazily so the
    module can be imported without earthengine-api configured. Sources can
    be instantiated in tests by passing an `ee_module` override.
    """

    def __init__(
        self,
        *,
        temporal_ranges: Sequence[Tuple[str, str]],
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
            raise ValueError("S2L2AGeeSource requires explicit temporal_ranges")
        if scout_factor <= 0:
            raise ValueError("scout_factor must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.temporal_ranges = tuple((str(start), str(end)) for start, end in temporal_ranges)
        self.sample_every_days = None if sample_every_days is None else int(sample_every_days)
        self.query_temporal_ranges = sample_temporal_ranges(
            self.temporal_ranges,
            sample_every_days=self.sample_every_days,
        )
        self.score_band = str(score_band)
        self.clear_threshold = float(clear_threshold)
        self.scout_factor = int(scout_factor)
        self.chunk_size = int(chunk_size)
        self.max_scenes = None if max_scenes is None else int(max_scenes)
        self._ee_module = ee_module
        temporal_key = temporal_ranges_name(
            self.temporal_ranges,
            sample_every_days=self.sample_every_days,
        )
        self._name = name or (
            f"s2-gee:cs:{self.score_band}:{self.scout_factor}x:chunk{self.chunk_size}:{temporal_key}"
        )
        self._scenes: Optional[Tuple[S2Scene, ...]] = None
        self._scene_cache_key: Optional[tuple] = None

    @property
    def name(self) -> str:
        return self._name

    def block_size(
        self,
        *,
        grid: GridSpec,
        band_names: Sequence[str],
    ) -> Optional[int]:
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
            bounds=snapped,
            crs=utm_crs,
            resolution=resolution,
            wgs84_bounds=wgs84_bounds,
        )

    def list_scenes(self, *, grid: GridSpec) -> Tuple[S2Scene, ...]:
        if grid.wgs84_bounds is None:
            raise ValueError("S2L2AGeeSource requires GridSpec with WGS84 bounds")
        key = (
            tuple(float(value) for value in grid.wgs84_bounds),
            self.query_temporal_ranges,
            self.max_scenes,
        )
        if self._scenes is not None and self._scene_cache_key == key:
            return self._scenes
        ee = self._ee()
        bbox = ee.Geometry.BBox(*grid.wgs84_bounds)
        items: list[dict[str, Any]] = []
        for start, end in self.query_temporal_ranges:
            collection = (
                ee.ImageCollection(S2_L2A_COLLECTION_ID)
                .filterBounds(bbox)
                .filterDate(start, end)
                .sort("system:time_start")
            )
            info = collection.aggregate_array("system:index").getInfo() or []
            times = collection.aggregate_array("system:time_start").getInfo() or []
            for system_index, ts in zip(info, times):
                items.append({"system_index": str(system_index), "timestamp_ms": int(ts)})
        items.sort(key=lambda entry: entry["timestamp_ms"])
        if self.max_scenes is not None:
            items = items[: self.max_scenes]
        scenes = tuple(
            S2Scene(
                scene_index=index,
                s2_image_id=f"{S2_L2A_COLLECTION_ID}/{item['system_index']}",
                cs_image_id=f"{CLOUD_SCORE_PLUS_COLLECTION_ID}/{item['system_index']}",
                system_index=item["system_index"],
                timestamp_ms=item["timestamp_ms"],
            )
            for index, item in enumerate(items)
        )
        self._scenes = scenes
        self._scene_cache_key = key
        return scenes

    def scout(
        self,
        *,
        grid: GridSpec,
        layout: ChunkLayout,
        band_names: Sequence[str],
    ) -> Sequence[SceneChunkStats]:
        del band_names
        scenes = self.list_scenes(grid=grid)
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
        scene = _find_scene(scenes, scene_index)
        if scene is None:
            return None
        window = plan.layout[chunk_id]
        return self._fetch_chunk(
            grid=grid,
            window=window,
            scene=scene,
            band_names=tuple(str(band) for band in band_names),
        )

    def _fetch_coarse_score(
        self,
        *,
        ee: Any,
        scene: S2Scene,
        coarse_grid: GridSpec,
        scale: float,
    ) -> Optional[np.ndarray]:
        cs_image = ee.Image(scene.cs_image_id).select(self.score_band)
        image = cs_image.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=4096,
            bestEffort=True,
        ).reproject(crs=coarse_grid.crs, scale=scale)
        request = _build_pixels_request(
            grid=coarse_grid,
            band_ids=(self.score_band,),
            expression=image,
        )
        try:
            raw = ee.data.computePixels(request)
        except Exception:
            return None
        return _structured_to_2d(raw, band=self.score_band)

    def _fetch_chunk(
        self,
        *,
        grid: GridSpec,
        window: ChunkWindow,
        scene: S2Scene,
        band_names: Sequence[str],
    ) -> Optional[Observation]:
        ee = self._ee()
        sub_grid = chunk_grid(grid, window)
        sr_band_ids = tuple(S2_L2A_BAND_MAP[band] for band in band_names)
        s2_image = ee.Image(scene.s2_image_id)
        cs_image = ee.Image(scene.cs_image_id).select(self.score_band)
        combined = s2_image.addBands(cs_image)
        all_band_ids = sr_band_ids + (self.score_band,)
        request = _build_pixels_request(
            grid=sub_grid,
            band_ids=all_band_ids,
            expression=combined,
        )
        try:
            raw = ee.data.computePixels(request)
        except Exception:
            return None
        arrays = _structured_to_band_stack(raw, band_ids=all_band_ids)
        sr_arrays = arrays[: len(sr_band_ids)]
        score_array = arrays[-1]
        data = np.stack(
            [apply_zero_as_nodata(arr * S2_SR_SCALE) for arr in sr_arrays],
            axis=0,
        ).astype("float32", copy=False)
        quality = cloud_score_to_quality(score_array, clear_threshold=self.clear_threshold)
        return Observation(
            data=data,
            quality=quality,
            band_names=band_names,
            source_id=scene.system_index,
            metadata={
                "s2_image_id": scene.s2_image_id,
                "cs_image_id": scene.cs_image_id,
                "system_index": scene.system_index,
                "timestamp_ms": scene.timestamp_ms,
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
                "S2L2AGeeSource requires the 'gee' extra: "
                "pip install 'surface-priors[gee]'"
            ) from exc
        _ensure_initialized(ee)
        return ee


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


def _find_scene(scenes: Sequence[S2Scene], scene_index: int) -> Optional[S2Scene]:
    for scene in scenes:
        if scene.scene_index == scene_index:
            return scene
    return None


def _coarse_grid(*, grid: GridSpec, scout_factor: int) -> GridSpec:
    coarse_resolution = float(grid.resolution) * int(scout_factor)
    coarse_width = max(1, int(math.ceil(grid.width / scout_factor)))
    coarse_height = max(1, int(math.ceil(grid.height / scout_factor)))
    xmin, _ymin, _xmax, ymax = grid.bounds
    coarse_xmax = xmin + coarse_width * coarse_resolution
    coarse_ymin = ymax - coarse_height * coarse_resolution
    return GridSpec(
        bounds=(xmin, coarse_ymin, coarse_xmax, ymax),
        crs=grid.crs,
        resolution=coarse_resolution,
        width=coarse_width,
        height=coarse_height,
        wgs84_bounds=grid.wgs84_bounds,
    )


def _build_pixels_request(
    *,
    grid: GridSpec,
    band_ids: Sequence[str],
    expression: Any,
) -> Mapping[str, Any]:
    xmin, _ymin, _xmax, ymax = grid.bounds
    return {
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": list(band_ids),
        "expression": expression,
        "grid": {
            "dimensions": {"width": int(grid.width), "height": int(grid.height)},
            "crsCode": grid.crs,
            "affineTransform": {
                "scaleX": float(grid.resolution),
                "shearX": 0.0,
                "translateX": float(xmin),
                "shearY": 0.0,
                "scaleY": -float(grid.resolution),
                "translateY": float(ymax),
            },
        },
    }


def _structured_to_band_stack(
    raw: Any,
    *,
    band_ids: Sequence[str],
) -> Tuple[np.ndarray, ...]:
    array = np.asarray(raw)
    if array.dtype.names is None:
        if len(band_ids) != 1:
            raise ValueError("computePixels returned a non-structured array for multiple bands")
        return (array.astype("float32", copy=False),)
    return tuple(np.asarray(array[name], dtype="float32") for name in band_ids)


def _structured_to_2d(raw: Any, *, band: str) -> np.ndarray:
    array = np.asarray(raw)
    if array.dtype.names is None:
        return array.astype("float32", copy=False)
    return np.asarray(array[band], dtype="float32")


def _utm_crs_from_wgs84_bounds(wgs84_bounds: Sequence[float]) -> str:
    west, south, east, north = (float(value) for value in wgs84_bounds)
    centre_lon = (west + east) / 2.0
    centre_lat = (south + north) / 2.0
    zone = int(math.floor((centre_lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    epsg = 32600 + zone if centre_lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _transform_wgs84_to_crs(
    wgs84_bounds: Sequence[float],
    dst_crs: str,
) -> Tuple[float, float, float, float]:
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("S2L2AGeeSource requires pyproj for grid alignment.") from exc
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    return transformer.transform_bounds(*[float(value) for value in wgs84_bounds], densify_pts=21)


def _snap_bounds_to_resolution(
    bounds: Sequence[float],
    *,
    resolution: float,
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    snapped_xmin = math.floor(xmin / resolution) * resolution
    snapped_ymin = math.floor(ymin / resolution) * resolution
    snapped_xmax = math.ceil(xmax / resolution) * resolution
    snapped_ymax = math.ceil(ymax / resolution) * resolution
    return snapped_xmin, snapped_ymin, snapped_xmax, snapped_ymax
