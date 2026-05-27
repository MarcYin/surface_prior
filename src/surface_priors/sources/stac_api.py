"""STAC-API surface reflectance source for Sentinel-2 L2A.

Targets STAC catalogues whose items expose per-band Cloud Optimized GeoTIFFs:
Element84 earth-search, Microsoft Planetary Computer, and Copernicus Data
Space Ecosystem (CDSE). The source uses rasterio's overview-aware `out_shape`
read to compute per-chunk clear-pixel statistics cheaply, then materialises
only the chunks selected by the caller via windowed reads against signed
asset hrefs.

Asset href signing varies by endpoint (anonymous, planetary-computer SDK,
CDSE bearer token) and plugs in through `AssetUrlSigner`.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from surface_priors.chunks import ChunkLayout, chunk_grid
from surface_priors.selection import SceneChunkStats, SelectionPlan
from surface_priors.sources.s2 import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SCOUT_FACTOR,
    aggregate_chunk_stats,
    apply_zero_as_nodata,
    cloud_score_to_quality,
    scl_clear_score,
    scl_to_quality,
    scl_valid_mask,
)
from surface_priors.sources.stac_cache import StacDiskCache, scenes_signature
from surface_priors.temporal import sample_temporal_ranges, temporal_ranges_name
from surface_priors.tile_classification import (
    ChunkTileRequirement,
    TilePartition,
    build_partition,
)
from surface_priors.types import GridSpec, Observation

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

S2_L2A_SR_SCALE = 0.0001

# GDAL options that materially reduce HTTP round-trips when opening remote
# COGs. These are safe for any /vsicurl/ HTTPS source; per-endpoint constructors
# layer extra options on top (e.g. AWS_NO_SIGN_REQUEST for Element84).
DEFAULT_GDAL_ENV: Mapping[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "VSI_CACHE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
}

EARTH_SEARCH_GDAL_ENV: Mapping[str, str] = {**DEFAULT_GDAL_ENV, "AWS_NO_SIGN_REQUEST": "YES"}


class AssetUrlSigner(Protocol):
    """Transforms a STAC item dict so its asset hrefs are GDAL-readable."""

    def sign_item(self, item: Mapping[str, Any]) -> Mapping[str, Any]: ...


class NoOpSigner:
    """Used by anonymous catalogues such as Element84 earth-search."""

    def sign_item(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        return item


class PlanetaryComputerSigner:
    """Sign Planetary Computer item asset hrefs via the planetary-computer SDK.

    The SDK is optional; if it is not installed the signer raises a clear
    ImportError when first used. Tests can substitute by passing a custom
    callable that mirrors `planetary_computer.sign`.
    """

    def __init__(self, *, subscription_key: Optional[str] = None) -> None:
        self.subscription_key = subscription_key

    def sign_item(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            import planetary_computer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PlanetaryComputerSigner requires the 'planetary-computer' package."
            ) from exc
        if self.subscription_key:
            planetary_computer.settings.set_subscription_key(self.subscription_key)
        return planetary_computer.sign(item)


class CdseTokenSigner:
    """Attach a CDSE bearer token to S3/HTTPS asset hrefs via GDAL options.

    The token is exposed as a header file that GDAL reads via
    `GDAL_HTTP_HEADER_FILE`. This signer only rewrites the item; callers
    must enter `cdse_env(...)` when reading rasters so GDAL sees the header.
    """

    def __init__(self, *, token: Optional[str] = None) -> None:
        self.token = token

    def sign_item(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        # CDSE asset hrefs are already https URLs; the token is applied at
        # GDAL read time, not by rewriting the dict.
        return item


@dataclass(frozen=True)
class StacAssetAliases:
    """Per-endpoint translation between surface_priors band names and STAC asset keys."""

    band_to_asset: Mapping[str, str]
    scl_asset: Optional[str] = None
    cs_asset: Optional[str] = None
    cs_subband: Optional[str] = None
    sr_scale: float = S2_L2A_SR_SCALE

    def quality_asset(self) -> str:
        if self.cs_asset is not None:
            return self.cs_asset
        if self.scl_asset is not None:
            return self.scl_asset
        raise ValueError("aliases declare neither cs_asset nor scl_asset")

    def uses_cloud_score_plus(self) -> bool:
        return self.cs_asset is not None


EARTH_SEARCH_S2_ALIASES = StacAssetAliases(
    band_to_asset={
        "s2_b01_aerosol": "coastal",
        "s2_b02_blue": "blue",
        "s2_b03_green": "green",
        "s2_b04_red": "red",
        "s2_b05_re1": "rededge1",
        "s2_b06_re2": "rededge2",
        "s2_b07_re3": "rededge3",
        "s2_b08_nir": "nir",
        "s2_b8a_nir_narrow": "nir08",
        "s2_b09_water_vapor": "nir09",
        "s2_b11_swir1": "swir16",
        "s2_b12_swir2": "swir22",
    },
    scl_asset="scl",
)


PLANETARY_COMPUTER_S2_ALIASES = StacAssetAliases(
    band_to_asset={
        "s2_b01_aerosol": "B01",
        "s2_b02_blue": "B02",
        "s2_b03_green": "B03",
        "s2_b04_red": "B04",
        "s2_b05_re1": "B05",
        "s2_b06_re2": "B06",
        "s2_b07_re3": "B07",
        "s2_b08_nir": "B08",
        "s2_b8a_nir_narrow": "B8A",
        "s2_b09_water_vapor": "B09",
        "s2_b11_swir1": "B11",
        "s2_b12_swir2": "B12",
    },
    scl_asset="SCL",
)


CDSE_S2_ALIASES = StacAssetAliases(
    band_to_asset={
        "s2_b01_aerosol": "B01_60m",
        "s2_b02_blue": "B02_10m",
        "s2_b03_green": "B03_10m",
        "s2_b04_red": "B04_10m",
        "s2_b05_re1": "B05_20m",
        "s2_b06_re2": "B06_20m",
        "s2_b07_re3": "B07_20m",
        "s2_b08_nir": "B08_10m",
        "s2_b8a_nir_narrow": "B8A_20m",
        "s2_b09_water_vapor": "B09_60m",
        "s2_b11_swir1": "B11_20m",
        "s2_b12_swir2": "B12_20m",
    },
    scl_asset="SCL_20m",
)


_MGRS_FROM_ITEM_ID = re.compile(r"S2[AB]_([0-9]{2}[A-Z]{3})_")


def _mgrs_tile_from_item(item_id: str, properties: Mapping[str, Any]) -> str:
    """Best-effort MGRS code extraction from item metadata.

    Earth-Search exposes ``grid:code`` (e.g., "MGRS-36RTU"); Planetary
    Computer and CDSE expose ``s2:mgrs_tile``. The Sentinel-2 item id
    encodes the tile as ``S2[AB]_<TILE>_…`` and is the most reliable
    fallback because it works across all three endpoints.
    """

    for key in ("s2:mgrs_tile", "mgrs:tile"):
        value = properties.get(key)
        if value:
            return str(value)
    grid_code = properties.get("grid:code")
    if isinstance(grid_code, str) and grid_code.upper().startswith("MGRS-"):
        return grid_code.split("-", 1)[1]
    match = _MGRS_FROM_ITEM_ID.match(str(item_id))
    if match:
        return match.group(1)
    return ""


@dataclass(frozen=True)
class StacScene:
    """A STAC item resolved to per-band hrefs and a stable scene_index."""

    scene_index: int
    item_id: str
    datetime: str
    asset_hrefs: Mapping[str, str]
    properties: Mapping[str, Any] = field(default_factory=dict)
    mgrs_tile: str = ""
    geometry: Optional[Mapping[str, Any]] = None


class StacApiSource:
    """Chunked surface reflectance source backed by a STAC-API catalogue."""

    def __init__(
        self,
        *,
        stac_url: str,
        collection: str,
        temporal_ranges: Sequence[Tuple[str, str]],
        aliases: StacAssetAliases,
        signer: Optional[AssetUrlSigner] = None,
        sample_every_days: Optional[int] = None,
        scout_factor: int = DEFAULT_SCOUT_FACTOR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_scenes: Optional[int] = None,
        scout_workers: int = 32,
        band_workers: int = 3,
        max_cloud_cover: Optional[float] = 90.0,
        disk_cache: Any = None,
        name: Optional[str] = None,
        stac_client: Any = None,
        opener: Any = None,
        gdal_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not temporal_ranges:
            raise ValueError("StacApiSource requires explicit temporal_ranges")
        if scout_factor <= 0:
            raise ValueError("scout_factor must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.stac_url = str(stac_url)
        self.collection = str(collection)
        self.aliases = aliases
        self.signer = signer or NoOpSigner()
        self.temporal_ranges = tuple((str(start), str(end)) for start, end in temporal_ranges)
        self.sample_every_days = None if sample_every_days is None else int(sample_every_days)
        self.query_temporal_ranges = sample_temporal_ranges(
            self.temporal_ranges,
            sample_every_days=self.sample_every_days,
        )
        self.scout_factor = int(scout_factor)
        self.chunk_size = int(chunk_size)
        self.max_scenes = None if max_scenes is None else int(max_scenes)
        self.scout_workers = max(1, int(scout_workers))
        self.band_workers = max(1, int(band_workers))
        self.max_cloud_cover = (
            None if max_cloud_cover is None else float(max_cloud_cover)
        )
        self.disk_cache = StacDiskCache.from_arg(disk_cache)
        self._stac_client = stac_client
        self._opener = opener  # injected rasterio.open replacement for tests
        self.gdal_env: Mapping[str, str] = dict(gdal_env) if gdal_env is not None else dict(
            DEFAULT_GDAL_ENV
        )
        temporal_key = temporal_ranges_name(
            self.temporal_ranges,
            sample_every_days=self.sample_every_days,
        )
        endpoint_key = self.stac_url.replace("https://", "").replace("/", "-")
        self._name = name or (
            f"stac:{endpoint_key}:{self.collection}:{self.scout_factor}x:chunk{self.chunk_size}:{temporal_key}"
        )
        self._scenes: Optional[Tuple[StacScene, ...]] = None
        self._scenes_key: Optional[tuple] = None
        self._raw_items: Tuple[Mapping[str, Any], ...] = ()
        self._partition_cache: dict[tuple, Optional[TilePartition]] = {}
        self._scout_cache: dict[tuple, dict[int, Tuple[SceneChunkStats, ...]]] = {}

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def earth_search_s2_l2a(
        cls,
        *,
        temporal_ranges: Sequence[Tuple[str, str]],
        **kwargs: Any,
    ) -> "StacApiSource":
        kwargs.setdefault("gdal_env", dict(EARTH_SEARCH_GDAL_ENV))
        return cls(
            stac_url=EARTH_SEARCH_URL,
            collection="sentinel-2-l2a",
            aliases=EARTH_SEARCH_S2_ALIASES,
            signer=NoOpSigner(),
            temporal_ranges=temporal_ranges,
            **kwargs,
        )

    @classmethod
    def planetary_computer_s2_l2a(
        cls,
        *,
        temporal_ranges: Sequence[Tuple[str, str]],
        subscription_key: Optional[str] = None,
        signer: Optional[AssetUrlSigner] = None,
        **kwargs: Any,
    ) -> "StacApiSource":
        return cls(
            stac_url=PLANETARY_COMPUTER_URL,
            collection="sentinel-2-l2a",
            aliases=PLANETARY_COMPUTER_S2_ALIASES,
            signer=signer or PlanetaryComputerSigner(subscription_key=subscription_key),
            temporal_ranges=temporal_ranges,
            **kwargs,
        )

    @classmethod
    def cdse_s2_l2a(
        cls,
        *,
        temporal_ranges: Sequence[Tuple[str, str]],
        token: Optional[str] = None,
        signer: Optional[AssetUrlSigner] = None,
        **kwargs: Any,
    ) -> "StacApiSource":
        return cls(
            stac_url=CDSE_STAC_URL,
            collection="SENTINEL-2",
            aliases=CDSE_S2_ALIASES,
            signer=signer or CdseTokenSigner(token=token),
            temporal_ranges=temporal_ranges,
            **kwargs,
        )

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

    def block_size(
        self,
        *,
        grid: GridSpec,
        band_names: Sequence[str],
    ) -> Optional[int]:
        del grid, band_names
        return None

    def tile_partition(
        self,
        *,
        grid: GridSpec,
        layout: ChunkLayout,
    ) -> Optional[TilePartition]:
        scenes = self.list_scenes(grid=grid)
        if not scenes:
            return None
        mem_key = (
            tuple(float(value) for value in grid.bounds),
            str(grid.crs),
            float(grid.resolution),
            int(grid.width),
            int(grid.height),
            int(layout.applied_chunk_size),
            tuple(window.chunk_id for window in layout),
        )
        cached = self._partition_cache.get(mem_key)
        if cached is not None or mem_key in self._partition_cache:
            return cached

        scene_tiles: dict[int, str] = {}
        scene_geometries: dict[int, Mapping[str, Any]] = {}
        for scene in scenes:
            if not scene.mgrs_tile or scene.geometry is None:
                continue
            scene_tiles[int(scene.scene_index)] = scene.mgrs_tile
            scene_geometries[int(scene.scene_index)] = scene.geometry
        if not scene_tiles:
            self._partition_cache[mem_key] = None
            return None

        disk_key: Optional[str] = None
        if self.disk_cache is not None:
            disk_key = self.disk_cache.partition_key(
                scenes_signature=scenes_signature(self._raw_items),
                grid_signature=(
                    tuple(float(v) for v in grid.bounds),
                    str(grid.crs),
                    float(grid.resolution),
                    int(grid.width),
                    int(grid.height),
                ),
                layout_signature=(int(layout.applied_chunk_size),),
            )
            payload = self.disk_cache.load_partition(disk_key)
            if payload is not None:
                partition = _partition_from_payload(payload)
                if partition is not None:
                    self._partition_cache[mem_key] = partition
                    return partition

        partition = build_partition(
            layout=layout,
            grid=grid,
            scene_tiles=scene_tiles,
            scene_geometries_wgs84=scene_geometries,
        )
        self._partition_cache[mem_key] = partition
        if partition is not None and self.disk_cache is not None and disk_key is not None:
            self.disk_cache.store_partition(disk_key, _partition_to_payload(partition))
        return partition

    def list_scenes(self, *, grid: GridSpec) -> Tuple[StacScene, ...]:
        if grid.wgs84_bounds is None:
            raise ValueError("StacApiSource requires GridSpec with WGS84 bounds")
        key = (
            tuple(float(value) for value in grid.wgs84_bounds),
            self.query_temporal_ranges,
            self.max_scenes,
        )
        if self._scenes is not None and self._scenes_key == key:
            return self._scenes
        raw_items = self._load_raw_items(grid=grid)
        scenes = self._build_scenes(raw_items)
        self._scenes = scenes
        self._scenes_key = key
        self._raw_items = raw_items
        self._partition_cache.clear()
        self._scout_cache.clear()
        return scenes

    def _load_raw_items(self, *, grid: GridSpec) -> Tuple[Mapping[str, Any], ...]:
        """Return unsigned STAC item dicts, hitting the disk cache when possible.

        Signing (Planetary Computer SAS tokens, CDSE bearer headers) is
        applied later in ``_build_scenes`` so cached items don't carry
        expired URLs.
        """

        raw_items: list[Mapping[str, Any]] = []
        cache = self.disk_cache
        for start, end in self.query_temporal_ranges:
            cache_key: Optional[str] = None
            if cache is not None:
                cache_key = cache.search_key(
                    stac_url=self.stac_url,
                    collection=self.collection,
                    wgs84_bounds=grid.wgs84_bounds,
                    datetime_range=(start, end),
                    max_cloud_cover=self.max_cloud_cover,
                )
                cached = cache.load_search(cache_key)
                if cached is not None:
                    raw_items.extend(cached)
                    continue
            range_items = self._search_range(grid=grid, start=start, end=end)
            raw_items.extend(range_items)
            if cache is not None and cache_key is not None:
                cache.store_search(cache_key, range_items)
        return tuple(raw_items)

    def _search_range(
        self,
        *,
        grid: GridSpec,
        start: str,
        end: str,
    ) -> list[Mapping[str, Any]]:
        client = self._client()
        search_kwargs: dict[str, Any] = {
            "collections": [self.collection],
            "bbox": list(grid.wgs84_bounds),
            "datetime": f"{start}/{end}",
        }
        if self.max_cloud_cover is not None and self.max_cloud_cover < 100.0:
            # Server-side cloud filter: drop scenes that are nothing-but-cloud
            # before they hit the wire. Trims list_scenes payload AND halves
            # scout reads. Threshold is intentionally high (default 90%)
            # because the producer's cloud_cover is a tile-level scalar and
            # an AOI may be clearer than its enclosing MGRS tile.
            search_kwargs["query"] = {
                "eo:cloud_cover": {"lt": float(self.max_cloud_cover)}
            }
        search = client.search(**search_kwargs)
        return [_item_to_dict(raw_item) for raw_item in _iter_items(search)]

    def _build_scenes(
        self,
        raw_items: Sequence[Mapping[str, Any]],
    ) -> Tuple[StacScene, ...]:
        signed_items: list[Mapping[str, Any]] = []
        for raw in raw_items:
            signed = self.signer.sign_item(raw)
            signed_items.append(signed)
        # Stable ordering on datetime then id so cache hits and live
        # searches yield identical scene_indices.
        signed_items.sort(
            key=lambda item: (
                str(item.get("properties", {}).get("datetime", "")),
                str(item.get("id", "")),
            )
        )
        if self.max_scenes is not None:
            signed_items = signed_items[: self.max_scenes]
        scenes: list[StacScene] = []
        for index, signed in enumerate(signed_items):
            asset_hrefs = {
                asset_key: asset["href"]
                for asset_key, asset in signed.get("assets", {}).items()
                if isinstance(asset, Mapping) and asset.get("href")
            }
            properties = dict(signed.get("properties", {}))
            item_id = str(signed.get("id", ""))
            raw_geom = signed.get("geometry")
            geometry = dict(raw_geom) if isinstance(raw_geom, Mapping) else None
            scenes.append(
                StacScene(
                    scene_index=index,
                    item_id=item_id,
                    datetime=str(properties.get("datetime", "")),
                    asset_hrefs=asset_hrefs,
                    properties=properties,
                    mgrs_tile=_mgrs_tile_from_item(item_id, properties),
                    geometry=geometry,
                )
            )
        return tuple(scenes)

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
        coarse_shape = (
            max(1, int(math.ceil(grid.height / self.scout_factor))),
            max(1, int(math.ceil(grid.width / self.scout_factor))),
        )

        # Per-scene scout output is a pure function of (grid, layout,
        # scout_factor). When the same source is reused for multiple
        # builds (e.g. monthly slices of one quarter), each scene's
        # stats are computed at most once.
        mem_cache_key = (
            tuple(float(value) for value in grid.bounds),
            str(grid.crs),
            float(grid.resolution),
            int(grid.width),
            int(grid.height),
            int(layout.applied_chunk_size),
            int(self.scout_factor),
        )
        cache = self._scout_cache.setdefault(mem_cache_key, {})

        grid_signature = (
            tuple(float(v) for v in grid.bounds),
            str(grid.crs),
            float(grid.resolution),
            int(grid.width),
            int(grid.height),
        )
        layout_signature = (int(layout.applied_chunk_size),)

        def scout_scene(scene: StacScene) -> Tuple[SceneChunkStats, ...]:
            score, valid = self._scout_quality(scene=scene, grid=grid, coarse_shape=coarse_shape)
            if score is None or valid is None:
                return tuple(
                    SceneChunkStats(
                        scene_index=scene.scene_index,
                        chunk_id=window.chunk_id,
                        usable_fraction=0.0,
                        mean_clear=float("nan"),
                    )
                    for window in layout
                )
            return aggregate_chunk_stats(
                scene_index=scene.scene_index,
                coarse_score=score,
                coarse_valid=valid,
                layout=layout,
                scout_factor=self.scout_factor,
            )

        def disk_key_for(scene: StacScene) -> Optional[str]:
            if self.disk_cache is None or not scene.item_id:
                return None
            return self.disk_cache.scout_key(
                stac_url=self.stac_url,
                collection=self.collection,
                item_id=scene.item_id,
                grid_signature=grid_signature,
                layout_signature=layout_signature,
                scout_factor=self.scout_factor,
            )

        to_scout: list[StacScene] = []
        for scene in scenes:
            if scene.scene_index in cache:
                continue
            disk_key = disk_key_for(scene)
            if disk_key is not None:
                disk_entry = self.disk_cache.load_scout(disk_key)
                if disk_entry is not None:
                    cache[scene.scene_index] = _scout_entry_from_payload(
                        disk_entry, scene_index=scene.scene_index
                    )
                    continue
            to_scout.append(scene)

        if to_scout:
            disk = self.disk_cache
            if self.scout_workers <= 1:
                for scene in to_scout:
                    entries = scout_scene(scene)
                    cache[scene.scene_index] = entries
                    if disk is not None:
                        disk_key = disk_key_for(scene)
                        if disk_key is not None:
                            disk.store_scout(disk_key, _scout_entry_to_payload(entries))
            else:
                with ThreadPoolExecutor(max_workers=self.scout_workers) as pool:
                    for scene, entries in zip(to_scout, pool.map(scout_scene, to_scout)):
                        cache[scene.scene_index] = entries
                        if disk is not None:
                            disk_key = disk_key_for(scene)
                            if disk_key is not None:
                                disk.store_scout(disk_key, _scout_entry_to_payload(entries))

        results: list[SceneChunkStats] = []
        for scene in scenes:
            results.extend(cache[scene.scene_index])
        return tuple(results)

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
        sub_grid = chunk_grid(grid, window)

        # Pre-validate that every requested SR band has an asset on this scene
        # so we can fail fast before launching reads.
        sr_assets = []
        for band in band_names:
            sr_asset = self.aliases.band_to_asset.get(band)
            if sr_asset is None or sr_asset not in scene.asset_hrefs:
                return None
            sr_assets.append(sr_asset)

        def read_sr(index_band):
            index, sr_asset = index_band
            arr = self._read_window(
                href=scene.asset_hrefs[sr_asset],
                grid=sub_grid,
                resample="bilinear",
            )
            if arr is None:
                return index, None
            arr = arr.astype("float32", copy=False) * float(self.aliases.sr_scale)
            return index, apply_zero_as_nodata(arr)

        def read_quality(_):
            return -1, self._read_quality_window(scene=scene, sub_grid=sub_grid)

        results: dict[int, Optional[np.ndarray]] = {}
        if self.band_workers <= 1:
            for index, sr_asset in enumerate(sr_assets):
                results[index] = read_sr((index, sr_asset))[1]
            results[-1] = self._read_quality_window(scene=scene, sub_grid=sub_grid)
        else:
            with ThreadPoolExecutor(max_workers=self.band_workers) as pool:
                futures = []
                for index, sr_asset in enumerate(sr_assets):
                    futures.append(pool.submit(read_sr, (index, sr_asset)))
                futures.append(pool.submit(read_quality, None))
                for future in futures:
                    index, arr = future.result()
                    results[index] = arr

        band_arrays = []
        for index in range(len(band_names)):
            arr = results.get(index)
            if arr is None:
                return None
            band_arrays.append(arr)
        quality = results.get(-1)
        if quality is None:
            return None

        data = np.stack(band_arrays, axis=0).astype("float32", copy=False)
        return Observation(
            data=data,
            quality=quality,
            band_names=tuple(str(band) for band in band_names),
            source_id=scene.item_id,
            metadata={
                "stac_collection": self.collection,
                "stac_url": self.stac_url,
                "datetime": scene.datetime,
                "chunk_id": int(window.chunk_id),
            },
        )

    def fetch_selected_for_scene(
        self,
        *,
        grid: GridSpec,
        plan: SelectionPlan,
        band_names: Sequence[str],
        scene_index: int,
        chunk_ids: Sequence[int],
    ) -> Mapping[int, Optional[Observation]]:
        """Open each band COG once and read all selected chunks for one scene.

        Tile-aware selection fans out multi-tile chunks across multiple
        scenes; the same scene often contributes to several chunks. Doing
        a per-(scene, chunk) ``fetch_selected`` opens each COG once per
        chunk. This method opens each band COG once per scene instead
        and reads every chunk window from the open dataset, eliminating
        redundant header GETs.
        """

        if not chunk_ids:
            return {}
        chunk_ids_int = [int(c) for c in chunk_ids]
        scenes = self.list_scenes(grid=grid)
        scene = _find_scene(scenes, scene_index)
        if scene is None:
            return dict.fromkeys(chunk_ids_int)

        sr_assets: list[str] = []
        for band in band_names:
            sr_asset = self.aliases.band_to_asset.get(band)
            if sr_asset is None or sr_asset not in scene.asset_hrefs:
                return dict.fromkeys(chunk_ids_int)
            sr_assets.append(sr_asset)

        quality_asset = self.aliases.quality_asset()
        quality_href = scene.asset_hrefs.get(quality_asset)
        if quality_href is None:
            return dict.fromkeys(chunk_ids_int)

        chunk_grids: Mapping[int, GridSpec] = {
            cid: chunk_grid(grid, plan.layout[cid]) for cid in chunk_ids_int
        }

        def read_asset_for_all_chunks(
            href: str,
            resample: str,
        ) -> Mapping[int, Optional[np.ndarray]]:
            return self._read_windows(href=href, grids=chunk_grids, resample=resample)

        sr_results: dict[str, Mapping[int, Optional[np.ndarray]]] = {}
        quality_results: Mapping[int, Optional[np.ndarray]] = {}
        if self.band_workers <= 1:
            for asset in sr_assets:
                sr_results[asset] = read_asset_for_all_chunks(scene.asset_hrefs[asset], "bilinear")
            quality_results = read_asset_for_all_chunks(quality_href, "nearest")
        else:
            with ThreadPoolExecutor(max_workers=self.band_workers) as pool:
                future_map: dict = {}
                for asset in sr_assets:
                    future = pool.submit(
                        read_asset_for_all_chunks,
                        scene.asset_hrefs[asset],
                        "bilinear",
                    )
                    future_map[future] = ("sr", asset)
                quality_future = pool.submit(
                    read_asset_for_all_chunks, quality_href, "nearest"
                )
                future_map[quality_future] = ("quality", None)
                for future, kind_asset in future_map.items():
                    kind, asset = kind_asset
                    result = future.result()
                    if kind == "sr":
                        sr_results[asset] = result
                    else:
                        quality_results = result

        out: dict[int, Optional[Observation]] = {}
        sr_scale = float(self.aliases.sr_scale)
        uses_cs_plus = self.aliases.uses_cloud_score_plus()
        for cid in chunk_ids_int:
            window = plan.layout[cid]
            band_arrays: list[np.ndarray] = []
            ok = True
            for asset in sr_assets:
                arr = sr_results.get(asset, {}).get(cid)
                if arr is None:
                    ok = False
                    break
                arr = arr.astype("float32", copy=False) * sr_scale
                band_arrays.append(apply_zero_as_nodata(arr))
            quality = quality_results.get(cid)
            if not ok or quality is None:
                out[cid] = None
                continue
            if uses_cs_plus:
                quality_array = cloud_score_to_quality(quality.astype("float32", copy=False))
            else:
                quality_array = scl_to_quality(quality.astype("int16", copy=False))
            data = np.stack(band_arrays, axis=0).astype("float32", copy=False)
            out[cid] = Observation(
                data=data,
                quality=quality_array,
                band_names=tuple(str(band) for band in band_names),
                source_id=scene.item_id,
                metadata={
                    "stac_collection": self.collection,
                    "stac_url": self.stac_url,
                    "datetime": scene.datetime,
                    "chunk_id": int(window.chunk_id),
                },
            )
        return out

    def _scout_quality(
        self,
        *,
        scene: StacScene,
        grid: GridSpec,
        coarse_shape: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        asset_key = self.aliases.quality_asset()
        href = scene.asset_hrefs.get(asset_key)
        if href is None:
            return None, None
        coarse_grid = _coarse_grid(grid=grid, coarse_shape=coarse_shape)
        raw = self._read_window(href=href, grid=coarse_grid, resample="nearest")
        if raw is None:
            return None, None
        if self.aliases.uses_cloud_score_plus():
            score = raw.astype("float32", copy=False)
            valid = np.isfinite(score) & (score >= 0.0) & (score <= 1.0)
            return score, valid
        scl = raw.astype("int16", copy=False)
        score = scl_clear_score(scl)
        valid = scl_valid_mask(scl)
        return score, valid

    def _read_quality_window(
        self,
        *,
        scene: StacScene,
        sub_grid: GridSpec,
    ) -> Optional[np.ndarray]:
        asset_key = self.aliases.quality_asset()
        href = scene.asset_hrefs.get(asset_key)
        if href is None:
            return None
        raw = self._read_window(href=href, grid=sub_grid, resample="nearest")
        if raw is None:
            return None
        if self.aliases.uses_cloud_score_plus():
            return cloud_score_to_quality(raw.astype("float32", copy=False))
        return scl_to_quality(raw.astype("int16", copy=False))

    def _read_window(
        self,
        *,
        href: str,
        grid: GridSpec,
        resample: str,
    ) -> Optional[np.ndarray]:
        opener = self._opener
        if opener is None:
            try:
                import rasterio  # type: ignore
            except ImportError as exc:
                raise ImportError("StacApiSource requires rasterio.") from exc
            opener = rasterio.open
        env = self._env_context()
        try:
            with env, opener(href) as dataset:
                return _read_to_grid(dataset, grid=grid, resample=resample)
        except (OSError, RuntimeError):
            return None

    def _read_windows(
        self,
        *,
        href: str,
        grids: Mapping[int, GridSpec],
        resample: str,
    ) -> Mapping[int, Optional[np.ndarray]]:
        """Open ``href`` once and return all requested chunk arrays.

        Reads a single window covering the union of every chunk's bounds,
        then slices the result per chunk. One HTTP range fetch covers all
        chunks for a (scene, band) instead of one per chunk — this is the
        big win for multi-chunk scenes under tile-aware selection.

        Falls back to per-chunk reads when the chunks share no common CRS
        or when the union read fails. Returned dict mirrors ``grids``
        keys; entries are ``None`` when the underlying read fails.
        """

        opener = self._opener
        if opener is None:
            try:
                import rasterio  # type: ignore
            except ImportError as exc:
                raise ImportError("StacApiSource requires rasterio.") from exc
            opener = rasterio.open
        env = self._env_context()
        out: dict[int, Optional[np.ndarray]] = dict.fromkeys(grids)
        if not grids:
            return out
        union = _union_grid(grids)
        try:
            with env, opener(href) as dataset:
                big = None
                if union is not None:
                    try:
                        big = _read_to_grid(dataset, grid=union, resample=resample)
                    except (OSError, RuntimeError):
                        big = None
                if big is not None:
                    for key, sub_grid in grids.items():
                        out[key] = _slice_from_union(big, union=union, sub_grid=sub_grid)
                else:
                    # Heterogeneous CRS or union failed — fall back per chunk.
                    for key, sub_grid in grids.items():
                        try:
                            out[key] = _read_to_grid(
                                dataset, grid=sub_grid, resample=resample
                            )
                        except (OSError, RuntimeError):
                            out[key] = None
        except (OSError, RuntimeError):
            return out
        return out

    def _env_context(self):
        if self._opener is not None or not self.gdal_env:
            from contextlib import nullcontext

            return nullcontext()
        try:
            import rasterio  # type: ignore
        except ImportError:
            from contextlib import nullcontext

            return nullcontext()
        return rasterio.Env(**self.gdal_env)

    def _client(self) -> Any:
        if self._stac_client is not None:
            return self._stac_client
        try:
            from pystac_client import Client  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "StacApiSource requires pystac-client: pip install pystac-client"
            ) from exc
        return Client.open(self.stac_url)


def _union_grid(grids: Mapping[int, GridSpec]) -> Optional[GridSpec]:
    """Smallest pixel-aligned bbox that contains every chunk grid.

    All chunks must share the same CRS and resolution (true for chunked
    layouts of a single grid). Returns ``None`` when that invariant is
    violated so callers can fall back to per-chunk reads.
    """

    items = list(grids.values())
    if not items:
        return None
    first = items[0]
    crs = first.crs
    res = float(first.resolution)
    for grid in items[1:]:
        if grid.crs != crs or float(grid.resolution) != res:
            return None
    xmin = min(grid.bounds[0] for grid in items)
    ymin = min(grid.bounds[1] for grid in items)
    xmax = max(grid.bounds[2] for grid in items)
    ymax = max(grid.bounds[3] for grid in items)
    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    if width <= 0 or height <= 0:
        return None
    return GridSpec(
        bounds=(xmin, ymin, xmax, ymax),
        crs=crs,
        resolution=res,
        width=width,
        height=height,
        wgs84_bounds=None,
    )


def _slice_from_union(
    big: np.ndarray,
    *,
    union: GridSpec,
    sub_grid: GridSpec,
) -> np.ndarray:
    """Carve a chunk's array out of the union read.

    Coordinate math is integer-pixel because both grids share resolution
    and CRS. ``big`` is shaped ``(union.height, union.width)``.
    """

    res = float(union.resolution)
    col_off = int(round((sub_grid.bounds[0] - union.bounds[0]) / res))
    row_off = int(round((union.bounds[3] - sub_grid.bounds[3]) / res))
    return big[
        row_off : row_off + int(sub_grid.height),
        col_off : col_off + int(sub_grid.width),
    ]


def _read_to_grid(dataset: Any, *, grid: GridSpec, resample: str) -> Optional[np.ndarray]:
    try:
        import rasterio  # type: ignore
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise ImportError("StacApiSource requires rasterio.") from exc

    resampling = Resampling.bilinear if resample == "bilinear" else Resampling.nearest
    dst = np.zeros((grid.height, grid.width), dtype=getattr(dataset, "dtypes", ("float32",))[0])
    target_crs = rasterio.crs.CRS.from_user_input(grid.crs)
    target_transform = rasterio.Affine(*grid.transform_tuple)
    source_crs = getattr(dataset, "crs", None)
    if source_crs is None:
        return None
    if source_crs.to_string() == target_crs.to_string():
        xmin, ymin, xmax, ymax = grid.bounds
        window = from_bounds(xmin, ymin, xmax, ymax, transform=dataset.transform)
        out_shape = (grid.height, grid.width)
        return dataset.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=resampling,
            boundless=True,
            fill_value=0,
        )
    src_array = dataset.read(1)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=dataset.transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling,
    )
    return dst


def _coarse_grid(*, grid: GridSpec, coarse_shape: Tuple[int, int]) -> GridSpec:
    coarse_height, coarse_width = coarse_shape
    coarse_resolution_x = (grid.bounds[2] - grid.bounds[0]) / coarse_width
    coarse_resolution_y = (grid.bounds[3] - grid.bounds[1]) / coarse_height
    resolution = max(coarse_resolution_x, coarse_resolution_y)
    xmin, _ymin, _xmax, ymax = grid.bounds
    xmax = xmin + coarse_width * resolution
    ymin = ymax - coarse_height * resolution
    return GridSpec(
        bounds=(xmin, ymin, xmax, ymax),
        crs=grid.crs,
        resolution=resolution,
        width=coarse_width,
        height=coarse_height,
        wgs84_bounds=grid.wgs84_bounds,
    )


def _iter_items(search: Any) -> Iterable[Any]:
    for attr in ("items", "get_items", "item_collection"):
        candidate = getattr(search, attr, None)
        if candidate is None:
            continue
        result = candidate() if callable(candidate) else candidate
        if hasattr(result, "items") and callable(result.items):
            yield from result.items
            return
        if hasattr(result, "__iter__"):
            yield from result
            return
    yield from search


def _item_to_dict(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return getattr(item, "__dict__", {})


def _find_scene(scenes: Sequence[StacScene], scene_index: int) -> Optional[StacScene]:
    for scene in scenes:
        if scene.scene_index == scene_index:
            return scene
    return None


def _filter_scenes_by_datetime(
    scenes: Sequence[StacScene],
    start: str,
    end: str,
) -> Tuple[StacScene, ...]:
    """Return scenes whose ISO datetime falls within [start, end] inclusive.

    `scene_index` is preserved so a SelectionPlan built from the filtered
    output still points at the right entry in the source's full cache when
    `fetch_selected` resolves it.
    """

    start_prefix = str(start)[:10]
    end_prefix = str(end)[:10]
    return tuple(
        scene
        for scene in scenes
        if start_prefix <= str(scene.datetime)[:10] <= end_prefix
    )


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
        raise ImportError("StacApiSource requires pyproj for grid alignment.") from exc
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    return transformer.transform_bounds(*[float(v) for v in wgs84_bounds], densify_pts=21)


def _scout_entry_to_payload(
    entries: Sequence[SceneChunkStats],
) -> list[Mapping[str, Any]]:
    """Serialise per-chunk stats for one scene; scene_index is dropped because
    the cache is keyed on item_id and rebuilt with the current source's
    scene_index on load."""

    return [
        {
            "chunk_id": int(entry.chunk_id),
            "usable_fraction": float(entry.usable_fraction),
            "mean_clear": float(entry.mean_clear),
        }
        for entry in entries
    ]


def _scout_entry_from_payload(
    payload: Sequence[Mapping[str, Any]],
    *,
    scene_index: int,
) -> Tuple[SceneChunkStats, ...]:
    return tuple(
        SceneChunkStats(
            scene_index=int(scene_index),
            chunk_id=int(item["chunk_id"]),
            usable_fraction=float(item["usable_fraction"]),
            mean_clear=float(item["mean_clear"]),
        )
        for item in payload
    )


def _partition_to_payload(partition: TilePartition) -> dict[str, Any]:
    return {
        "tiles": list(partition.tiles),
        "scene_to_tile": {str(k): v for k, v in partition.scene_to_tile.items()},
        "requirements": {
            str(chunk_id): {
                "chunk_id": int(req.chunk_id),
                "required_tiles": list(req.required_tiles),
                "unreachable_pixel_fraction": float(req.unreachable_pixel_fraction),
            }
            for chunk_id, req in partition.requirements.items()
        },
    }


def _partition_from_payload(payload: Mapping[str, Any]) -> Optional[TilePartition]:
    try:
        tiles = tuple(str(t) for t in payload.get("tiles", ()))
        scene_to_tile = {
            int(k): str(v) for k, v in payload.get("scene_to_tile", {}).items()
        }
        requirements = {}
        for chunk_id, req_dict in payload.get("requirements", {}).items():
            requirements[int(chunk_id)] = ChunkTileRequirement(
                chunk_id=int(req_dict["chunk_id"]),
                required_tiles=tuple(str(t) for t in req_dict["required_tiles"]),
                unreachable_pixel_fraction=float(req_dict["unreachable_pixel_fraction"]),
            )
    except (KeyError, TypeError, ValueError):
        return None
    return TilePartition(
        requirements=requirements,
        scene_to_tile=scene_to_tile,
        tiles=tiles,
    )


def _snap_bounds_to_resolution(
    bounds: Sequence[float],
    *,
    resolution: float,
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    return (
        math.floor(xmin / resolution) * resolution,
        math.floor(ymin / resolution) * resolution,
        math.ceil(xmax / resolution) * resolution,
        math.ceil(ymax / resolution) * resolution,
    )
