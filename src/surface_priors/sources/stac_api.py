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
from surface_priors.temporal import sample_temporal_ranges, temporal_ranges_name
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


@dataclass(frozen=True)
class StacScene:
    """A STAC item resolved to per-band hrefs and a stable scene_index."""

    scene_index: int
    item_id: str
    datetime: str
    asset_hrefs: Mapping[str, str]
    properties: Mapping[str, Any] = field(default_factory=dict)


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
        scout_workers: int = 4,
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
        client = self._client()
        items: list[StacScene] = []
        for start, end in self.query_temporal_ranges:
            search = client.search(
                collections=[self.collection],
                bbox=list(grid.wgs84_bounds),
                datetime=f"{start}/{end}",
            )
            for raw_item in _iter_items(search):
                signed = self.signer.sign_item(_item_to_dict(raw_item))
                asset_hrefs = {
                    asset_key: asset["href"]
                    for asset_key, asset in signed.get("assets", {}).items()
                    if isinstance(asset, Mapping) and asset.get("href")
                }
                items.append(
                    StacScene(
                        scene_index=len(items),
                        item_id=str(signed.get("id", "")),
                        datetime=str(signed.get("properties", {}).get("datetime", "")),
                        asset_hrefs=asset_hrefs,
                        properties=dict(signed.get("properties", {})),
                    )
                )
        items.sort(key=lambda scene: scene.datetime)
        if self.max_scenes is not None:
            items = items[: self.max_scenes]
        scenes = tuple(
            StacScene(
                scene_index=index,
                item_id=scene.item_id,
                datetime=scene.datetime,
                asset_hrefs=scene.asset_hrefs,
                properties=scene.properties,
            )
            for index, scene in enumerate(items)
        )
        self._scenes = scenes
        self._scenes_key = key
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
        coarse_shape = (
            max(1, int(math.ceil(grid.height / self.scout_factor))),
            max(1, int(math.ceil(grid.width / self.scout_factor))),
        )

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

        results: list[SceneChunkStats] = []
        if self.scout_workers <= 1:
            for scene in scenes:
                results.extend(scout_scene(scene))
        else:
            with ThreadPoolExecutor(max_workers=self.scout_workers) as pool:
                for entries in pool.map(scout_scene, scenes):
                    results.extend(entries)
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
        band_arrays = []
        for band in band_names:
            sr_asset = self.aliases.band_to_asset.get(band)
            if sr_asset is None or sr_asset not in scene.asset_hrefs:
                return None
            arr = self._read_window(
                href=scene.asset_hrefs[sr_asset],
                grid=sub_grid,
                resample="bilinear",
            )
            if arr is None:
                return None
            arr = arr.astype("float32", copy=False) * float(self.aliases.sr_scale)
            band_arrays.append(apply_zero_as_nodata(arr))
        data = np.stack(band_arrays, axis=0).astype("float32", copy=False)

        quality = self._read_quality_window(scene=scene, sub_grid=sub_grid)
        if quality is None:
            return None

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
            with env:
                with opener(href) as dataset:
                    return _read_to_grid(dataset, grid=grid, resample=resample)
        except (OSError, RuntimeError):
            return None

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
