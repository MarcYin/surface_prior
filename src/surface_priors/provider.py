from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from surface_priors.chunks import ChunkLayout
from surface_priors.composite import ChunkedCompositor, PriorCompositor
from surface_priors.persistence import CompositeStore, stable_json_hash
from surface_priors.selection import SelectionPlan, SelectionPolicy, select
from surface_priors.sources.base import ObservationSource
from surface_priors.types import (
    DEFAULT_BANDS,
    DEFAULT_NATIVE_CRS,
    GridSpec,
    Observation,
    PriorProduct,
)


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "surface-priors"


@dataclass
class ProviderConfig:
    """Configuration for a surface prior provider."""

    cache_dir: Union[str, Path] = field(default_factory=default_cache_dir)
    source: Optional[ObservationSource] = None
    source_name: Optional[str] = None
    compositor: PriorCompositor = field(default_factory=PriorCompositor)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    chunk_size: int = 512
    selection_policy: SelectionPolicy = field(default_factory=SelectionPolicy)
    fetch_workers: int = 8
    emit_uncertainty: bool = False


class Provider:
    """Build or retrieve a native-grid surface prior product."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()
        self.store = CompositeStore(self.config.cache_dir)

    def build_prior(
        self,
        *,
        wgs84_bounds: Sequence[float],
        resolution: float,
        product_id: str,
        native_crs: str = DEFAULT_NATIVE_CRS,
        brdf_crs: Optional[str] = None,
        band_names: Sequence[str] = DEFAULT_BANDS,
        composite_period: Optional[str] = None,
        observations: Optional[Sequence[Observation]] = None,
        rebuild: bool = False,
    ) -> PriorProduct:
        crs = native_crs if brdf_crs is None else brdf_crs
        band_names = tuple(str(band) for band in band_names)
        grid = self._grid_for_request(
            wgs84_bounds=wgs84_bounds,
            native_crs=crs,
            resolution=resolution,
            band_names=band_names,
        )
        request = self._request_payload(
            grid=grid,
            product_id=product_id,
            band_names=band_names,
            composite_period=composite_period,
        )
        request_hash = stable_json_hash(request)
        if not rebuild and self.store.has_product(request_hash):
            return self.store.load(request_hash, request={**request, "request_hash": request_hash})

        if observations is None and _is_chunked_source(self.config.source):
            composite = self._build_chunked(
                product_id=product_id,
                grid=grid,
                band_names=band_names,
            )
        else:
            if observations is None:
                if self.config.source is None:
                    stac_path = self.store.product_dir(request_hash) / "stac-item.json"
                    raise RuntimeError(
                        "cache miss and no observations or ObservationSource configured. "
                        f"Configure ProviderConfig(source=...), pass observations=..., or create {stac_path} first."
                    )
                observations = self.config.source.load_observations(
                    grid=grid, band_names=band_names
                )
            compositor = self._eager_compositor()
            composite = compositor.compose(
                product_id=product_id,
                grid=grid,
                band_names=band_names,
                observations=observations,
            )
        return self.store.save(
            request_hash=request_hash,
            request=request,
            composite=composite,
        )

    def _build_chunked(
        self,
        *,
        product_id: str,
        grid: GridSpec,
        band_names: Sequence[str],
    ):
        source = self.config.source
        assert source is not None
        block_size_fn = getattr(source, "block_size", None)
        block_size = None
        if callable(block_size_fn):
            block_size = block_size_fn(grid=grid, band_names=band_names)
        layout = ChunkLayout.from_grid(
            grid,
            chunk_size=self.config.chunk_size,
            block_size=block_size,
        )
        stats = source.scout(grid=grid, layout=layout, band_names=band_names)
        plan = select(
            layout=layout,
            stats=stats,
            policy=self.config.selection_policy,
        )
        cache = _prefetch_chunks(
            source=source,
            grid=grid,
            plan=plan,
            band_names=band_names,
            workers=self.config.fetch_workers,
        )

        def chunk_loader(scene_index: int, chunk_id: int) -> Optional[Observation]:
            return cache.get((int(scene_index), int(chunk_id)))

        compositor = ChunkedCompositor(
            quality_rules=self.config.compositor.quality_rules,
            output_dtype=self.config.compositor.output_dtype,
            emit_uncertainty=self._resolved_emit_uncertainty(),
        )
        return compositor.compose(
            product_id=product_id,
            grid=grid,
            band_names=band_names,
            plan=plan,
            chunk_loader=chunk_loader,
        )

    def request_hash(
        self,
        *,
        wgs84_bounds: Sequence[float],
        resolution: float,
        product_id: str,
        native_crs: str = DEFAULT_NATIVE_CRS,
        brdf_crs: Optional[str] = None,
        band_names: Sequence[str] = DEFAULT_BANDS,
        composite_period: Optional[str] = None,
    ) -> str:
        crs = native_crs if brdf_crs is None else brdf_crs
        band_names = tuple(str(band) for band in band_names)
        grid = self._grid_for_request(
            wgs84_bounds=wgs84_bounds,
            native_crs=crs,
            resolution=resolution,
            band_names=band_names,
        )
        return stable_json_hash(
            self._request_payload(
                grid=grid,
                product_id=product_id,
                band_names=band_names,
                composite_period=composite_period,
            )
        )

    def _grid_for_request(
        self,
        *,
        wgs84_bounds: Sequence[float],
        native_crs: str,
        resolution: float,
        band_names: Sequence[str],
    ) -> GridSpec:
        if self.config.source is not None:
            resolver = getattr(self.config.source, "resolve_grid", None)
            if callable(resolver):
                crs_key = _native_crs_parameter_name(resolver)
                return resolver(
                    wgs84_bounds=wgs84_bounds,
                    **{crs_key: native_crs},
                    resolution=resolution,
                    band_names=band_names,
                )
        return GridSpec.from_wgs84_bounds(
            wgs84_bounds=wgs84_bounds,
            native_crs=native_crs,
            resolution=resolution,
        )

    def _request_payload(
        self,
        *,
        grid: GridSpec,
        product_id: str,
        band_names: Sequence[str],
        composite_period: Optional[str],
    ) -> dict[str, Any]:
        source_name = self.config.source_name
        if source_name is None:
            source_name = "direct-observations" if self.config.source is None else self.config.source.name
        payload: dict[str, Any] = {
            "product_id": str(product_id),
            "wgs84_bounds": None if grid.wgs84_bounds is None else list(grid.wgs84_bounds),
            "native_bounds": list(grid.bounds),
            "native_crs": grid.crs,
            "resolution": grid.resolution,
            "width": grid.width,
            "height": grid.height,
            "band_names": list(band_names),
            "source": source_name,
            "provider_metadata": dict(self.config.metadata),
        }
        if composite_period is not None:
            payload["composite_period"] = str(composite_period)
        if _is_chunked_source(self.config.source):
            policy = self.config.selection_policy
            payload["chunking"] = {
                "chunk_size": int(self.config.chunk_size),
                "top_k": int(policy.top_k),
                "min_usable_fraction": float(policy.min_usable_fraction),
                "min_clear_score": float(policy.min_clear_score),
            }
        return payload

    get_prior = build_prior

    def _eager_compositor(self) -> PriorCompositor:
        base = self.config.compositor
        emit = self._resolved_emit_uncertainty()
        if base.emit_uncertainty == emit:
            return base
        return PriorCompositor(
            quality_rules=base.quality_rules,
            output_dtype=base.output_dtype,
            emit_uncertainty=emit,
        )

    def _resolved_emit_uncertainty(self) -> bool:
        # The compositor field's value wins when callers handed in a fully
        # configured compositor; otherwise the top-level ProviderConfig flag.
        compositor_default = PriorCompositor()
        if self.config.compositor.emit_uncertainty != compositor_default.emit_uncertainty:
            return self.config.compositor.emit_uncertainty
        return self.config.emit_uncertainty


def _is_chunked_source(source: Any) -> bool:
    if source is None:
        return False
    return callable(getattr(source, "scout", None)) and callable(
        getattr(source, "fetch_selected", None)
    )


def _prefetch_chunks(
    *,
    source: Any,
    grid: GridSpec,
    plan: SelectionPlan,
    band_names: Sequence[str],
    workers: int,
) -> dict[tuple[int, int], Optional[Observation]]:
    tasks: list[tuple[int, int]] = []
    for chunk_id, scenes in plan.selected.items():
        for scene_index in scenes:
            tasks.append((int(scene_index), int(chunk_id)))
    if not tasks:
        return {}

    def fetch(item: tuple[int, int]) -> tuple[tuple[int, int], Optional[Observation]]:
        scene_index, chunk_id = item
        return item, source.fetch_selected(
            grid=grid,
            plan=plan,
            band_names=band_names,
            scene_index=scene_index,
            chunk_id=chunk_id,
        )

    cache: dict[tuple[int, int], Optional[Observation]] = {}
    if workers <= 1:
        for task in tasks:
            key, value = fetch(task)
            cache[key] = value
        return cache
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        for key, value in pool.map(fetch, tasks):
            cache[key] = value
    return cache


def _native_crs_parameter_name(resolver: Any) -> str:
    """Choose the source grid resolver CRS keyword with legacy compatibility."""

    try:
        parameters = inspect.signature(resolver).parameters
    except (TypeError, ValueError):
        return "native_crs"
    if "native_crs" in parameters:
        return "native_crs"
    if "brdf_crs" in parameters:
        return "brdf_crs"
    return "native_crs"
