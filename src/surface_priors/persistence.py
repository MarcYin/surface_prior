from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from surface_priors._version import __version__
from surface_priors.encoding import decode_prior, decode_relative_uncertainty
from surface_priors.geotiff import write_prior_band_geotiff, write_uncertainty_band_geotiff
from surface_priors.stac import asset_period_stem, asset_stem, build_stac_item, normalize_href
from surface_priors.types import (
    DEFAULT_PRIOR_NODATA,
    SCHEMA_VERSION,
    GridSpec,
    PriorComposite,
    PriorProduct,
    utc_now_iso,
)

STAC_ITEM_NAME = "stac-item.json"


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


class CompositeStore:
    """File-system store for STAC Item surface prior products."""

    DEFAULT_WRITE_WORKERS = 8

    def __init__(
        self,
        root: Union[str, Path],
        *,
        write_workers: Optional[int] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.write_workers = (
            int(write_workers) if write_workers is not None else self.DEFAULT_WRITE_WORKERS
        )

    def product_dir(self, request_hash: str) -> Path:
        return self.root / request_hash

    def has_product(self, request_hash: str) -> bool:
        stac_path = self.product_dir(request_hash) / STAC_ITEM_NAME
        if not stac_path.exists():
            return False
        try:
            with stac_path.open("r", encoding="utf-8") as handle:
                stac_item = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        return stac_item.get("properties", {}).get("surface:schema_version") == SCHEMA_VERSION

    def save(
        self,
        *,
        request_hash: str,
        request: Mapping[str, Any],
        composite: PriorComposite,
    ) -> PriorProduct:
        destination = self.product_dir(request_hash)
        assets_dir = destination / "assets"
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        composite_period = _composite_period(request, composite)
        write_uncertainty = bool(np.any(np.isfinite(composite.uncertainty)))
        prior_dir = _asset_kind_dir(assets_dir, "prior", composite_period)
        prior_dir.mkdir(parents=True, exist_ok=True)
        uncertainty_dir = None
        if write_uncertainty:
            uncertainty_dir = _asset_kind_dir(assets_dir, "uncertainty", composite_period)
            uncertainty_dir.mkdir(parents=True, exist_ok=True)
        _validate_unique_asset_stems(composite.band_names)
        if composite_period is not None and composite.attrs.get("composite_period") != composite_period:
            composite = replace(
                composite,
                attrs={**dict(composite.attrs), "composite_period": composite_period},
            )
        band_count = len(composite.band_names)
        prior_results: list[Optional[str]] = [None] * band_count
        uncertainty_results: list[Optional[str]] = [None] * band_count

        def write_prior(band_index: int) -> None:
            band_name = composite.band_names[band_index]
            filename = f"{asset_stem(band_name)}.tif"
            path = write_prior_band_geotiff(
                prior_dir / filename,
                composite=composite,
                band_index=band_index,
            )
            prior_results[band_index] = normalize_href(path, destination)

        def write_uncertainty_band(band_index: int) -> None:
            assert uncertainty_dir is not None
            band_name = composite.band_names[band_index]
            filename = f"{asset_stem(band_name)}.tif"
            path = write_uncertainty_band_geotiff(
                uncertainty_dir / filename,
                composite=composite,
                band_index=band_index,
            )
            uncertainty_results[band_index] = normalize_href(path, destination)

        write_jobs: list = [(write_prior, i) for i in range(band_count)]
        if write_uncertainty and uncertainty_dir is not None:
            write_jobs.extend((write_uncertainty_band, i) for i in range(band_count))

        if self.write_workers <= 1 or len(write_jobs) <= 1:
            for fn, idx in write_jobs:
                fn(idx)
        else:
            with ThreadPoolExecutor(max_workers=self.write_workers) as pool:
                list(pool.map(lambda job: job[0](job[1]), write_jobs))

        prior_hrefs = [href for href in prior_results if href is not None]
        uncertainty_hrefs = (
            [href for href in uncertainty_results if href is not None]
            if write_uncertainty
            else []
        )
        created_at = utc_now_iso()
        stac_item = build_stac_item(
            composite=composite,
            request_hash=request_hash,
            prior_hrefs=prior_hrefs,
            uncertainty_hrefs=uncertainty_hrefs,
            created_at=created_at,
            composite_period=composite_period,
        )
        stac_item["properties"]["surface:schema_version"] = SCHEMA_VERSION
        stac_item["properties"]["surface:package_version"] = __version__
        stac_item["properties"]["surface:source_items"] = list(composite.source_items)
        stac_item["properties"]["surface:attrs"] = dict(composite.attrs)
        with (destination / STAC_ITEM_NAME).open("w", encoding="utf-8") as handle:
            json.dump(stac_item, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return PriorProduct(
            request={**dict(request), "request_hash": request_hash},
            grid=composite.grid,
            composite=composite,
            stac_item=stac_item,
            output_dir=str(destination),
            created_at=created_at,
            package_version=__version__,
        )

    def load(self, request_hash: str, request: Optional[Mapping[str, Any]] = None) -> PriorProduct:
        source = self.product_dir(request_hash)
        stac_path = source / STAC_ITEM_NAME
        if not stac_path.exists():
            raise FileNotFoundError(f"no STAC item at {stac_path}")
        return load_product(source, request=request)


def load_product(path: Union[str, Path], request: Optional[Mapping[str, Any]] = None) -> PriorProduct:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Loading persisted GeoTIFF products requires rasterio.") from exc

    source = Path(path).expanduser().resolve()
    with (source / STAC_ITEM_NAME).open("r", encoding="utf-8") as handle:
        stac_item = json.load(handle)
    if stac_item["properties"].get("surface:schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "unsupported schema version "
            f"{stac_item['properties'].get('surface:schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )

    band_names = tuple(str(name) for name in stac_item["properties"].get("surface:band_names", ()))
    if not band_names:
        raise ValueError("STAC item does not declare surface:band_names")

    prior_encoded, grid = _read_single_band_stack(
        rasterio,
        source=source,
        stac_item=stac_item,
        kind="prior",
        band_names=band_names,
    )
    data = decode_prior(prior_encoded)
    if _stac_has_assets(stac_item, kind="uncertainty"):
        uncertainty_encoded, uncertainty_grid = _read_single_band_stack(
            rasterio,
            source=source,
            stac_item=stac_item,
            kind="uncertainty",
            band_names=band_names,
        )
        if uncertainty_grid.to_dict() != grid.to_dict():
            raise ValueError("uncertainty assets do not share the prior asset grid")
        uncertainty = decode_relative_uncertainty(uncertainty_encoded)
    else:
        uncertainty = np.full(data.shape, np.nan, dtype="float32")
    shape = grid.shape
    attrs = dict(stac_item["properties"].get("surface:attrs", {}))
    composite_period = stac_item["properties"].get("surface:composite_period")
    if composite_period is not None:
        attrs.setdefault("composite_period", composite_period)
    composite = PriorComposite(
        product_id=stac_item["id"],
        grid=grid,
        band_names=band_names,
        data=data,
        uncertainty=uncertainty,
        quality=np.full(shape, DEFAULT_PRIOR_NODATA, dtype="uint16"),
        sample_index=np.full(shape, -1, dtype="int16"),
        selected_observation=np.full(shape, -1, dtype="int16"),
        observation_count=np.zeros(shape, dtype="uint16"),
        source_items=stac_item["properties"].get("surface:source_items", ()),
        attrs=attrs,
    )
    request_payload = {} if request is None else dict(request)
    request_payload.setdefault("request_hash", stac_item["properties"]["surface:request_hash"])
    return PriorProduct(
        request=request_payload,
        grid=grid,
        composite=composite,
        stac_item=stac_item,
        output_dir=str(source),
        package_version=stac_item["properties"].get("surface:package_version", "unknown"),
    )


def _stac_has_assets(stac_item: Mapping[str, Any], *, kind: str) -> bool:
    for asset in stac_item.get("assets", {}).values():
        if asset.get("surface:asset_kind") == kind:
            return True
    return False


def _read_single_band_stack(
    rasterio_module: Any,
    *,
    source: Path,
    stac_item: Mapping[str, Any],
    kind: str,
    band_names: tuple[str, ...],
) -> tuple[np.ndarray, GridSpec]:
    arrays = []
    grid = None
    reference_signature = None
    for band_index, asset in enumerate(_ordered_band_assets(stac_item, kind, len(band_names))):
        href = asset["href"]
        with rasterio_module.open(_asset_href(source, href)) as dataset:
            if dataset.count != 1:
                raise ValueError(f"{href} contains {dataset.count} bands; expected one")
            asset_band_name = asset.get("surface:band_name")
            if asset_band_name != band_names[band_index]:
                raise ValueError(
                    f"{href} declares band {asset_band_name!r}; expected {band_names[band_index]!r}"
                )
            signature = _dataset_signature(dataset)
            if reference_signature is None:
                reference_signature = signature
                grid = GridSpec(
                    bounds=tuple(float(value) for value in stac_item["proj:bbox"]),
                    crs=dataset.crs.to_wkt() if dataset.crs else stac_item["proj:wkt2"],
                    resolution=abs(float(dataset.transform.a)),
                    width=dataset.width,
                    height=dataset.height,
                    wgs84_bounds=None
                    if stac_item.get("bbox") is None
                    else tuple(float(value) for value in stac_item["bbox"]),
                )
            elif signature != reference_signature:
                raise ValueError(f"{href} grid does not match the first {kind} asset")
            arrays.append(dataset.read(1))
    if grid is None:
        raise ValueError(f"STAC item has no {kind} assets")
    return np.stack(arrays, axis=0), grid


def _ordered_band_assets(
    stac_item: Mapping[str, Any],
    kind: str,
    band_count: int,
) -> list[Mapping[str, Any]]:
    assets_by_index: dict[int, Mapping[str, Any]] = {}
    for asset in stac_item.get("assets", {}).values():
        if asset.get("surface:asset_kind") != kind:
            continue
        band_index = asset.get("surface:band_index")
        if not isinstance(band_index, int) or band_index < 0 or band_index >= band_count:
            raise ValueError(f"{kind} asset has invalid surface:band_index {band_index!r}")
        if band_index in assets_by_index:
            raise ValueError(f"duplicate {kind} asset for surface:band_index {band_index}")
        assets_by_index[band_index] = asset
    missing = [index for index in range(band_count) if index not in assets_by_index]
    if missing:
        raise ValueError(f"STAC item is missing {kind} assets for band indices {missing}")
    return [assets_by_index[index] for index in range(band_count)]


def _asset_href(source: Path, href: str) -> str | Path:
    if "://" in href:
        return href
    return source / href


def _composite_period(request: Mapping[str, Any], composite: PriorComposite) -> Optional[str]:
    value = request.get("composite_period", composite.attrs.get("composite_period"))
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _asset_kind_dir(assets_dir: Path, kind: str, composite_period: Optional[str]) -> Path:
    directory = assets_dir / kind
    if composite_period is not None:
        directory = directory / asset_period_stem(composite_period)
    return directory


def _validate_unique_asset_stems(band_names: Any) -> None:
    stems: dict[str, str] = {}
    for band_name in band_names:
        stem = asset_stem(str(band_name))
        previous = stems.get(stem)
        if previous is not None:
            raise ValueError(
                f"band names {previous!r} and {band_name!r} both map to asset filename {stem!r}"
            )
        stems[stem] = str(band_name)


def _dataset_signature(dataset: Any) -> tuple[Any, ...]:
    transform = dataset.transform
    return (
        dataset.width,
        dataset.height,
        tuple(
            float(value)
            for value in (
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            )
        ),
        dataset.crs.to_wkt() if dataset.crs else None,
    )


def stac_item_path(path: Union[str, Path], request_hash: Optional[str] = None) -> Path:
    base = Path(path).expanduser().resolve()
    if request_hash is not None:
        base = base / request_hash
    return base / STAC_ITEM_NAME


manifest_path = stac_item_path
