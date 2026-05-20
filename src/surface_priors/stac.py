from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from surface_priors.types import (
    PROJECTION_EXTENSION,
    RASTER_EXTENSION,
    SCHEMA_VERSION,
    STAC_VERSION,
    GridSpec,
    PriorComposite,
    utc_now_iso,
)


def build_stac_item(
    *,
    composite: PriorComposite,
    request_hash: str,
    prior_hrefs: Sequence[str],
    uncertainty_hrefs: Sequence[str],
    created_at: Optional[str] = None,
    composite_period: Optional[str] = None,
) -> Dict[str, Any]:
    if len(prior_hrefs) != len(composite.band_names):
        raise ValueError("prior_hrefs length must match composite band count")
    if uncertainty_hrefs and len(uncertainty_hrefs) != len(composite.band_names):
        raise ValueError(
            "uncertainty_hrefs must be empty or match composite band count"
        )
    grid = composite.grid
    if composite_period is None:
        composite_period = composite.attrs.get("composite_period")
    geometry, bbox = _wgs84_geometry_and_bbox(grid)
    properties: Dict[str, Any] = {
        "datetime": None,
        "created": created_at or utc_now_iso(),
        "surface:request_hash": request_hash,
        "surface:schema_version": SCHEMA_VERSION,
        "surface:prior_type": composite.attrs.get("prior_type", "brdf"),
        "surface:asset_layout": "single-band-geotiff-per-band",
        "surface:band_names": list(composite.band_names),
        "surface:compositor": composite.attrs.get("compositor", "best_pixel_v2"),
        "surface:source_count": len(composite.source_items),
    }
    if composite_period is not None:
        properties["surface:composite_period"] = str(composite_period)
    item: Dict[str, Any] = {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [PROJECTION_EXTENSION, RASTER_EXTENSION],
        "id": composite.product_id,
        "geometry": geometry,
        "properties": properties,
        "links": [],
        "assets": _band_assets(
            band_names=composite.band_names,
            prior_hrefs=prior_hrefs,
            uncertainty_hrefs=uncertainty_hrefs,
            composite_period=composite_period,
        ),
        "proj:shape": [grid.height, grid.width],
        "proj:transform": list(grid.transform_tuple),
        "proj:bbox": list(grid.bounds),
        "proj:wkt2": grid.crs,
    }
    if bbox is not None:
        item["bbox"] = bbox
    return item


def normalize_href(path: str | Path, root: str | Path) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.as_uri()


def safe_asset_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in str(value)
    )
    token = token.strip("._-")
    return token or "value"


def asset_stem(band_name: str) -> str:
    return safe_asset_token(band_name)


def asset_period_stem(composite_period: str) -> str:
    return safe_asset_token(composite_period)


def _band_assets(
    *,
    band_names: Sequence[str],
    prior_hrefs: Sequence[str],
    uncertainty_hrefs: Sequence[str],
    composite_period: Optional[str],
) -> Dict[str, Any]:
    assets: Dict[str, Any] = {}
    include_uncertainty = bool(uncertainty_hrefs)
    for index, band_name in enumerate(band_names):
        key_suffix = asset_stem(band_name)
        assets[f"prior_{key_suffix}"] = _prior_asset(
            prior_hrefs[index],
            band_name=band_name,
            band_index=index,
            composite_period=composite_period,
        )
        if include_uncertainty:
            assets[f"uncertainty_{key_suffix}"] = _uncertainty_asset(
                uncertainty_hrefs[index],
                band_name=band_name,
                band_index=index,
                composite_period=composite_period,
            )
    return assets


def _prior_asset(
    href: str,
    *,
    band_name: str,
    band_index: int,
    composite_period: Optional[str],
) -> Dict[str, Any]:
    asset: Dict[str, Any] = {
        "href": href,
        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
        "title": f"Scaled surface prior: {band_name}",
        "roles": ["data"],
        "surface:asset_kind": "prior",
        "surface:band_name": band_name,
        "surface:band_index": band_index,
        "raster:bands": [
            {
                "name": band_name,
                "data_type": "uint16",
                "scale": 0.0001,
                "nodata": 65535,
            }
        ],
    }
    if composite_period is not None:
        asset["surface:composite_period"] = str(composite_period)
    return asset


def _uncertainty_asset(
    href: str,
    *,
    band_name: str,
    band_index: int,
    composite_period: Optional[str],
) -> Dict[str, Any]:
    asset: Dict[str, Any] = {
        "href": href,
        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
        "title": f"Relative surface prior uncertainty: {band_name}",
        "roles": ["metadata", "uncertainty"],
        "surface:asset_kind": "uncertainty",
        "surface:band_name": band_name,
        "surface:band_index": band_index,
        "raster:bands": [
            {
                "name": f"{band_name}_relative_uncertainty",
                "data_type": "uint8",
                "unit": "percent",
                "nodata": 255,
                "statistics": {"minimum": 0, "maximum": 200},
            }
        ],
    }
    if composite_period is not None:
        asset["surface:composite_period"] = str(composite_period)
    return asset


def _wgs84_geometry_and_bbox(grid: GridSpec) -> tuple[Optional[Mapping[str, Any]], Optional[list[float]]]:
    if grid.wgs84_bounds is not None:
        west, south, east, north = grid.wgs84_bounds
        bbox = [float(west), float(south), float(east), float(north)]
        return _bbox_geometry(bbox), bbox

    try:
        from pyproj import Transformer
    except ImportError:
        return None, None

    try:
        transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        west, south, east, north = transformer.transform_bounds(*grid.bounds, densify_pts=21)
    except Exception:
        return None, None

    bbox = [float(west), float(south), float(east), float(north)]
    return _bbox_geometry(bbox), bbox


def _bbox_geometry(bbox: Sequence[float]) -> Mapping[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]
        ],
    }
