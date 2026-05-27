"""Geometry-only chunk-to-tile classification.

Sentinel-2 (and other tiled providers) deliver scenes that each cover only
one MGRS tile. When a user-requested chunk straddles two MGRS tiles, no
single scene can fill the whole chunk; the compositor must merge scenes
from each required tile. This module classifies chunks by which tile
footprints they need, independent of cloud cover or any scoring.

The classification is computed once per ``(grid, layout, scenes)`` from
the union of item geometries per MGRS code, then handed to selection so
that the per-chunk top-K is taken *per required tile* and unioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from surface_priors.chunks import ChunkLayout, chunk_grid
from surface_priors.types import GridSpec


@dataclass(frozen=True)
class ChunkTileRequirement:
    """Which MGRS tiles a chunk needs scenes from to be fully covered.

    ``required_tiles`` is ordered by exclusive coverage (most first). It
    contains every tile with non-zero exclusive coverage in the chunk, or
    a single fallback tile when the chunk lies entirely inside an
    overlap region. ``unreachable_pixel_fraction`` is the share of chunk
    pixels outside every tile's footprint; non-zero values indicate a
    structural data gap that no scene selection can fill.
    """

    chunk_id: int
    required_tiles: Tuple[str, ...]
    unreachable_pixel_fraction: float = 0.0


@dataclass(frozen=True)
class TilePartition:
    """Tile classification for one ``(grid, layout)`` pair.

    ``scene_to_tile`` maps each scene_index to the MGRS tile its asset
    hrefs belong to. ``requirements`` is keyed by chunk_id and lists the
    tiles each chunk needs. Sources with no tile concept return ``None``
    instead of a partition; selection then degenerates to a single
    synthetic tile.
    """

    requirements: Mapping[int, ChunkTileRequirement]
    scene_to_tile: Mapping[int, str]
    tiles: Tuple[str, ...] = field(default_factory=tuple)

    def tiles_for(self, chunk_id: int) -> Tuple[str, ...]:
        req = self.requirements.get(int(chunk_id))
        return () if req is None else req.required_tiles

    def tile_of(self, scene_index: int) -> Optional[str]:
        return self.scene_to_tile.get(int(scene_index))


def build_tile_footprints(
    *,
    scene_tiles: Mapping[int, str],
    scene_geometries_wgs84: Mapping[int, Mapping[str, Any]],
    grid_crs: str,
) -> Mapping[str, Any]:
    """Union per-tile item geometries and return them reprojected to ``grid_crs``.

    Scenes missing a tile assignment or geometry are skipped. The return
    value maps each MGRS code to a single Shapely geometry in the grid's
    CRS. Callers should treat shapely as an optional dep; this function
    raises ``ImportError`` if shapely is unavailable.
    """

    try:
        from shapely.geometry import shape
        from shapely.ops import transform as shapely_transform
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError(
            "tile classification requires shapely (transitively installed with rasterio)"
        ) from exc

    by_tile: dict[str, list[Any]] = {}
    for scene_index, tile_code in scene_tiles.items():
        geom_dict = scene_geometries_wgs84.get(scene_index)
        if not tile_code or geom_dict is None:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:  # noqa: BLE001 — defensive; bad GeoJSON shouldn't kill the build
            continue
        if geom.is_empty:
            continue
        by_tile.setdefault(tile_code, []).append(geom)

    if not by_tile:
        return {}

    reprojector = _wgs84_reprojector(grid_crs)
    footprints: dict[str, Any] = {}
    for tile_code, geoms in by_tile.items():
        merged = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        if merged.is_empty:
            continue
        if reprojector is None:
            footprints[tile_code] = merged
        else:
            footprints[tile_code] = shapely_transform(reprojector, merged)
    return footprints


def classify_chunks(
    *,
    layout: ChunkLayout,
    grid: GridSpec,
    tile_footprints: Mapping[str, Any],
    min_exclusive_pixels: int = 1,
) -> Mapping[int, ChunkTileRequirement]:
    """For each chunk, list the tiles whose data is needed to fill it.

    A tile is required if its *exclusive* coverage of the chunk (pixels
    covered by this tile but no other) is at least ``min_exclusive_pixels``.
    Chunks lying entirely inside a multi-tile overlap region fall back to
    the single tile with the largest total intersection, so single-tile
    chunks always remain single-pick.
    """

    try:
        from shapely.geometry import box
    except ImportError as exc:
        raise ImportError(
            "tile classification requires shapely (transitively installed with rasterio)"
        ) from exc

    if not tile_footprints:
        return {}

    pixel_area = float(grid.resolution) * float(grid.resolution)
    min_exclusive_area = max(0.0, float(min_exclusive_pixels)) * pixel_area
    tile_codes = tuple(sorted(tile_footprints))

    requirements: dict[int, ChunkTileRequirement] = {}
    for window in layout:
        sub_grid = chunk_grid(grid, window)
        xmin, ymin, xmax, ymax = sub_grid.bounds
        chunk_geom = box(xmin, ymin, xmax, ymax)
        chunk_area = chunk_geom.area

        intersections: dict[str, Any] = {}
        intersect_areas: dict[str, float] = {}
        for tile_code in tile_codes:
            inter = chunk_geom.intersection(tile_footprints[tile_code])
            if inter.is_empty or inter.area <= 0.0:
                # Skip degenerate line-of-contact intersections — they touch
                # the chunk's boundary but contain no usable pixels.
                continue
            intersections[tile_code] = inter
            intersect_areas[tile_code] = inter.area

        if not intersections:
            requirements[window.chunk_id] = ChunkTileRequirement(
                chunk_id=window.chunk_id,
                required_tiles=(),
                unreachable_pixel_fraction=1.0,
            )
            continue

        exclusive_areas: dict[str, float] = {}
        for tile_code, inter in intersections.items():
            other_inters = [
                intersections[other]
                for other in intersections
                if other != tile_code
            ]
            if other_inters:
                from shapely.ops import unary_union

                exclusive = inter.difference(unary_union(other_inters))
                exclusive_areas[tile_code] = exclusive.area
            else:
                exclusive_areas[tile_code] = inter.area

        required = [
            tile_code
            for tile_code, area in exclusive_areas.items()
            if area >= min_exclusive_area
        ]
        if not required:
            # Chunk sits entirely in an overlap region; one tile is enough.
            required = [max(intersect_areas, key=intersect_areas.get)]

        required.sort(key=lambda code: exclusive_areas.get(code, 0.0), reverse=True)

        from shapely.ops import unary_union

        reachable_area = unary_union(list(intersections.values())).area
        unreachable_fraction = max(0.0, 1.0 - (reachable_area / chunk_area)) if chunk_area > 0 else 0.0
        requirements[window.chunk_id] = ChunkTileRequirement(
            chunk_id=window.chunk_id,
            required_tiles=tuple(required),
            unreachable_pixel_fraction=float(unreachable_fraction),
        )

    return requirements


def build_partition(
    *,
    layout: ChunkLayout,
    grid: GridSpec,
    scene_tiles: Mapping[int, str],
    scene_geometries_wgs84: Mapping[int, Mapping[str, Any]],
    min_exclusive_pixels: int = 1,
) -> Optional[TilePartition]:
    """Convenience: footprints + classification + scene→tile map in one call.

    Returns ``None`` if no tile metadata was supplied — selection treats
    that as the legacy single-tile case.
    """

    tiles = {
        int(scene_index): str(tile_code)
        for scene_index, tile_code in scene_tiles.items()
        if tile_code
    }
    if not tiles:
        return None
    footprints = build_tile_footprints(
        scene_tiles=tiles,
        scene_geometries_wgs84=scene_geometries_wgs84,
        grid_crs=grid.crs,
    )
    if not footprints:
        return None
    requirements = classify_chunks(
        layout=layout,
        grid=grid,
        tile_footprints=footprints,
        min_exclusive_pixels=min_exclusive_pixels,
    )
    return TilePartition(
        requirements=dict(requirements),
        scene_to_tile=dict(tiles),
        tiles=tuple(sorted(footprints)),
    )


def summarise(
    partition: Optional[TilePartition],
    layout: ChunkLayout,
) -> Mapping[str, Any]:
    """Compact stats for run attrs / logging."""

    if partition is None:
        return {"tile_aware": False}
    multi_tile = sum(
        1 for req in partition.requirements.values() if len(req.required_tiles) > 1
    )
    unreachable = sum(
        1
        for req in partition.requirements.values()
        if req.unreachable_pixel_fraction > 0.0
    )
    return {
        "tile_aware": True,
        "tiles_used": list(partition.tiles),
        "multi_tile_chunk_count": int(multi_tile),
        "unreachable_chunk_count": int(unreachable),
        "chunk_count": len(layout),
    }


def _wgs84_reprojector(grid_crs: str):
    """Return a (x, y) → (x, y) callable for shapely.ops.transform, or None.

    None means the grid is already in WGS84 and no projection is needed.
    """

    if str(grid_crs).upper() in {"EPSG:4326", "OGC:CRS84", "CRS84"}:
        return None
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "tile classification requires pyproj for grid CRS reprojection"
        ) from exc
    transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)

    def reproject(x, y, z=None):  # shapely 2 passes (x, y) or (x, y, z)
        new_x, new_y = transformer.transform(x, y)
        if z is None:
            return new_x, new_y
        return new_x, new_y, z

    return reproject


def iter_required_tile_keys(
    partition: Optional[TilePartition],
    chunk_id: int,
) -> Iterable[Tuple[int, str]]:
    """Yield ``(chunk_id, tile_code)`` pairs for partition-aware loops."""

    if partition is None:
        return
    for tile_code in partition.tiles_for(chunk_id):
        yield chunk_id, tile_code


def required_tiles_for_chunks(
    partition: Optional[TilePartition],
    chunk_ids: Sequence[int],
) -> Mapping[int, Tuple[str, ...]]:
    """Bulk lookup for required tiles per chunk."""

    if partition is None:
        return {int(cid): () for cid in chunk_ids}
    return {int(cid): partition.tiles_for(int(cid)) for cid in chunk_ids}
