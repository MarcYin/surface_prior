"""Geometry-only tile classification.

These tests pin the seam-straddling case that motivated the rewrite: a
chunk that has exclusive coverage from two tiles must list both as
required, while a chunk in the overlap region only needs one.
"""

from __future__ import annotations

import pytest

shapely = pytest.importorskip("shapely")

from surface_priors.chunks import ChunkLayout
from surface_priors.tile_classification import (
    build_partition,
    classify_chunks,
)
from surface_priors.types import GridSpec


def _grid_with_wgs84(bounds, crs, resolution, wgs84):
    return GridSpec.from_bounds(bounds=bounds, crs=crs, resolution=resolution, wgs84_bounds=wgs84)


def test_classify_chunks_marks_seam_chunk_as_multi_tile():
    grid = GridSpec.from_bounds((0, 0, 1024, 1024), "EPSG:32630", 1.0)
    layout = ChunkLayout.from_grid(grid, chunk_size=512)
    # Tile T spans x ∈ [0, 700]; tile U spans x ∈ [600, 1024]. The overlap
    # region is x ∈ [600, 700]; the seam runs through chunks 1 and 3 where
    # each tile has its own exclusive coverage.
    from shapely.geometry import box

    footprints = {
        "T": box(0, 0, 700, 1024),
        "U": box(600, 0, 1024, 1024),
    }
    requirements = classify_chunks(layout=layout, grid=grid, tile_footprints=footprints)

    # Chunks 0 (top-left) and 2 (bottom-left) sit inside T only — single-tile.
    assert requirements[0].required_tiles == ("T",)
    assert requirements[2].required_tiles == ("T",)
    # Chunks 1 and 3 sit at the seam: T covers x<600 exclusively, U covers
    # x>600 exclusively. Both tiles are required.
    assert set(requirements[1].required_tiles) == {"T", "U"}
    assert set(requirements[3].required_tiles) == {"T", "U"}


def test_classify_chunks_in_pure_overlap_picks_single_tile():
    grid = GridSpec.from_bounds((0, 0, 200, 200), "EPSG:32630", 1.0)
    layout = ChunkLayout.from_grid(grid, chunk_size=200)
    from shapely.geometry import box

    # Both tiles fully cover the chunk; neither has exclusive area.
    footprints = {
        "T": box(-50, -50, 250, 250),
        "U": box(-50, -50, 250, 250),
    }
    requirements = classify_chunks(layout=layout, grid=grid, tile_footprints=footprints)

    assert len(requirements[0].required_tiles) == 1
    assert requirements[0].required_tiles[0] in {"T", "U"}
    assert requirements[0].unreachable_pixel_fraction == 0.0


def test_classify_chunks_flags_unreachable_chunks():
    grid = GridSpec.from_bounds((0, 0, 1024, 1024), "EPSG:32630", 1.0)
    layout = ChunkLayout.from_grid(grid, chunk_size=512)
    from shapely.geometry import box

    # Tile T covers only the top half. The bottom row of chunks should be
    # entirely unreachable.
    footprints = {"T": box(0, 512, 1024, 1024)}
    requirements = classify_chunks(layout=layout, grid=grid, tile_footprints=footprints)

    assert requirements[2].required_tiles == ()
    assert requirements[2].unreachable_pixel_fraction == 1.0
    assert requirements[0].required_tiles == ("T",)
    assert requirements[0].unreachable_pixel_fraction == 0.0


def test_build_partition_returns_none_without_tile_info():
    grid = _grid_with_wgs84((0, 0, 1024, 1024), "EPSG:32630", 1.0, None)
    layout = ChunkLayout.from_grid(grid, chunk_size=512)
    partition = build_partition(
        layout=layout,
        grid=grid,
        scene_tiles={},
        scene_geometries_wgs84={},
    )
    assert partition is None


def test_build_partition_round_trips_wgs84_geometry():
    # Grid in WGS84 so we can hand WGS84 polygons through without reprojection.
    grid = _grid_with_wgs84((0.0, 0.0, 1.0, 1.0), "EPSG:4326", 0.5, None)
    layout = ChunkLayout.from_grid(grid, chunk_size=1)
    scene_tiles = {0: "T", 1: "U"}
    scene_geoms = {
        0: {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0], [0.0, 0.0]]]},
        1: {"type": "Polygon", "coordinates": [[[0.4, 0.0], [1.0, 0.0], [1.0, 1.0], [0.4, 1.0], [0.4, 0.0]]]},
    }
    partition = build_partition(
        layout=layout,
        grid=grid,
        scene_tiles=scene_tiles,
        scene_geometries_wgs84=scene_geoms,
    )
    assert partition is not None
    assert set(partition.tiles) == {"T", "U"}
    assert partition.scene_to_tile == {0: "T", 1: "U"}
    # Layout is 2x2 (chunk_size=1 in 2x2 native pixels). chunks 0 and 2
    # (left column) need only T; chunks 1 and 3 (right column) need only U.
    assert partition.tiles_for(0) == ("T",)
    assert partition.tiles_for(1) == ("U",)
