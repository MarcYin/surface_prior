import pytest

from surface_priors.chunks import ChunkLayout, chunk_grid
from surface_priors.types import GridSpec


def test_layout_tiles_grid_in_row_major_order():
    grid = GridSpec.from_bounds((0, 0, 1024, 1024), "EPSG:32630", 1)

    layout = ChunkLayout.from_grid(grid, chunk_size=512)

    assert layout.applied_chunk_size == 512
    assert layout.grid_shape == (1024, 1024)
    assert len(layout) == 4
    chunk_ids = [window.chunk_id for window in layout]
    assert chunk_ids == [0, 1, 2, 3]
    assert (layout[0].row_off, layout[0].col_off) == (0, 0)
    assert (layout[1].row_off, layout[1].col_off) == (0, 512)
    assert (layout[2].row_off, layout[2].col_off) == (512, 0)
    assert (layout[3].row_off, layout[3].col_off) == (512, 512)
    for window in layout:
        assert window.shape == (512, 512)


def test_layout_edge_chunks_are_smaller_when_grid_is_not_multiple():
    grid = GridSpec.from_bounds((0, 0, 700, 1300), "EPSG:32630", 1)

    layout = ChunkLayout.from_grid(grid, chunk_size=512)

    # 1300/512 = 3 rows (512, 512, 276); 700/512 = 2 cols (512, 188).
    assert layout.grid_shape == (1300, 700)
    assert len(layout) == 6
    edges = {(window.row_off, window.col_off): window.shape for window in layout}
    assert edges[(0, 0)] == (512, 512)
    assert edges[(0, 512)] == (512, 188)
    assert edges[(1024, 0)] == (276, 512)
    assert edges[(1024, 512)] == (276, 188)


def test_layout_snaps_chunk_size_to_block_multiple():
    grid = GridSpec.from_bounds((0, 0, 2048, 2048), "EPSG:32630", 1)

    layout = ChunkLayout.from_grid(grid, chunk_size=600, block_size=256)

    # 600 // 256 = 2 -> snapped to 512.
    assert layout.applied_chunk_size == 512
    assert layout.effective_chunk_size == 512
    assert len(layout) == 16


def test_layout_promotes_to_block_size_when_chunk_smaller():
    grid = GridSpec.from_bounds((0, 0, 1024, 1024), "EPSG:32630", 1)

    layout = ChunkLayout.from_grid(grid, chunk_size=128, block_size=256)

    assert layout.applied_chunk_size == 256
    assert layout.effective_chunk_size == 256
    assert len(layout) == 16


def test_chunk_grid_translates_origin_correctly():
    grid = GridSpec.from_bounds((100.0, 200.0, 1124.0, 1224.0), "EPSG:32630", 1)
    layout = ChunkLayout.from_grid(grid, chunk_size=512)
    bottom_right = next(
        window for window in layout if window.row_off == 512 and window.col_off == 512
    )

    sub_grid = chunk_grid(grid, bottom_right)

    assert sub_grid.width == 512
    assert sub_grid.height == 512
    assert sub_grid.bounds == (612.0, 200.0, 1124.0, 712.0)
    assert sub_grid.crs == grid.crs
    assert sub_grid.resolution == grid.resolution


def test_chunk_size_must_be_positive():
    grid = GridSpec.from_bounds((0, 0, 10, 10), "EPSG:32630", 1)
    with pytest.raises(ValueError):
        ChunkLayout.from_grid(grid, chunk_size=0)
