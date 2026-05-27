import numpy as np
import pytest

from surface_priors.chunks import ChunkLayout
from surface_priors.composite import ChunkedCompositor
from surface_priors.selection import (
    SceneChunkStats,
    SelectionPolicy,
    select,
)
from surface_priors.types import GridSpec, Observation


def _full_grid():
    return GridSpec.from_bounds((0, 0, 4, 4), "EPSG:32630", 1)


def _scene_observation(*, value: float, quality: int, window_shape):
    height, width = window_shape
    data = np.full((1, height, width), value, dtype="float32")
    qa = np.full((height, width), quality, dtype="uint16")
    return data, qa


def test_chunked_compositor_picks_best_quality_per_chunk():
    grid = _full_grid()
    layout = ChunkLayout.from_grid(grid, chunk_size=2)

    # Two scenes; scene 0 wins in chunks 0, 3 (lower quality), scene 1 in 1, 2.
    scene_quality = {
        0: {0: 0, 1: 4, 2: 4, 3: 0},
        1: {0: 4, 1: 0, 2: 0, 3: 4},
    }
    scene_value = {0: 0.10, 1: 0.20}
    bands = ("iso",)

    def chunk_loader(scene_index, chunk_id):
        window = layout[chunk_id]
        data, qa = _scene_observation(
            value=scene_value[scene_index],
            quality=scene_quality[scene_index][chunk_id],
            window_shape=window.shape,
        )
        return Observation(data=data, quality=qa, band_names=bands)

    stats = [
        SceneChunkStats(scene_index=scene, chunk_id=chunk, usable_fraction=1.0, mean_clear=1.0)
        for scene in (0, 1)
        for chunk in range(len(layout))
    ]
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=2))

    composite = ChunkedCompositor().compose(
        product_id="t",
        grid=grid,
        band_names=bands,
        plan=plan,
        chunk_loader=chunk_loader,
    )

    # Verify each chunk took on the winning scene's value.
    assert composite.data.shape == (1, 4, 4)
    np.testing.assert_array_equal(
        composite.data[0],
        np.array(
            [
                [0.10, 0.10, 0.20, 0.20],
                [0.10, 0.10, 0.20, 0.20],
                [0.20, 0.20, 0.10, 0.10],
                [0.20, 0.20, 0.10, 0.10],
            ],
            dtype="float32",
        ),
    )
    # selected_observation must hold the GLOBAL scene index, not the per-chunk local index.
    np.testing.assert_array_equal(
        composite.selected_observation,
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype="int16",
        ),
    )


def test_empty_chunks_left_as_nodata():
    grid = _full_grid()
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    bands = ("iso",)

    stats = [
        # Only chunk 0 has an eligible scene; everything else fails the usable_fraction floor.
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=1.0, mean_clear=0.9),
        SceneChunkStats(scene_index=0, chunk_id=1, usable_fraction=0.0, mean_clear=float("nan")),
        SceneChunkStats(scene_index=0, chunk_id=2, usable_fraction=0.0, mean_clear=float("nan")),
        SceneChunkStats(scene_index=0, chunk_id=3, usable_fraction=0.0, mean_clear=float("nan")),
    ]
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=1))

    def chunk_loader(scene_index, chunk_id):
        window = layout[chunk_id]
        data, qa = _scene_observation(value=0.5, quality=0, window_shape=window.shape)
        return Observation(data=data, quality=qa, band_names=bands)

    composite = ChunkedCompositor().compose(
        product_id="t",
        grid=grid,
        band_names=bands,
        plan=plan,
        chunk_loader=chunk_loader,
    )

    assert np.allclose(composite.data[0, :2, :2], 0.5)
    assert np.isnan(composite.data[0, 2:, :]).all()
    assert np.isnan(composite.data[0, :, 2:]).all()
    assert composite.attrs["empty_chunk_count"] == 3
    assert composite.attrs["compositor"] == "chunked_best_pixel_v2"


def test_chunk_loader_returning_none_skips_observation():
    grid = _full_grid()
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    bands = ("iso",)

    stats = [
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=1.0, mean_clear=0.5),
        SceneChunkStats(scene_index=1, chunk_id=0, usable_fraction=1.0, mean_clear=0.9),
    ]
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=2))

    def chunk_loader(scene_index, chunk_id):
        if scene_index == 1:
            return None
        window = layout[chunk_id]
        data, qa = _scene_observation(value=0.4, quality=0, window_shape=window.shape)
        return Observation(data=data, quality=qa, band_names=bands)

    composite = ChunkedCompositor().compose(
        product_id="t",
        grid=grid,
        band_names=bands,
        plan=plan,
        chunk_loader=chunk_loader,
    )

    assert composite.data[0, 0, 0] == np.float32(0.4)
    # Only scene 0 actually contributed observation data, so selected_observation is 0 everywhere it ran.
    assert composite.selected_observation[0, 0] == 0


def test_layout_grid_shape_must_match_compose_grid():
    grid = GridSpec.from_bounds((0, 0, 4, 4), "EPSG:32630", 1)
    other_grid = GridSpec.from_bounds((0, 0, 8, 8), "EPSG:32630", 1)
    layout = ChunkLayout.from_grid(other_grid, chunk_size=2)
    plan = select(layout=layout, stats=(), policy=SelectionPolicy())

    with pytest.raises(ValueError):
        ChunkedCompositor().compose(
            product_id="t",
            grid=grid,
            band_names=("iso",),
            plan=plan,
            chunk_loader=lambda scene, chunk: None,
        )
