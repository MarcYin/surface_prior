import math

import pytest

from surface_priors.chunks import ChunkLayout
from surface_priors.selection import (
    SceneChunkStats,
    SelectionPlan,
    SelectionPolicy,
    select,
)
from surface_priors.types import GridSpec


@pytest.fixture
def two_by_two_layout():
    grid = GridSpec.from_bounds((0, 0, 1024, 1024), "EPSG:32630", 1)
    return ChunkLayout.from_grid(grid, chunk_size=512)


def test_select_keeps_top_k_by_mean_clear(two_by_two_layout):
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=1.0, mean_clear=0.95),
        SceneChunkStats(scene_index=1, chunk_id=0, usable_fraction=1.0, mean_clear=0.50),
        SceneChunkStats(scene_index=2, chunk_id=0, usable_fraction=1.0, mean_clear=0.80),
        SceneChunkStats(scene_index=3, chunk_id=0, usable_fraction=1.0, mean_clear=0.10),
    ]

    plan = select(layout=two_by_two_layout, stats=stats, policy=SelectionPolicy(top_k=3))

    assert plan.scenes_for(0) == (0, 2, 1)
    assert 0 not in plan.empty_chunks


def test_select_filters_below_min_usable_fraction(two_by_two_layout):
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=0.95, mean_clear=0.7),
        SceneChunkStats(scene_index=1, chunk_id=0, usable_fraction=0.30, mean_clear=0.99),
        SceneChunkStats(scene_index=2, chunk_id=0, usable_fraction=0.51, mean_clear=0.40),
    ]

    plan = select(
        layout=two_by_two_layout,
        stats=stats,
        policy=SelectionPolicy(top_k=3, min_usable_fraction=0.5),
    )

    # Scene 1 fails the usable_fraction floor and must be excluded.
    assert plan.scenes_for(0) == (0, 2)


def test_select_marks_empty_chunks_when_no_scene_passes(two_by_two_layout):
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=1, usable_fraction=0.2, mean_clear=0.99),
        SceneChunkStats(scene_index=1, chunk_id=1, usable_fraction=0.1, mean_clear=0.99),
    ]

    plan = select(
        layout=two_by_two_layout,
        stats=stats,
        policy=SelectionPolicy(top_k=3, min_usable_fraction=0.5),
    )

    assert plan.scenes_for(1) == ()
    assert set(plan.empty_chunks) == {0, 1, 2, 3}


def test_select_treats_nan_clear_as_unrankable(two_by_two_layout):
    stats = [
        SceneChunkStats(
            scene_index=0,
            chunk_id=0,
            usable_fraction=1.0,
            mean_clear=float("nan"),
        ),
        SceneChunkStats(scene_index=1, chunk_id=0, usable_fraction=1.0, mean_clear=0.4),
    ]

    plan = select(layout=two_by_two_layout, stats=stats, policy=SelectionPolicy(top_k=2))

    assert plan.scenes_for(0)[0] == 1
    assert math.isnan(stats[0].mean_clear)


def test_selection_plan_scene_indices_is_union_of_chunks(two_by_two_layout):
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=1.0, mean_clear=0.9),
        SceneChunkStats(scene_index=1, chunk_id=1, usable_fraction=1.0, mean_clear=0.8),
        SceneChunkStats(scene_index=2, chunk_id=2, usable_fraction=1.0, mean_clear=0.7),
        SceneChunkStats(scene_index=1, chunk_id=3, usable_fraction=1.0, mean_clear=0.6),
    ]

    plan = select(layout=two_by_two_layout, stats=stats, policy=SelectionPolicy(top_k=1))

    assert plan.scene_indices == (0, 1, 2)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        SelectionPolicy(top_k=0)
    with pytest.raises(ValueError):
        SelectionPolicy(min_usable_fraction=1.5)
    with pytest.raises(ValueError):
        SelectionPolicy(min_clear_score=-0.1)


def test_unknown_chunk_id_raises(two_by_two_layout):
    bad = [SceneChunkStats(scene_index=0, chunk_id=99, usable_fraction=1.0, mean_clear=0.9)]
    with pytest.raises(ValueError):
        select(layout=two_by_two_layout, stats=bad, policy=SelectionPolicy())


def test_selection_plan_empty_default():
    grid = GridSpec.from_bounds((0, 0, 4, 4), "EPSG:32630", 1)
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    plan = SelectionPlan(layout=layout, policy=SelectionPolicy())
    assert plan.scenes_for(0) == ()
    assert plan.scene_indices == ()
