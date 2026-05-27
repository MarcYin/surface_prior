import math

import pytest

from surface_priors.chunks import ChunkLayout
from surface_priors.selection import (
    SceneChunkStats,
    SelectionPlan,
    SelectionPolicy,
    select,
)
from surface_priors.tile_classification import ChunkTileRequirement, TilePartition
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


def test_select_drops_zero_usable_fraction(two_by_two_layout):
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=0.95, mean_clear=0.7),
        SceneChunkStats(scene_index=1, chunk_id=0, usable_fraction=0.0, mean_clear=0.99),
        SceneChunkStats(scene_index=2, chunk_id=0, usable_fraction=0.05, mean_clear=0.40),
    ]

    plan = select(
        layout=two_by_two_layout,
        stats=stats,
        policy=SelectionPolicy(top_k=3),
    )

    # Scene 1 is dropped (no contribution); scene 2's tiny usable_fraction is
    # kept because tile-aware selection no longer applies a global floor.
    assert plan.scenes_for(0) == (0, 2)


def test_select_marks_empty_chunks_when_no_scene_passes(two_by_two_layout):
    # Only chunk 1 sees any scene candidates, and they all have 0 usable
    # contribution — the chunk stays empty.
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=1, usable_fraction=0.0, mean_clear=float("nan")),
        SceneChunkStats(scene_index=1, chunk_id=1, usable_fraction=0.0, mean_clear=float("nan")),
    ]

    plan = select(
        layout=two_by_two_layout,
        stats=stats,
        policy=SelectionPolicy(top_k=3),
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
        SelectionPolicy(min_clear_score=-0.1)


def test_policy_rejects_legacy_min_usable_fraction_kwarg():
    with pytest.raises(TypeError):
        SelectionPolicy(min_usable_fraction=0.5)  # type: ignore[call-arg]


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


def test_select_partitions_top_k_per_required_tile(two_by_two_layout):
    # Chunk 0 needs both tile T and tile U. Without tile awareness, the four
    # T-tile scenes would shut out the U-tile scene because they have higher
    # mean_clear. With tile awareness, each required tile contributes its own
    # top-K so chunk 0's U-side gets filled too.
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=1.0, mean_clear=0.99),
        SceneChunkStats(scene_index=1, chunk_id=0, usable_fraction=1.0, mean_clear=0.95),
        SceneChunkStats(scene_index=2, chunk_id=0, usable_fraction=1.0, mean_clear=0.90),
        SceneChunkStats(scene_index=3, chunk_id=0, usable_fraction=1.0, mean_clear=0.85),
        SceneChunkStats(scene_index=4, chunk_id=0, usable_fraction=1.0, mean_clear=0.40),
    ]
    partition = TilePartition(
        requirements={0: ChunkTileRequirement(chunk_id=0, required_tiles=("T", "U"))},
        scene_to_tile={0: "T", 1: "T", 2: "T", 3: "T", 4: "U"},
        tiles=("T", "U"),
    )

    plan = select(
        layout=two_by_two_layout,
        stats=stats,
        policy=SelectionPolicy(top_k=2),
        partition=partition,
    )

    # Round-robin order: T#1, U#1, T#2 — U has only one scene so it appears once.
    assert plan.scenes_for(0) == (0, 4, 1)
