import numpy as np
import pytest

from surface_priors.chunks import ChunkLayout
from surface_priors.sources.s2 import (
    NODATA_QUALITY,
    aggregate_chunk_stats,
    apply_zero_as_nodata,
    cloud_score_to_quality,
    cloud_score_valid_mask,
    scl_clear_score,
    scl_to_quality,
    scl_valid_mask,
)
from surface_priors.types import GridSpec


def test_cloud_score_to_quality_inverts_score():
    score = np.array([[1.0, 0.5], [0.0, 0.25]], dtype="float32")

    quality = cloud_score_to_quality(score)

    assert quality.dtype == np.uint16
    assert quality[0, 0] == 0  # clearest -> lowest quality value
    assert quality[1, 0] == 10000  # cs=0 -> (1 - 0) * 10000, still well below the nodata sentinel
    assert quality[0, 1] == 5000
    assert quality[1, 1] == 7500
    assert quality[1, 0] < NODATA_QUALITY


def test_cloud_score_to_quality_threshold_marks_below_as_nodata():
    score = np.array([0.7, 0.5, 0.2, np.nan, -0.1], dtype="float32")

    quality = cloud_score_to_quality(score, clear_threshold=0.5)

    assert quality[0] == 3000
    assert quality[1] == 5000
    assert quality[2] == NODATA_QUALITY
    assert quality[3] == NODATA_QUALITY
    assert quality[4] == NODATA_QUALITY


def test_cloud_score_to_quality_caps_at_uint16_limit():
    score = np.array([0.0], dtype="float32")
    quality = cloud_score_to_quality(score)
    # (1 - 0) * 10000 = 10000, well below NODATA_QUALITY-1=65534.
    assert quality[0] == 10000


def test_apply_zero_as_nodata_promotes_zero_to_nan():
    data = np.array([[0.0, 1.0], [2.0, 0.0]], dtype="float32")

    cleaned = apply_zero_as_nodata(data)

    assert np.isnan(cleaned[0, 0])
    assert np.isnan(cleaned[1, 1])
    assert cleaned[0, 1] == np.float32(1.0)
    assert cleaned[1, 0] == np.float32(2.0)


def test_cloud_score_valid_mask_excludes_nan_and_out_of_range():
    score = np.array([0.0, 0.5, 1.0, np.nan, 1.5, -0.1], dtype="float32")
    mask = cloud_score_valid_mask(score)
    np.testing.assert_array_equal(mask, [True, True, True, False, False, False])


def test_aggregate_chunk_stats_reduces_coarse_to_chunks():
    grid = GridSpec.from_bounds((0, 0, 1024, 1024), "EPSG:32630", 1)
    layout = ChunkLayout.from_grid(grid, chunk_size=512)
    # Coarse grid at 8x => 128x128.
    coarse_score = np.full((128, 128), 0.8, dtype="float32")
    coarse_valid = np.ones((128, 128), dtype="bool")
    # Make the top-left coarse block partly cloudy (clear=0.4) and the
    # bottom-right block half nodata.
    coarse_score[:64, :64] = 0.4
    coarse_valid[64:, 64:][:32, :32] = False

    stats = aggregate_chunk_stats(
        scene_index=7,
        coarse_score=coarse_score,
        coarse_valid=coarse_valid,
        layout=layout,
        scout_factor=8,
    )
    by_chunk = {entry.chunk_id: entry for entry in stats}

    assert by_chunk[0].scene_index == 7
    assert by_chunk[0].mean_clear == pytest.approx(0.4, rel=1e-5)
    assert by_chunk[0].usable_fraction == 1.0
    assert by_chunk[3].usable_fraction == 0.75  # one quarter of the block is masked out
    assert by_chunk[3].mean_clear == pytest.approx(0.8, rel=1e-5)
    assert by_chunk[1].mean_clear == pytest.approx(0.8, rel=1e-5)  # untouched block


def test_scl_to_quality_buckets_classes_correctly():
    scl = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype="int16")
    quality = scl_to_quality(scl)
    assert quality[0] == NODATA_QUALITY  # no_data
    assert quality[1] == NODATA_QUALITY  # saturated
    assert quality[2] == 2  # dark area
    assert quality[3] == 2  # cloud shadow
    assert quality[4] == 0  # vegetation
    assert quality[5] == 0  # bare
    assert quality[6] == 0  # water
    assert quality[7] == 1  # unclassified
    assert quality[8] == NODATA_QUALITY  # cloud medium
    assert quality[9] == NODATA_QUALITY  # cloud high
    assert quality[10] == NODATA_QUALITY  # cirrus
    assert quality[11] == 0  # snow


def test_scl_valid_mask_excludes_zero():
    scl = np.array([[0, 4], [5, 1]], dtype="int16")
    mask = scl_valid_mask(scl)
    np.testing.assert_array_equal(mask, [[False, True], [True, True]])


def test_scl_clear_score_is_one_for_clear_classes_only():
    scl = np.array([4, 8, 11, 7], dtype="int16")
    score = scl_clear_score(scl)
    np.testing.assert_array_equal(score, [1.0, 0.0, 1.0, 0.0])


def test_aggregate_chunk_stats_handles_fully_masked_block():
    grid = GridSpec.from_bounds((0, 0, 512, 512), "EPSG:32630", 1)
    layout = ChunkLayout.from_grid(grid, chunk_size=512)
    coarse_score = np.full((64, 64), np.nan, dtype="float32")
    coarse_valid = np.zeros((64, 64), dtype="bool")

    (entry,) = aggregate_chunk_stats(
        scene_index=0,
        coarse_score=coarse_score,
        coarse_valid=coarse_valid,
        layout=layout,
        scout_factor=8,
    )

    assert entry.usable_fraction == 0.0
    assert np.isnan(entry.mean_clear)
