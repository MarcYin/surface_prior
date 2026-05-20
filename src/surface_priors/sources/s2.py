"""Shared Sentinel-2 L2A constants and quality helpers.

Used by both the Google Earth Engine source (`s2_gee.py`) and any future
STAC-API source. Keeping the band map, score band, and SCL fallback in a
single module ensures GEE and STAC produce comparable priors.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from surface_priors.chunks import ChunkLayout
from surface_priors.selection import SceneChunkStats

S2_L2A_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS_COLLECTION_ID = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

S2_L2A_BAND_MAP: Mapping[str, str] = {
    "s2_b01_aerosol": "B1",
    "s2_b02_blue": "B2",
    "s2_b03_green": "B3",
    "s2_b04_red": "B4",
    "s2_b05_re1": "B5",
    "s2_b06_re2": "B6",
    "s2_b07_re3": "B7",
    "s2_b08_nir": "B8",
    "s2_b8a_nir_narrow": "B8A",
    "s2_b09_water_vapor": "B9",
    "s2_b11_swir1": "B11",
    "s2_b12_swir2": "B12",
}

DEFAULT_SCORE_BAND = "cs"
DEFAULT_CLEAR_THRESHOLD = 0.0
DEFAULT_SCOUT_FACTOR = 8
DEFAULT_CHUNK_SIZE = 512
NODATA_QUALITY = 65535

SCL_NODATA = 0
SCL_CLEAR_CLASSES = (4, 5, 6, 11)
SCL_MARGINAL_CLASSES = (7,)
SCL_DARK_CLASSES = (2, 3)
SCL_BAD_CLASSES = (1, 8, 9, 10)


def cloud_score_to_quality(
    score: np.ndarray,
    *,
    clear_threshold: float = DEFAULT_CLEAR_THRESHOLD,
) -> np.ndarray:
    """Map Cloud Score+ values (0 cloudy, 1 clear) to compositor quality.

    Higher `cs` becomes lower numeric quality so the existing best-pixel
    score prefers clearer pixels. Pixels below `clear_threshold`, NaN, or
    outside [0, 1] are stored as the package-wide nodata sentinel.
    """

    score = np.asarray(score, dtype="float32")
    quality = np.full(score.shape, NODATA_QUALITY, dtype="uint16")
    valid = np.isfinite(score) & (score >= float(clear_threshold)) & (score <= 1.0)
    if np.any(valid):
        scaled = np.rint((1.0 - score[valid]) * 10_000.0)
        scaled = np.clip(scaled, 0, NODATA_QUALITY - 1)
        quality[valid] = scaled.astype("uint16", copy=False)
    return quality


def cloud_score_valid_mask(score: np.ndarray) -> np.ndarray:
    """Pixels with finite Cloud Score+ in the valid [0, 1] range."""

    score = np.asarray(score)
    return np.isfinite(score) & (score >= 0.0) & (score <= 1.0)


def scl_to_quality(scl: np.ndarray) -> np.ndarray:
    """Map Sentinel-2 SCL classes to compositor quality (lower = better).

    Clear classes (4, 5, 6, 11) map to 0. Unclassified (7) maps to 1.
    Dark/shadow (2, 3) map to 2. SCL=0 (no-data) and 1, 8, 9, 10
    (saturated/cloud/cirrus) map to the package-wide nodata sentinel.
    """

    scl = np.asarray(scl)
    quality = np.full(scl.shape, NODATA_QUALITY, dtype="uint16")
    quality[np.isin(scl, SCL_CLEAR_CLASSES)] = 0
    quality[np.isin(scl, SCL_MARGINAL_CLASSES)] = 1
    quality[np.isin(scl, SCL_DARK_CLASSES)] = 2
    return quality


def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    """Pixels with non-zero SCL (i.e., real data)."""

    return np.asarray(scl) != SCL_NODATA


def scl_clear_score(scl: np.ndarray) -> np.ndarray:
    """Per-pixel "clear" score for SCL: 1.0 if clear class, 0.0 otherwise.

    Used so SCL-based scouting and CS+ scouting both report `mean_clear`
    in [0, 1] with the same direction (higher = clearer).
    """

    scl = np.asarray(scl)
    return np.where(np.isin(scl, SCL_CLEAR_CLASSES), 1.0, 0.0).astype("float32")


def apply_zero_as_nodata(
    data: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Mark `fill_value` pixels (default 0) as NaN in float surface reflectance.

    Sentinel-2 L2A COGs frequently use 0 as off-swath nodata even when the
    file metadata omits an explicit nodata. The compositor drops NaN pixels
    via `valid_pixel_mask` regardless of band ordering.
    """

    arr = np.asarray(data, dtype="float32", copy=True)
    arr[arr == np.float32(fill_value)] = np.nan
    return arr


def aggregate_chunk_stats(
    *,
    scene_index: int,
    coarse_score: np.ndarray,
    coarse_valid: np.ndarray,
    layout: ChunkLayout,
    scout_factor: int,
) -> Tuple[SceneChunkStats, ...]:
    """Reduce a coarse cloud score raster into per-chunk usable/clear stats.

    `coarse_score` and `coarse_valid` cover the full AOI at
    `scout_factor`-times downsampled resolution; both share the same shape.
    The aggregator maps each chunk window to its block in the coarse raster
    (rounded), then computes `mean(score | valid)` and `valid / total`.
    """

    if scout_factor <= 0:
        raise ValueError("scout_factor must be positive")
    score = np.asarray(coarse_score, dtype="float32")
    valid = np.asarray(coarse_valid, dtype="bool")
    if score.shape != valid.shape:
        raise ValueError("coarse_score and coarse_valid must share shape")

    stats = []
    coarse_h, coarse_w = score.shape
    for window in layout:
        cr0 = window.row_off // scout_factor
        cc0 = window.col_off // scout_factor
        cr1 = min(coarse_h, _ceil_div(window.row_off + window.height, scout_factor))
        cc1 = min(coarse_w, _ceil_div(window.col_off + window.width, scout_factor))
        block_score = score[cr0:cr1, cc0:cc1]
        block_valid = valid[cr0:cr1, cc0:cc1]
        total = block_valid.size
        if total == 0:
            stats.append(
                SceneChunkStats(
                    scene_index=scene_index,
                    chunk_id=window.chunk_id,
                    usable_fraction=0.0,
                    mean_clear=float("nan"),
                )
            )
            continue
        n_valid = int(np.count_nonzero(block_valid))
        usable_fraction = float(n_valid) / float(total)
        mean_clear = (
            float("nan") if n_valid == 0 else float(np.mean(block_score[block_valid]))
        )
        stats.append(
            SceneChunkStats(
                scene_index=scene_index,
                chunk_id=window.chunk_id,
                usable_fraction=usable_fraction,
                mean_clear=mean_clear,
            )
        )
    return tuple(stats)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))
