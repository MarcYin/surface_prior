from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class QualityRules:
    """Scoring rules for best-pixel selection.

    Lower scores are better. The core builder intentionally has no
    observation-day, month, or year preference. Time-window selection belongs
    outside this package.
    """

    max_usable_quality: int = 32767
    nodata_quality: int = 65535
    quality_weight: float = 1_000_000.0
    sample_index_weight: float = 1.0


def valid_pixel_mask(data: np.ndarray, quality: np.ndarray, rules: QualityRules) -> np.ndarray:
    finite_data = np.all(np.isfinite(data), axis=0)
    finite_quality = np.isfinite(quality)
    usable_quality = quality <= rules.max_usable_quality
    return finite_data & finite_quality & usable_quality


def score_pixels(
    *,
    quality: np.ndarray,
    sample_index: Optional[np.ndarray],
    valid_mask: np.ndarray,
    source_order: int,
    rules: QualityRules,
) -> np.ndarray:
    quality_score = quality.astype("float64", copy=False) * rules.quality_weight
    if sample_index is None:
        sample_score = 0.0
    else:
        # Promote to float64 before the fill: the nodata sentinel (65535) does
        # not fit sample_index's int16 dtype, and NumPy >= 2.5 raises an
        # OverflowError when it weak-promotes the Python scalar into int16.
        sample_float = sample_index.astype("float64", copy=False)
        sample_score = np.where(np.isfinite(sample_float), sample_float, float(rules.nodata_quality))
        sample_score = sample_score * rules.sample_index_weight
    score = quality_score + sample_score + float(source_order)
    return np.where(valid_mask, score, np.inf)

