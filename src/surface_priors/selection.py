from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

from surface_priors.chunks import ChunkLayout


@dataclass(frozen=True)
class SceneChunkStats:
    """Per-(scene, chunk) scout statistics produced by an observation source.

    `usable_fraction` is the share of pixels in the chunk that have non-nodata
    cloud information. `mean_clear` is the mean clear-pixel score over those
    valid pixels (1.0 = perfectly clear, 0.0 = fully cloudy); NaN means no
    valid pixels were available.
    """

    scene_index: int
    chunk_id: int
    usable_fraction: float
    mean_clear: float


@dataclass(frozen=True)
class SelectionPolicy:
    """Rules for selecting which scenes feed which chunks.

    `top_k` caps the number of scenes per chunk after eligibility filtering.
    `min_usable_fraction` excludes scene/chunk pairs where too many pixels are
    nodata (off-swath, end-of-orbit, masked); this matters more for S2 than
    for cloud cover alone. `min_clear_score` is an optional cloud-score floor
    that defaults to no filter.
    """

    top_k: int = 3
    min_usable_fraction: float = 0.5
    min_clear_score: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.min_usable_fraction <= 1.0:
            raise ValueError("min_usable_fraction must be within [0, 1]")
        if self.min_clear_score < 0.0:
            raise ValueError("min_clear_score must be non-negative")


@dataclass(frozen=True)
class SelectionPlan:
    """Sparse mapping of chunks to the scene indices that should fill them.

    Scene indices are ordered best-first per chunk. Chunks that received no
    eligible scene are omitted from `selected`; callers should treat missing
    chunks as nodata in the output.
    """

    layout: ChunkLayout
    policy: SelectionPolicy
    selected: Mapping[int, Tuple[int, ...]] = field(default_factory=dict)
    empty_chunks: Tuple[int, ...] = ()

    def scenes_for(self, chunk_id: int) -> Tuple[int, ...]:
        return tuple(self.selected.get(chunk_id, ()))

    @property
    def scene_indices(self) -> Tuple[int, ...]:
        unique: dict[int, None] = {}
        for scenes in self.selected.values():
            for scene in scenes:
                unique.setdefault(int(scene), None)
        return tuple(sorted(unique))


def select(
    *,
    layout: ChunkLayout,
    stats: Sequence[SceneChunkStats],
    policy: SelectionPolicy,
) -> SelectionPlan:
    """Pick the top-K scenes per chunk after applying eligibility floors."""

    by_chunk: Dict[int, list[SceneChunkStats]] = {window.chunk_id: [] for window in layout}
    for entry in stats:
        if entry.chunk_id not in by_chunk:
            raise ValueError(
                f"stats reference chunk_id {entry.chunk_id} not present in layout"
            )
        by_chunk[entry.chunk_id].append(entry)

    selected: Dict[int, Tuple[int, ...]] = {}
    empty = []
    for chunk_id, entries in by_chunk.items():
        eligible = [
            entry
            for entry in entries
            if entry.usable_fraction >= policy.min_usable_fraction
            and _passes_clear_floor(entry, policy.min_clear_score)
        ]
        eligible.sort(key=_clear_sort_key, reverse=True)
        chosen = tuple(int(entry.scene_index) for entry in eligible[: policy.top_k])
        if chosen:
            selected[chunk_id] = chosen
        else:
            empty.append(chunk_id)

    return SelectionPlan(
        layout=layout,
        policy=policy,
        selected=selected,
        empty_chunks=tuple(empty),
    )


def _passes_clear_floor(entry: SceneChunkStats, floor: float) -> bool:
    if floor <= 0.0:
        return True
    if math.isnan(entry.mean_clear):
        return False
    return entry.mean_clear >= floor


def _clear_sort_key(entry: SceneChunkStats) -> Tuple[float, float, int]:
    clear = entry.mean_clear
    if math.isnan(clear):
        clear = -math.inf
    return clear, entry.usable_fraction, -entry.scene_index
