from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from surface_priors.chunks import ChunkLayout
from surface_priors.tile_classification import TilePartition

_SINGLE_TILE = "__single__"


@dataclass(frozen=True)
class SceneChunkStats:
    """Per-(scene, chunk) scout statistics produced by an observation source.

    ``usable_fraction`` is the share of pixels in the chunk that have
    non-nodata cloud information; selection drops entries with
    ``usable_fraction <= 0`` (off-swath / no contribution) and uses the
    remainder only as a sort tiebreaker. ``mean_clear`` is the mean
    clear-pixel score over those valid pixels; NaN means no valid pixels
    were available.
    """

    scene_index: int
    chunk_id: int
    usable_fraction: float
    mean_clear: float


@dataclass(frozen=True)
class SelectionPolicy:
    """Rules for selecting which scenes feed which chunks.

    ``top_k`` caps the number of scenes picked per (chunk, tile) bucket
    after eligibility filtering. With tile-aware selection a chunk that
    needs two tiles will end up with up to ``2 * top_k`` scenes. The
    legacy ``min_usable_fraction`` floor has been removed: it was a
    workaround for tile-seam stripes that the tile-aware selector now
    handles structurally. ``min_clear_score`` remains as an optional
    cloud-score floor (defaults to no filter).
    """

    top_k: int = 3
    min_clear_score: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.min_clear_score < 0.0:
            raise ValueError("min_clear_score must be non-negative")


@dataclass(frozen=True)
class SelectionPlan:
    """Sparse mapping of chunks to the scene indices that should fill them.

    Scene indices are ordered best-first per chunk. Chunks that received
    no eligible scene are omitted from ``selected``; callers should
    treat missing chunks as nodata in the output.
    """

    layout: ChunkLayout
    policy: SelectionPolicy
    selected: Mapping[int, Tuple[int, ...]] = field(default_factory=dict)
    empty_chunks: Tuple[int, ...] = ()
    partition: Optional[TilePartition] = None

    def scenes_for(self, chunk_id: int) -> Tuple[int, ...]:
        return tuple(self.selected.get(chunk_id, ()))

    @property
    def scene_indices(self) -> Tuple[int, ...]:
        unique: Dict[int, None] = {}
        for scenes in self.selected.values():
            for scene in scenes:
                unique.setdefault(int(scene), None)
        return tuple(sorted(unique))


def select(
    *,
    layout: ChunkLayout,
    stats: Sequence[SceneChunkStats],
    policy: SelectionPolicy,
    partition: Optional[TilePartition] = None,
) -> SelectionPlan:
    """Pick the top-K scenes per (chunk, required-tile) after eligibility floors.

    When ``partition`` is supplied, scenes are grouped by the MGRS tile
    they belong to and ranking happens within each required tile for the
    chunk; the chunk's final pick is the ordered union across its
    required tiles. When ``partition`` is ``None`` the behavior reduces
    to "single synthetic tile per chunk" — equivalent to legacy ranking
    minus the dropped usable_fraction floor.
    """

    chunk_ids = {window.chunk_id for window in layout}
    by_chunk_tile: Dict[int, Dict[str, List[SceneChunkStats]]] = {
        cid: {} for cid in chunk_ids
    }
    for entry in stats:
        if entry.chunk_id not in by_chunk_tile:
            raise ValueError(
                f"stats reference chunk_id {entry.chunk_id} not present in layout"
            )
        if entry.usable_fraction <= 0.0:
            continue
        if not _passes_clear_floor(entry, policy.min_clear_score):
            continue
        tile = _tile_of(partition, entry.scene_index)
        by_chunk_tile[entry.chunk_id].setdefault(tile, []).append(entry)

    selected: Dict[int, Tuple[int, ...]] = {}
    empty: List[int] = []
    for chunk_id in sorted(chunk_ids):
        required = _required_tiles_for(partition, chunk_id)
        chosen = _pick_for_chunk(
            tile_to_entries=by_chunk_tile.get(chunk_id, {}),
            required_tiles=required,
            top_k=policy.top_k,
        )
        if chosen:
            selected[chunk_id] = chosen
        else:
            empty.append(chunk_id)

    return SelectionPlan(
        layout=layout,
        policy=policy,
        selected=selected,
        empty_chunks=tuple(empty),
        partition=partition,
    )


def _pick_for_chunk(
    *,
    tile_to_entries: Mapping[str, List[SceneChunkStats]],
    required_tiles: Tuple[str, ...],
    top_k: int,
) -> Tuple[int, ...]:
    """Per-tile top-K then union, preserving best-first order across tiles."""

    if not tile_to_entries:
        return ()
    tiles_in_order: List[str] = []
    seen: set[str] = set()
    for tile in required_tiles:
        if tile in tile_to_entries and tile not in seen:
            tiles_in_order.append(tile)
            seen.add(tile)
    # Tiles in stats but not classified as required (e.g. partition is None
    # or the scout reported a scene from an unexpected tile) still contribute.
    for tile in tile_to_entries:
        if tile not in seen:
            tiles_in_order.append(tile)
            seen.add(tile)

    per_tile_picks: List[List[SceneChunkStats]] = []
    for tile in tiles_in_order:
        entries = list(tile_to_entries.get(tile, ()))
        entries.sort(key=_clear_sort_key, reverse=True)
        per_tile_picks.append(entries[:top_k])

    out: List[int] = []
    out_seen: set[int] = set()
    # Round-robin so each required tile contributes its #1 pick before any
    # tile contributes its #2 — keeps both sides of a seam represented.
    max_picks = max((len(p) for p in per_tile_picks), default=0)
    for rank in range(max_picks):
        for picks in per_tile_picks:
            if rank < len(picks):
                scene = int(picks[rank].scene_index)
                if scene not in out_seen:
                    out.append(scene)
                    out_seen.add(scene)
    return tuple(out)


def _required_tiles_for(
    partition: Optional[TilePartition],
    chunk_id: int,
) -> Tuple[str, ...]:
    if partition is None:
        return (_SINGLE_TILE,)
    tiles = partition.tiles_for(chunk_id)
    return tiles if tiles else ()


def _tile_of(partition: Optional[TilePartition], scene_index: int) -> str:
    if partition is None:
        return _SINGLE_TILE
    tile = partition.tile_of(scene_index)
    return tile if tile is not None else _SINGLE_TILE


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
