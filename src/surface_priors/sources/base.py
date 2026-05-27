from __future__ import annotations

from typing import Optional, Protocol, Sequence, Tuple

from surface_priors.chunks import ChunkLayout
from surface_priors.selection import SceneChunkStats, SelectionPlan
from surface_priors.tile_classification import TilePartition
from surface_priors.types import GridSpec, Observation


class ObservationSource(Protocol):
    """Protocol for sources that return native-grid prior observations."""

    @property
    def name(self) -> str:
        """Stable source name included in request hashes."""

    def load_observations(
        self,
        *,
        grid: GridSpec,
        band_names: Sequence[str],
    ) -> Sequence[Observation]:
        """Return observations already aligned to `grid`."""


class ChunkedObservationSource(Protocol):
    """Optional protocol for sources that support cost-bounded chunked access.

    A chunked source replaces the eager `load_observations` flow with three
    cheaper passes: a coarse scout that returns per-(scene, chunk) statistics,
    a sparse fetch that materialises only the chunks selected by the caller,
    and (optionally) a way to declare a preferred chunk size derived from the
    underlying storage layout.
    """

    @property
    def name(self) -> str:
        """Stable source name included in request hashes."""

    def scout(
        self,
        *,
        grid: GridSpec,
        layout: ChunkLayout,
        band_names: Sequence[str],
        temporal_filter: Optional[Tuple[str, str]] = None,
    ) -> Sequence[SceneChunkStats]:
        """Return clear/usable statistics per (scene, chunk) using cheap reads.

        `temporal_filter` lets callers restrict the cached scene list to a
        sub-range (e.g., one month inside a 3-month search). Sources that
        don't implement filtering may ignore the argument.
        """

    def fetch_selected(
        self,
        *,
        grid: GridSpec,
        plan: SelectionPlan,
        band_names: Sequence[str],
        scene_index: int,
        chunk_id: int,
    ) -> Optional[Observation]:
        """Materialise one (scene, chunk) observation or return None if absent."""

    def block_size(
        self,
        *,
        grid: GridSpec,
        band_names: Sequence[str],
    ) -> Optional[int]:
        """Preferred chunk-size snap derived from storage block size, if known."""

    def tile_partition(
        self,
        *,
        grid: GridSpec,
        layout: ChunkLayout,
    ) -> Optional[TilePartition]:
        """Classify each chunk by the source tiles needed to cover it.

        Sources whose scenes each cover only one tile (Sentinel-2 L2A on
        STAC, MGRS-tiled products in general) implement this so that
        selection can take top-K per required tile and union, avoiding
        seam-stripe gaps. Sources with no tile concept omit the method or
        return ``None`` — selection then ranks globally per chunk.
        """
