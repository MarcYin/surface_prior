from __future__ import annotations

from typing import Optional, Protocol, Sequence

from surface_priors.chunks import ChunkLayout
from surface_priors.selection import SceneChunkStats, SelectionPlan
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
    ) -> Sequence[SceneChunkStats]:
        """Return clear/usable statistics per (scene, chunk) using cheap reads."""

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
