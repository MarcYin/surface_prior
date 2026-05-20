from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from surface_priors.types import GridSpec


@dataclass(frozen=True)
class ChunkWindow:
    """One tile of a `ChunkLayout`, addressed by integer pixel offsets.

    `chunk_id` is the position of the window in the layout's row-major ordering
    and is stable across runs for a given `ChunkLayout`.
    """

    chunk_id: int
    row_off: int
    col_off: int
    height: int
    width: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row_off, self.row_off + self.height)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_off, self.col_off + self.width)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width


@dataclass(frozen=True)
class ChunkLayout:
    """Row-major tiling of a `GridSpec` into fixed-size chunks.

    Edge chunks on the bottom row and right column can be smaller than
    `chunk_size` when the grid is not an exact multiple. `chunk_size` is the
    requested target; `effective_chunk_size` reports the value actually used
    after any block-size snap.
    """

    chunk_size: int
    grid_shape: Tuple[int, int]
    windows: Tuple[ChunkWindow, ...]
    effective_chunk_size: Optional[int] = None

    @classmethod
    def from_grid(
        cls,
        grid: GridSpec,
        *,
        chunk_size: int = 512,
        block_size: Optional[int] = None,
    ) -> "ChunkLayout":
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        effective = _snap_chunk_size(chunk_size, block_size=block_size)
        height, width = grid.shape
        windows = tuple(_generate_windows(height=height, width=width, chunk_size=effective))
        return cls(
            chunk_size=int(chunk_size),
            grid_shape=(int(height), int(width)),
            windows=windows,
            effective_chunk_size=effective if effective != chunk_size else None,
        )

    def __iter__(self) -> Iterator[ChunkWindow]:
        return iter(self.windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, chunk_id: int) -> ChunkWindow:
        return self.windows[chunk_id]

    @property
    def applied_chunk_size(self) -> int:
        return (
            self.effective_chunk_size
            if self.effective_chunk_size is not None
            else self.chunk_size
        )


def _generate_windows(
    *,
    height: int,
    width: int,
    chunk_size: int,
) -> Iterator[ChunkWindow]:
    if height <= 0 or width <= 0:
        return
    chunk_id = 0
    row = 0
    while row < height:
        h = min(chunk_size, height - row)
        col = 0
        while col < width:
            w = min(chunk_size, width - col)
            yield ChunkWindow(
                chunk_id=chunk_id,
                row_off=row,
                col_off=col,
                height=h,
                width=w,
            )
            chunk_id += 1
            col += chunk_size
        row += chunk_size


def _snap_chunk_size(chunk_size: int, *, block_size: Optional[int]) -> int:
    if block_size is None:
        return int(chunk_size)
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    if chunk_size <= block:
        return block
    return (int(chunk_size) // block) * block


def chunk_grid(grid: GridSpec, window: ChunkWindow) -> GridSpec:
    """Build a `GridSpec` covering a single chunk in the parent grid's CRS."""

    xmin, _ymin, _xmax, ymax = grid.bounds
    resolution = grid.resolution
    chunk_xmin = xmin + window.col_off * resolution
    chunk_xmax = chunk_xmin + window.width * resolution
    chunk_ymax = ymax - window.row_off * resolution
    chunk_ymin = chunk_ymax - window.height * resolution
    return GridSpec(
        bounds=(chunk_xmin, chunk_ymin, chunk_xmax, chunk_ymax),
        crs=grid.crs,
        resolution=resolution,
        width=window.width,
        height=window.height,
        wgs84_bounds=None,
    )
