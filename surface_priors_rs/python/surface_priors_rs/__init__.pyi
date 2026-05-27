"""Type stubs for the surface_priors_rs native extension."""

from typing import Any, Optional, Sequence

def build_composite(
    bbox: Sequence[float],
    datetime: str,
    resolution: float = 60.0,
    top_k: int = 3,
    max_cloud_cover: float = 90.0,
    concurrency: int = 600,
    endpoint: str = "auto",
    disk_cache: Optional[str] = None,
    scout_factor: int = 8,
    bands: Optional[Sequence[str]] = None,
    output_crs: str = "native",
    coverage_target: float = 0.0,
    min_k: int = 2,
    max_k: int = 8,
    windowed_fetch: bool = False,
) -> dict[str, Any]:
    """Build one best-pixel composite over ``bbox`` for the ``datetime`` range.

    Returns a dict with per-band ``numpy`` arrays plus ``observation_count``,
    ``selected_observation``, ``quality``, ``source_ids``, ``grid``,
    ``timings``, and tile-partition diagnostics.

    Scene selection:
      - ``top_k`` (default): fixed scenes-per-tile-per-chunk. Tile-aware when
        the AOI spans an MGRS seam.
      - Adaptive depth: set ``coverage_target`` in (0, 1) to decide k *per
        chunk* from the SCL. Scout keeps a coarse clear/observed mask per
        (scene, chunk); selection stacks scenes by greatest marginal clear-cell
        gain until their union covers ``coverage_target`` of the chunk's
        reachable area, with a ``min_k`` floor (best-pixel redundancy) and
        ``max_k`` cap. Cloudy chunks pull more depth; clear ones stop early.

    Fetch:
      - ``windowed_fetch`` (adaptive only): read each selected scene over just
        the bounding window of the chunks that requested it, instead of the
        whole grid. Cuts bytes fetched sharply when depth is concentrated
        (e.g. an under-observed swath-edge corner). ``timings['read_megapixels']``
        reports total source pixels read.

    ``endpoint``: ``"auto"`` | ``"earth-search"`` | ``"pc"`` | ``"hls"`` |
    ``"mcd43a4"``. ``output_crs``: ``"native"`` | ``"utm"``.
    """
    ...

def build_monthly_composites(
    bbox: Sequence[float],
    years: Sequence[int],
    months: Sequence[int],
    resolution: float = 60.0,
    top_k: int = 3,
    max_cloud_cover: float = 90.0,
    concurrency: int = 600,
    endpoint: str = "auto",
    disk_cache: Optional[str] = None,
    scout_factor: int = 8,
    bands: Optional[Sequence[str]] = None,
    output_crs: str = "native",
    coverage_target: float = 0.0,
    min_k: int = 2,
    max_k: int = 8,
    windowed_fetch: bool = False,
) -> list[dict[str, Any]]:
    """Build one composite per (year, month) in a single batch.

    Shares one scout pass and overlaps the per-period STAC searches, then
    composes each period sequentially. Returns a list of per-period result
    dicts shaped like :func:`build_composite`'s, each tagged with ``year`` and
    ``month``. All selection/fetch params behave as in :func:`build_composite`.
    """
    ...
