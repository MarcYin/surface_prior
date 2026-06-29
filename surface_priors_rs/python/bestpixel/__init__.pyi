"""Type stubs for the surface_priors_rs native extension."""

from typing import Any, Optional, Sequence

def build_composite(
    bbox: Sequence[float],
    datetime: str,
    resolution: float = 60.0,
    top_k: int = 3,
    max_cloud_cover: float = 90.0,
    concurrency: int = 192,
    endpoint: str = "auto",
    disk_cache: Optional[str] = None,
    scout_factor: int = 8,
    bands: Optional[Sequence[str]] = None,
    output_crs: str = "native",
    coverage_target: float = 0.0,
    min_k: int = 2,
    max_k: int = 8,
    windowed_fetch: bool = False,
    emit_uncertainty: bool = False,
) -> dict[str, Any]:
    """Build one best-pixel composite over ``bbox`` for the ``datetime`` range.

    When ``emit_uncertainty`` is set, the result also carries ``boa_unc``: a
    dict ``{band_name: (H, W) float32}`` of per-band temporal-spread uncertainty
    (reflectance DN, same scale as ``bands``; NaN where the pixel is nodata).

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
    concurrency: int = 192,
    endpoint: str = "auto",
    disk_cache: Optional[str] = None,
    scout_factor: int = 8,
    bands: Optional[Sequence[str]] = None,
    output_crs: str = "native",
    coverage_target: float = 0.0,
    min_k: int = 2,
    max_k: int = 8,
    windowed_fetch: bool = False,
    aod_by_day: Optional[dict[str, float]] = None,
    aod_max: Optional[float] = None,
    reject_unknown: bool = False,
    emit_uncertainty: bool = False,
) -> list[dict[str, Any]]:
    """Build one composite per (year, month) in a single batch.

    Shares one scout pass and overlaps the per-period STAC searches, then
    composes each period sequentially. Returns a list of per-period result
    dicts shaped like :func:`build_composite`'s, each tagged with ``year`` and
    ``month``. All selection/fetch params behave as in :func:`build_composite`.

    Optional external-aerosol (e.g. MAIAC) day gate: pass ``aod_by_day`` mapping
    ``"YYYY-MM-DD"`` acquisition days to an AOD value plus an ``aod_max``
    threshold, and scenes whose day exceeds the threshold are dropped *before*
    best-pixel selection/compositing — so the composite is built only from
    low-AOD (atmospherically clean) days. Both must be given to activate the
    gate; days absent from the map are kept (unknown is not treated as dirty)
    unless ``reject_unknown=True``, which instead DROPS any day missing from the
    map (keep only days with a vouched-for AOD). Each period's ``timings`` then
    carries an ``aod_rejected`` count.

    ``emit_uncertainty=True`` adds a per-band ``boa_unc`` dict to each period
    result (same shape/scale as ``bands``; see :func:`build_composite`).
    """
    ...

def correct_toa(
    toa: Any,
    xap: Sequence[float],
    xbp: Sequence[float],
    xcp: Sequence[float],
) -> Any:
    """6S-correct a TOA reflectance stack to surface reflectance (Rust core).

    ``toa`` is a ``(band, H, W)`` float32 numpy array of TOA reflectance (0..1);
    ``xap``/``xbp``/``xcp`` give one 6S coefficient per band (already
    interpolated to the scene's water vapour). Returns a ``(band, H, W)`` float32
    array: ``rho_boa = y / (1 + xcp*y)``, ``y = xap*rho_toa - xbp``; negatives
    clamp to 0, non-finite (nodata) passes through.
    """
    ...

def build_l1c_composite(
    bbox: Sequence[float],
    datetime: tuple[str, str],
    sidecar: str,
    *,
    resolution: float = 60.0,
    epsg: Optional[int] = None,
    bands: Optional[Sequence[str]] = None,
    low_aod_frac: float = 0.6,
    cs_thresh: float = 0.6,
    rank: str = "aod",
    chunk: int = 1024,
    scout_factor: int = 8,
    coverage_target: float = 0.98,
    min_k: int = 2,
    max_k: int = 8,
    workers: int = 16,
    out: Optional[str] = None,
    ee_module: Optional[Any] = None,
) -> dict[str, Any]:
    """Sentinel-2 L1C custom-AC monthly composite (requires the ``gee`` extra).

    MAIAC-selects the lowest-AOD ``low_aod_frac`` of the sidecar's clean days,
    scout-first fetches raw L1C TOA + Cloud Score+ from GEE (only the patches
    that win a chunk), 6S-corrects each scene via :func:`correct_toa`, and
    best-pixel composites preferring the lowest-AOD clear pixel
    (``rank="aod"``; ``cs`` tie-break) or the clearest (``rank="cs"``).

    Returns ``{"bands": {name: (H,W) float32}, "grid": {...}, "count": (H,W),
    "scenes": [...]}`` and, when ``out`` is given, writes a scaled int16 GeoTIFF.
    """
    ...
