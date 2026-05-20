from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from surface_priors.chunks import ChunkLayout
from surface_priors.selection import SceneChunkStats, SelectionPolicy, select
from surface_priors.sources.stac_api import (
    EARTH_SEARCH_S2_ALIASES,
    PLANETARY_COMPUTER_S2_ALIASES,
    NoOpSigner,
    StacApiSource,
    _snap_bounds_to_resolution,
    _utm_crs_from_wgs84_bounds,
)
from surface_priors.types import GridSpec

rasterio = pytest.importorskip("rasterio")


# --- Lightweight fakes ----------------------------------------------------


class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeStacClient:
    def __init__(self, items):
        self._items = items
        self.search_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return _FakeSearch(self._items)


def _write_geotiff(path: Path, array: np.ndarray, *, transform, crs: str) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)


def _make_item(scene_id: str, assets: dict, datetime: str = "2024-07-15T10:00:00Z") -> dict:
    return {
        "id": scene_id,
        "properties": {"datetime": datetime},
        "assets": {key: {"href": str(path)} for key, path in assets.items()},
    }


# --- Pure helpers ---------------------------------------------------------


def test_utm_crs_from_wgs84_bounds_picks_zone():
    assert _utm_crs_from_wgs84_bounds((-1.0, 51.0, 0.5, 52.0)) == "EPSG:32630"


def test_snap_bounds_rounds_outwards():
    assert _snap_bounds_to_resolution((100.1, 200.9, 299.5, 401.2), resolution=10.0) == (
        100.0,
        200.0,
        300.0,
        410.0,
    )


def test_quality_asset_prefers_cs_when_present():
    assert EARTH_SEARCH_S2_ALIASES.quality_asset() == "scl"
    assert PLANETARY_COMPUTER_S2_ALIASES.uses_cloud_score_plus() is False


# --- Source-level scout and fetch via real local GeoTIFFs -----------------


@pytest.fixture
def scene_geotiffs(tmp_path):
    crs = "EPSG:32630"
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 40.0)
    red = np.full((4, 4), 3000, dtype="int16")  # 3000 * 0.0001 = 0.3
    scl = np.array(
        [
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [8, 8, 8, 8],
            [8, 8, 8, 8],
        ],
        dtype="uint8",
    )
    red_path = tmp_path / "red.tif"
    scl_path = tmp_path / "scl.tif"
    _write_geotiff(red_path, red, transform=transform, crs=crs)
    _write_geotiff(scl_path, scl, transform=transform, crs=crs)
    return red_path, scl_path


def _make_source(items, *, scout_factor=2, chunk_size=2):
    return StacApiSource(
        stac_url="https://example.invalid/stac",
        collection="sentinel-2-l2a",
        temporal_ranges=(("2024-07-01", "2024-07-31"),),
        aliases=EARTH_SEARCH_S2_ALIASES,
        signer=NoOpSigner(),
        scout_factor=scout_factor,
        chunk_size=chunk_size,
        scout_workers=1,
        stac_client=_FakeStacClient(items),
    )


def test_scout_uses_scl_to_compute_clear_fraction(scene_geotiffs):
    red_path, scl_path = scene_geotiffs
    item = _make_item("S2A_T31U_20240715", {"red": red_path, "scl": scl_path})
    source = _make_source([item])
    grid = GridSpec.from_bounds(
        (0.0, 0.0, 40.0, 40.0),
        crs="EPSG:32630",
        resolution=10.0,
        wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
    )
    layout = ChunkLayout.from_grid(grid, chunk_size=2)

    stats = source.scout(grid=grid, layout=layout, band_names=("s2_b04_red",))
    by_chunk = {entry.chunk_id: entry for entry in stats}

    # Top chunks should be clear (SCL=4), bottom chunks cloudy (SCL=8).
    assert by_chunk[0].usable_fraction == 1.0
    assert by_chunk[0].mean_clear == 1.0
    assert by_chunk[1].mean_clear == 1.0
    assert by_chunk[2].mean_clear == 0.0
    assert by_chunk[3].mean_clear == 0.0


def test_fetch_selected_returns_observation_with_scl_quality(scene_geotiffs):
    red_path, scl_path = scene_geotiffs
    item = _make_item("S2A_T31U_20240715", {"red": red_path, "scl": scl_path})
    source = _make_source([item])
    grid = GridSpec.from_bounds(
        (0.0, 0.0, 40.0, 40.0),
        crs="EPSG:32630",
        resolution=10.0,
        wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
    )
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=cid, usable_fraction=1.0, mean_clear=1.0)
        for cid in range(len(layout))
    ]
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=1))

    chunk_top_left = source.fetch_selected(
        grid=grid,
        plan=plan,
        band_names=("s2_b04_red",),
        scene_index=0,
        chunk_id=0,
    )
    chunk_bottom_left = source.fetch_selected(
        grid=grid,
        plan=plan,
        band_names=("s2_b04_red",),
        scene_index=0,
        chunk_id=2,
    )

    assert chunk_top_left is not None
    assert chunk_top_left.data.shape == (1, 2, 2)
    np.testing.assert_allclose(chunk_top_left.data, 0.3)
    # SCL=4 → quality=0 (clear).
    assert (chunk_top_left.quality == 0).all()

    assert chunk_bottom_left is not None
    # SCL=8 → quality=65535 (nodata).
    assert (chunk_bottom_left.quality == 65535).all()


def test_missing_band_asset_returns_none(scene_geotiffs):
    _, scl_path = scene_geotiffs
    # Item has SCL but no red band asset; fetch must short-circuit.
    item = _make_item("S2A_T31U_20240715", {"scl": scl_path})
    source = _make_source([item])
    grid = GridSpec.from_bounds(
        (0.0, 0.0, 40.0, 40.0),
        crs="EPSG:32630",
        resolution=10.0,
        wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
    )
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    stats = [SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=1.0, mean_clear=1.0)]
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=1))

    result = source.fetch_selected(
        grid=grid,
        plan=plan,
        band_names=("s2_b04_red",),
        scene_index=0,
        chunk_id=0,
    )

    assert result is None


# --- Provider integration -------------------------------------------------


def test_provider_runs_stac_source_end_to_end(scene_geotiffs, tmp_path):
    from surface_priors.provider import Provider, ProviderConfig

    red_path, scl_path = scene_geotiffs
    item = _make_item("S2A_T31U_20240715", {"red": red_path, "scl": scl_path})

    class _BoundedStacSource(StacApiSource):
        # Override resolve_grid so the Provider's pipeline uses the exact same
        # UTM grid the synthetic GeoTIFFs were written for.
        def resolve_grid(self, *, wgs84_bounds, native_crs, resolution, band_names):
            return GridSpec.from_bounds(
                (0.0, 0.0, 40.0, 40.0),
                crs="EPSG:32630",
                resolution=10.0,
                wgs84_bounds=wgs84_bounds,
            )

    source = _BoundedStacSource(
        stac_url="https://example.invalid/stac",
        collection="sentinel-2-l2a",
        temporal_ranges=(("2024-07-01", "2024-07-31"),),
        aliases=EARTH_SEARCH_S2_ALIASES,
        signer=NoOpSigner(),
        scout_factor=2,
        chunk_size=2,
        scout_workers=1,
        stac_client=_FakeStacClient([item]),
    )

    provider = Provider(
        ProviderConfig(
            cache_dir=tmp_path / "cache",
            source=source,
            chunk_size=2,
            selection_policy=SelectionPolicy(top_k=1, min_usable_fraction=0.5),
            fetch_workers=1,
        )
    )

    product = provider.build_prior(
        wgs84_bounds=(-1.0, 51.0, -0.99, 51.01),
        resolution=10.0,
        product_id="stac-test",
        band_names=("s2_b04_red",),
    )

    composite = product.composite
    # Top two rows: SCL=4 (clear), pixel value 0.3.
    np.testing.assert_allclose(composite.data[0, :2, :], 0.3)
    # Bottom two rows: SCL=8 → quality=65535 → compositor drops them as nodata.
    assert np.isnan(composite.data[0, 2:, :]).all()
    np.testing.assert_array_equal(composite.observation_count[2:, :], 0)
    np.testing.assert_array_equal(composite.observation_count[:2, :], 1)
    assert composite.attrs["chunk_size"] == 2
