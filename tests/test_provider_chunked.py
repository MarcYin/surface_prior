import numpy as np

from surface_priors.chunks import ChunkLayout
from surface_priors.provider import Provider, ProviderConfig
from surface_priors.selection import SceneChunkStats, SelectionPolicy
from surface_priors.types import Observation


class _FakeChunkedSource:
    """Minimal in-memory chunked source for Provider integration tests."""

    def __init__(self, *, observations_by_scene, stats):
        self._observations_by_scene = observations_by_scene
        self._stats = stats
        self.scout_calls = 0
        self.fetch_calls = []
        self.name = "fake-chunked"

    def block_size(self, *, grid, band_names):
        return None

    def scout(self, *, grid, layout, band_names, temporal_filter=None):
        self.scout_calls += 1
        if temporal_filter is None:
            return self._stats
        # Only emit stats for scenes whose timestamp lands within the filter.
        start, end = temporal_filter
        keep = self._scene_filter(start, end)
        return tuple(entry for entry in self._stats if entry.scene_index in keep)

    def _scene_filter(self, start, end):
        # Test sources can override this hook to declare per-scene datetimes.
        return set(range(len(self._observations_by_scene)))

    def fetch_selected(self, *, grid, plan, band_names, scene_index, chunk_id):
        self.fetch_calls.append((scene_index, chunk_id))
        observation_grid = self._observations_by_scene[scene_index]
        window = plan.layout[chunk_id]
        data = observation_grid["data"][
            :, window.row_slice, window.col_slice
        ].astype("float32", copy=True)
        quality = observation_grid["quality"][window.row_slice, window.col_slice].astype(
            "uint16", copy=True
        )
        return Observation(
            data=data,
            quality=quality,
            band_names=band_names,
        )


def test_provider_runs_chunked_pipeline_end_to_end(tmp_path):
    bands = ("iso",)

    full_data = np.full((1, 4, 4), 0.10, dtype="float32")
    full_quality = np.full((4, 4), 4, dtype="uint16")
    scene_0 = {"data": full_data, "quality": full_quality}

    full_data_2 = np.full((1, 4, 4), 0.20, dtype="float32")
    full_quality_2 = np.full((4, 4), 0, dtype="uint16")  # best quality
    scene_1 = {"data": full_data_2, "quality": full_quality_2}

    stats = [
        SceneChunkStats(scene_index=scene, chunk_id=chunk, usable_fraction=1.0, mean_clear=0.8)
        for scene in (0, 1)
        for chunk in range(4)
    ]
    source = _FakeChunkedSource(
        observations_by_scene={0: scene_0, 1: scene_1},
        stats=stats,
    )

    provider = Provider(
        ProviderConfig(
            cache_dir=tmp_path / "cache",
            source=source,
            chunk_size=2,
            selection_policy=SelectionPolicy(top_k=2, min_usable_fraction=0.5),
            fetch_workers=1,
        )
    )

    product = provider.build_prior(
        wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.25,
        product_id="fake-chunked-prior",
        native_crs="EPSG:4326",
        band_names=bands,
    )

    # Scene 1 has the best quality everywhere so it must win every chunk.
    np.testing.assert_allclose(
        product.composite.data[0], np.full((4, 4), 0.20, dtype="float32")
    )
    assert np.all(product.composite.selected_observation == 1)
    assert product.composite.attrs["compositor"] == "chunked_best_pixel_v2"
    assert product.composite.attrs["chunk_size"] == 2
    assert source.scout_calls == 1
    # Two scenes x four chunks each were selected, so fetch should have been called eight times.
    assert len(source.fetch_calls) == 8
    assert "chunking" in product.request


def test_chunked_request_hash_changes_with_policy(tmp_path):
    bands = ("iso",)
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=chunk, usable_fraction=1.0, mean_clear=0.8)
        for chunk in range(4)
    ]
    obs_grid = {
        "data": np.full((1, 4, 4), 0.1, dtype="float32"),
        "quality": np.zeros((4, 4), dtype="uint16"),
    }
    source = _FakeChunkedSource(observations_by_scene={0: obs_grid}, stats=stats)

    provider_a = Provider(
        ProviderConfig(
            cache_dir=tmp_path / "a",
            source=source,
            chunk_size=2,
            selection_policy=SelectionPolicy(top_k=1, min_usable_fraction=0.5),
        )
    )
    provider_b = Provider(
        ProviderConfig(
            cache_dir=tmp_path / "b",
            source=source,
            chunk_size=2,
            selection_policy=SelectionPolicy(top_k=3, min_usable_fraction=0.5),
        )
    )

    args = {
        "wgs84_bounds": (0.0, 0.0, 1.0, 1.0),
        "resolution": 0.25,
        "product_id": "fake",
        "native_crs": "EPSG:4326",
        "band_names": bands,
    }
    assert provider_a.request_hash(**args) != provider_b.request_hash(**args)


def test_chunked_pipeline_skips_chunks_below_usable_floor(tmp_path):
    bands = ("iso",)
    obs_grid = {
        "data": np.full((1, 4, 4), 0.30, dtype="float32"),
        "quality": np.zeros((4, 4), dtype="uint16"),
    }
    # Only chunk 0 passes the usable fraction floor; chunks 1..3 are too cloudy/sparse.
    stats = [
        SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=0.9, mean_clear=0.9),
        SceneChunkStats(scene_index=0, chunk_id=1, usable_fraction=0.1, mean_clear=0.9),
        SceneChunkStats(scene_index=0, chunk_id=2, usable_fraction=0.0, mean_clear=float("nan")),
        SceneChunkStats(scene_index=0, chunk_id=3, usable_fraction=0.0, mean_clear=float("nan")),
    ]
    source = _FakeChunkedSource(observations_by_scene={0: obs_grid}, stats=stats)

    provider = Provider(
        ProviderConfig(
            cache_dir=tmp_path / "cache",
            source=source,
            chunk_size=2,
            selection_policy=SelectionPolicy(top_k=1, min_usable_fraction=0.5),
        )
    )

    product = provider.build_prior(
        wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.25,
        product_id="cloudy",
        native_crs="EPSG:4326",
        band_names=bands,
    )

    assert source.fetch_calls == [(0, 0)]
    np.testing.assert_allclose(product.composite.data[0, :2, :2], 0.30)
    assert np.isnan(product.composite.data[0, :2, 2:]).all()
    assert np.isnan(product.composite.data[0, 2:, :]).all()
    assert product.composite.attrs["empty_chunk_count"] == 3


def test_layout_uses_block_size_when_source_advertises_one(tmp_path):
    """A non-None block_size from the source must override chunk_size in the layout."""

    class _BlockedSource(_FakeChunkedSource):
        def block_size(self, *, grid, band_names):
            return 4

    bands = ("iso",)
    obs_grid = {
        "data": np.full((1, 4, 4), 0.5, dtype="float32"),
        "quality": np.zeros((4, 4), dtype="uint16"),
    }
    stats = [SceneChunkStats(scene_index=0, chunk_id=0, usable_fraction=1.0, mean_clear=1.0)]
    source = _BlockedSource(observations_by_scene={0: obs_grid}, stats=stats)

    provider = Provider(
        ProviderConfig(
            cache_dir=tmp_path / "cache",
            source=source,
            chunk_size=2,
            selection_policy=SelectionPolicy(top_k=1, min_usable_fraction=0.5),
        )
    )
    product = provider.build_prior(
        wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.25,
        product_id="blocked",
        native_crs="EPSG:4326",
        band_names=bands,
    )

    # block_size=4 promotes the chunk to 4, so we get a single chunk and one fetch.
    assert product.composite.attrs["chunk_size"] == 4
    assert source.fetch_calls == [(0, 0)]


def test_temporal_filter_amortizes_scene_listing_across_months(tmp_path):
    """Three monthly builds against one chunked source should share scout
    bookkeeping but produce distinct request hashes and only fetch scenes
    inside the per-build temporal filter."""

    bands = ("iso",)

    class _DatedFakeSource(_FakeChunkedSource):
        def __init__(self):
            obs = {}
            for index in range(6):
                obs[index] = {
                    "data": np.full((1, 4, 4), 0.10 + 0.01 * index, dtype="float32"),
                    "quality": np.full((4, 4), 4, dtype="uint16"),
                }
            stats = []
            self.scene_dates = {
                0: "2024-07-05",
                1: "2024-07-25",
                2: "2024-08-08",
                3: "2024-08-22",
                4: "2024-09-04",
                5: "2024-09-19",
            }
            for scene in range(6):
                for chunk in range(4):
                    stats.append(
                        SceneChunkStats(
                            scene_index=scene,
                            chunk_id=chunk,
                            usable_fraction=1.0,
                            mean_clear=0.9 - 0.01 * scene,
                        )
                    )
            super().__init__(observations_by_scene=obs, stats=stats)
            self.list_calls = 0

        def _scene_filter(self, start, end):
            return {
                index
                for index, date in self.scene_dates.items()
                if start[:10] <= date <= end[:10]
            }

    source = _DatedFakeSource()
    provider = Provider(
        ProviderConfig(
            cache_dir=tmp_path / "cache",
            source=source,
            chunk_size=2,
            selection_policy=SelectionPolicy(top_k=2, min_usable_fraction=0.5),
            fetch_workers=1,
        )
    )

    months = [
        ("2024-07-01", "2024-07-31"),
        ("2024-08-01", "2024-08-31"),
        ("2024-09-01", "2024-09-30"),
    ]
    hashes = []
    fetched_scenes_per_month = []
    for start, end in months:
        before = len(source.fetch_calls)
        product = provider.build_prior(
            wgs84_bounds=(0.0, 0.0, 1.0, 1.0),
            resolution=0.25,
            product_id=f"monthly-{start[:7]}",
            native_crs="EPSG:4326",
            band_names=bands,
            temporal_filter=(start, end),
            composite_period=start[:7],
        )
        hashes.append(product.request["request_hash"])
        new_calls = source.fetch_calls[before:]
        fetched_scenes_per_month.append(sorted({scene for scene, _ in new_calls}))

    # Distinct cache directories per month.
    assert len(set(hashes)) == 3

    # Each build only fetched scenes from its own month.
    assert fetched_scenes_per_month[0] == [0, 1]
    assert fetched_scenes_per_month[1] == [2, 3]
    assert fetched_scenes_per_month[2] == [4, 5]


def test_layout_iteration_independent_of_provider():
    grid_layout = ChunkLayout.from_grid(
        # The Provider creates a layout internally; this test just confirms the
        # tiling math the integration above relies on.
        __import__("surface_priors").GridSpec.from_bounds((0, 0, 4, 4), "EPSG:4326", 1),
        chunk_size=2,
    )
    assert len(grid_layout) == 4
