from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from surface_priors.chunks import ChunkLayout
from surface_priors.selection import SceneChunkStats, SelectionPolicy, select
from surface_priors.sources.s2_gee import (
    S2L2AGeeSource,
    _snap_bounds_to_resolution,
    _utm_crs_from_wgs84_bounds,
)
from surface_priors.types import GridSpec

# --- Pure helpers ---------------------------------------------------------


def test_utm_crs_from_wgs84_bounds_picks_correct_zone():
    # London bounds (zone 30 N).
    crs = _utm_crs_from_wgs84_bounds((-1.0, 51.0, 0.5, 52.0))
    assert crs == "EPSG:32630"


def test_utm_crs_handles_southern_hemisphere():
    # Sao Paulo (~zone 23 S).
    crs = _utm_crs_from_wgs84_bounds((-47.0, -24.0, -46.0, -23.0))
    assert crs == "EPSG:32723"


def test_snap_bounds_rounds_outwards():
    snapped = _snap_bounds_to_resolution((100.1, 200.9, 299.5, 401.2), resolution=10.0)
    assert snapped == (100.0, 200.0, 300.0, 410.0)


# --- Resolve grid ---------------------------------------------------------


def test_resolve_grid_returns_utm_grid_aligned_to_resolution():
    source = _make_source()

    grid = source.resolve_grid(
        wgs84_bounds=(-1.0, 51.0, -0.99, 51.01),
        native_crs="ignored",
        resolution=20.0,
        band_names=("s2_b04_red",),
    )

    assert grid.crs == "EPSG:32630"
    # Snapping to 20m is exact along both axes.
    for value in grid.bounds:
        assert value % 20.0 == 0.0
    assert grid.width > 0
    assert grid.height > 0
    assert grid.wgs84_bounds == (-1.0, 51.0, -0.99, 51.01)


# --- Scout and fetch with a fake ee module --------------------------------


@dataclass
class _FakeRecorder:
    """Captures requests issued to the fake ee.data.computePixels."""

    requests: list[dict] = None
    coarse_arrays: dict[str, np.ndarray] = None
    chunk_arrays: dict[str, np.ndarray] = None

    def __post_init__(self):
        self.requests = []
        self.coarse_arrays = {}
        self.chunk_arrays = {}


class _FakeImage:
    def __init__(self, image_id: str, recorder: _FakeRecorder, *, bands=()):
        self._image_id = image_id
        self._recorder = recorder
        self._bands = tuple(bands)
        self._is_coarse = False

    def select(self, band):
        if isinstance(band, str):
            return _FakeImage(self._image_id, self._recorder, bands=(band,))
        return _FakeImage(self._image_id, self._recorder, bands=tuple(band))

    def reduceResolution(self, **kwargs):
        new = _FakeImage(self._image_id, self._recorder, bands=self._bands)
        new._is_coarse = True
        return new

    def reproject(self, **kwargs):
        return self

    def addBands(self, other):
        combined = _FakeImage(
            self._image_id,
            self._recorder,
            bands=self._bands + tuple(other._bands),
        )
        combined._is_coarse = False
        return combined


class _FakeCollection:
    def __init__(self, collection_id: str, recorder: _FakeRecorder, *, scenes):
        self._collection_id = collection_id
        self._recorder = recorder
        self._scenes = scenes

    def filterBounds(self, bbox):
        return self

    def filterDate(self, start, end):
        return self

    def sort(self, key):
        return self

    def aggregate_array(self, key):
        if key == "system:index":
            return _FakeList([scene["system_index"] for scene in self._scenes])
        if key == "system:time_start":
            return _FakeList([scene["timestamp_ms"] for scene in self._scenes])
        raise KeyError(key)


class _FakeList:
    def __init__(self, values):
        self._values = list(values)

    def getInfo(self):
        return list(self._values)


class _FakeData:
    def __init__(self, recorder: _FakeRecorder):
        self._recorder = recorder

    def computePixels(self, request):
        self._recorder.requests.append(request)
        expression = request["expression"]
        bands = tuple(request["bandIds"])
        dimensions = request["grid"]["dimensions"]
        width = int(dimensions["width"])
        height = int(dimensions["height"])

        if expression._is_coarse:
            key = expression._image_id
            return self._recorder.coarse_arrays.get(
                key, np.full((height, width), 1.0, dtype="float32")
            )
        # Per-chunk SR + cs request.
        key = (expression._image_id, width, height)
        arrays = self._recorder.chunk_arrays.get(key)
        if arrays is None:
            dtype = np.dtype([(band, "float32") for band in bands])
            structured = np.zeros((height, width), dtype=dtype)
            for band in bands:
                if band == "cs":
                    structured[band] = 0.9
                else:
                    structured[band] = 0.3 / 0.0001  # so SR scale -> 0.3
            return structured
        return arrays


class _FakeNumber:
    def __init__(self, value):
        self._value = value

    def getInfo(self):
        return self._value


class _FakeReducer:
    @staticmethod
    def mean():
        return "mean"


class _FakeGeometry:
    @staticmethod
    def BBox(*args):
        return ("bbox", tuple(args))


class _FakeEE:
    def __init__(self, recorder: _FakeRecorder, *, scenes):
        self.data = _FakeData(recorder)
        self.Reducer = _FakeReducer
        self.Geometry = _FakeGeometry
        self._recorder = recorder
        self._scenes = scenes

    def Number(self, value):
        return _FakeNumber(value)

    def Image(self, image_id):
        return _FakeImage(image_id, self._recorder)

    def ImageCollection(self, collection_id):
        return _FakeCollection(collection_id, self._recorder, scenes=self._scenes)

    def Initialize(self, *args, **kwargs):
        return None


def _make_source(*, scenes=None, recorder=None, scout_factor=2):
    scenes = scenes or [
        {"system_index": "20240715T103631_T31UCP", "timestamp_ms": 1_000_000_000_000},
    ]
    rec = recorder or _FakeRecorder()
    fake_ee = _FakeEE(rec, scenes=scenes)
    return S2L2AGeeSource(
        temporal_ranges=(("2024-07-01", "2024-07-31"),),
        chunk_size=2,
        scout_factor=scout_factor,
        ee_module=fake_ee,
    )


def test_scout_returns_one_entry_per_scene_chunk():
    grid = GridSpec.from_bounds((0, 0, 4, 4), "EPSG:32630", 1, wgs84_bounds=(0, 0, 4, 4))
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    recorder = _FakeRecorder()
    coarse_score = np.array(
        [[0.95, 0.95], [0.10, 0.95]],
        dtype="float32",
    )
    recorder.coarse_arrays = {
        "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED/20240715T103631_T31UCP": coarse_score,
    }
    source = _make_source(recorder=recorder)

    stats = source.scout(grid=grid, layout=layout, band_names=("s2_b04_red",))

    assert {entry.chunk_id for entry in stats} == {0, 1, 2, 3}
    # Chunk 2 (bottom-left) covers coarse rows [1:2], col [0:1] => score 0.10.
    chunk2 = next(entry for entry in stats if entry.chunk_id == 2)
    assert chunk2.mean_clear == pytest.approx(0.10, rel=1e-5)
    assert chunk2.usable_fraction == 1.0
    chunk0 = next(entry for entry in stats if entry.chunk_id == 0)
    assert chunk0.mean_clear == pytest.approx(0.95, rel=1e-5)


def test_fetch_selected_returns_observation_with_quality_from_cs():
    grid = GridSpec.from_bounds((0, 0, 4, 4), "EPSG:32630", 1, wgs84_bounds=(0, 0, 4, 4))
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    source = _make_source()

    stats = [
        SceneChunkStats(scene_index=0, chunk_id=cid, usable_fraction=1.0, mean_clear=0.9)
        for cid in range(4)
    ]
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=1))

    observation = source.fetch_selected(
        grid=grid,
        plan=plan,
        band_names=("s2_b04_red",),
        scene_index=0,
        chunk_id=0,
    )

    assert observation is not None
    assert observation.data.shape == (1, 2, 2)
    # SR scale should multiply 3000 (raw) by 0.0001 -> 0.3.
    assert np.allclose(observation.data, 0.3)
    # cs=0.9 → quality = (1-0.9)*10000 = 1000.
    assert int(observation.quality[0, 0]) == 1000
    assert observation.source_id == "20240715T103631_T31UCP"


def test_scene_listing_caches_within_run():
    source = _make_source()
    grid = GridSpec.from_bounds((0, 0, 4, 4), "EPSG:32630", 1, wgs84_bounds=(0, 0, 4, 4))

    first = source.list_scenes(grid=grid)
    second = source.list_scenes(grid=grid)

    assert first is second
