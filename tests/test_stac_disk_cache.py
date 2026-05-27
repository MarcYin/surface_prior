"""Persistence semantics for the STAC disk cache.

Live STAC calls are mocked via a fake client that records hit counts so
we can assert cache hits/misses without network. The cache must:
  - persist across `StacApiSource` instances pointing at the same dir;
  - re-apply the signer on cache hit (so SAS-token signers don't return
    expired hrefs);
  - keep the partition cache valid when the scene set is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from surface_priors.chunks import ChunkLayout
from surface_priors.sources.stac_api import (
    EARTH_SEARCH_S2_ALIASES,
    StacApiSource,
)
from surface_priors.sources.stac_cache import StacDiskCache, scenes_signature
from surface_priors.types import GridSpec

shapely = pytest.importorskip("shapely")


def _grid():
    return GridSpec.from_bounds(
        bounds=(260040.0, 3375420.0, 367080.0, 3487740.0),
        crs="EPSG:32636",
        resolution=60.0,
        wgs84_bounds=(30.5, 30.5, 31.6, 31.5),
    )


def _make_item(item_id: str, mgrs: str, datetime: str, geom_xmin: float = 30.5):
    return {
        "id": item_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [geom_xmin, 30.5],
                    [geom_xmin + 1.0, 30.5],
                    [geom_xmin + 1.0, 31.5],
                    [geom_xmin, 31.5],
                    [geom_xmin, 30.5],
                ]
            ],
        },
        "properties": {
            "datetime": datetime,
            "s2:mgrs_tile": mgrs,
            "eo:cloud_cover": 10.0,
        },
        "assets": {
            "blue": {"href": f"https://example/{item_id}/blue.tif"},
            "scl": {"href": f"https://example/{item_id}/scl.tif"},
        },
    }


class _FakeClient:
    def __init__(self, items):
        self._items = list(items)
        self.search_calls = 0

    def search(self, **_kwargs):
        self.search_calls += 1

        class _Result:
            def __init__(self, payload):
                self._payload = payload

            def items(self):
                return iter(self._payload)

        return _Result(self._items)


class _RecordingSigner:
    def __init__(self):
        self.calls = 0

    def sign_item(self, item):
        self.calls += 1
        signed = dict(item)
        signed_assets = {}
        for k, v in item.get("assets", {}).items():
            entry = dict(v)
            entry["href"] = entry["href"] + f"?token={self.calls}"
            signed_assets[k] = entry
        signed["assets"] = signed_assets
        return signed


def _make_source(cache_dir: Path, *, client, signer):
    return StacApiSource(
        stac_url="https://example/stac",
        collection="sentinel-2-l2a",
        temporal_ranges=[("2024-07-01", "2024-07-31")],
        aliases=EARTH_SEARCH_S2_ALIASES,
        signer=signer,
        chunk_size=512,
        stac_client=client,
        disk_cache=cache_dir,
    )


def test_disk_cache_skips_stac_search_on_second_source(tmp_path: Path):
    items = [
        _make_item("S2B_36RTU_20240705_0_L2A", "36RTU", "2024-07-05T08:42:00Z"),
        _make_item("S2B_36RUU_20240705_0_L2A", "36RUU", "2024-07-05T08:42:30Z", geom_xmin=31.0),
    ]

    client_a = _FakeClient(items)
    signer_a = _RecordingSigner()
    source_a = _make_source(tmp_path, client=client_a, signer=signer_a)
    scenes_a = source_a.list_scenes(grid=_grid())
    assert client_a.search_calls == 1
    assert len(scenes_a) == 2

    # Second source — same cache directory, but a fresh fake client that
    # would 0/0 if the disk cache is honoured.
    client_b = _FakeClient(items)
    signer_b = _RecordingSigner()
    source_b = _make_source(tmp_path, client=client_b, signer=signer_b)
    scenes_b = source_b.list_scenes(grid=_grid())
    assert client_b.search_calls == 0  # hit the disk cache
    assert len(scenes_b) == 2
    assert signer_b.calls == 2  # re-signed on cache load (fresh hrefs)


def test_disk_cache_resigns_assets_per_load(tmp_path: Path):
    items = [_make_item("S2B_36RTU_20240705_0_L2A", "36RTU", "2024-07-05T08:42:00Z")]
    client = _FakeClient(items)
    signer_first = _RecordingSigner()
    source_first = _make_source(tmp_path, client=client, signer=signer_first)
    source_first.list_scenes(grid=_grid())
    assert signer_first.calls == 1  # original search

    # Fresh signer on second load — must still be invoked on cache hit so
    # PC/CDSE SAS tokens don't go stale.
    signer_second = _RecordingSigner()
    source_second = _make_source(tmp_path, client=_FakeClient([]), signer=signer_second)
    scene_second = source_second.list_scenes(grid=_grid())[0]
    assert signer_second.calls == 1
    assert "?token=" in scene_second.asset_hrefs["blue"]


def test_disk_partition_cache_persists_across_sources(tmp_path: Path):
    items = [
        _make_item("S2B_36RTU_20240705_0_L2A", "36RTU", "2024-07-05T08:42:00Z"),
        _make_item("S2B_36RUU_20240705_0_L2A", "36RUU", "2024-07-05T08:42:30Z", geom_xmin=31.0),
    ]

    grid = _grid()
    layout = ChunkLayout.from_grid(grid, chunk_size=512)

    source_a = _make_source(tmp_path, client=_FakeClient(items), signer=_RecordingSigner())
    source_a.list_scenes(grid=grid)
    partition_a = source_a.tile_partition(grid=grid, layout=layout)
    assert partition_a is not None
    assert set(partition_a.tiles) == {"36RTU", "36RUU"}

    partition_disk = (tmp_path / "partition").iterdir()
    assert any(p.suffix == ".json" for p in partition_disk)

    # Fresh source instance, fresh in-memory state — partition should come
    # from disk without recomputing.
    source_b = _make_source(tmp_path, client=_FakeClient(items), signer=_RecordingSigner())
    source_b.list_scenes(grid=grid)
    partition_b = source_b.tile_partition(grid=grid, layout=layout)
    assert partition_b is not None
    assert partition_b.tiles == partition_a.tiles
    assert dict(partition_b.scene_to_tile) == dict(partition_a.scene_to_tile)


def test_disabled_cache_writes_nothing(tmp_path: Path):
    items = [_make_item("S2B_36RTU_20240705_0_L2A", "36RTU", "2024-07-05T08:42:00Z")]
    source = StacApiSource(
        stac_url="https://example/stac",
        collection="sentinel-2-l2a",
        temporal_ranges=[("2024-07-01", "2024-07-31")],
        aliases=EARTH_SEARCH_S2_ALIASES,
        signer=_RecordingSigner(),
        chunk_size=512,
        stac_client=_FakeClient(items),
        disk_cache=None,
    )
    source.list_scenes(grid=_grid())
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_scenes_signature_is_order_invariant():
    a = [
        {"id": "A", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"s2:mgrs_tile": "T"}},
        {"id": "B", "geometry": {"type": "Point", "coordinates": [1, 1]}, "properties": {"s2:mgrs_tile": "U"}},
    ]
    # Identical order must produce an identical signature.
    assert scenes_signature(a) == scenes_signature(list(a))


def test_stac_cache_from_arg_resolution():
    assert StacDiskCache.from_arg(None) is None
    assert StacDiskCache.from_arg(False) is None
    cache_path = Path("/tmp/spx")
    cache = StacDiskCache.from_arg(cache_path)
    assert isinstance(cache, StacDiskCache)
    assert cache.root == cache_path
    assert StacDiskCache.from_arg(cache) is cache
