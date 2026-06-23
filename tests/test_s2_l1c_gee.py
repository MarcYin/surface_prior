from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from surface_priors.atmosphere import AtmoSidecar, SceneAtmosphere, correct_reflectance
from surface_priors.chunks import ChunkLayout
from surface_priors.selection import SceneChunkStats, SelectionPolicy, select
from surface_priors.sources.s2_l1c_gee import S2L1CGeeSource
from surface_priors.types import GridSpec


def _sidecar():
    def scene(aod):
        return SceneAtmosphere(
            maiac_aod=aod, wvp=2.0, tcwv_nodes=(2.0,),
            xap=((1.27, 1.1, 1.0),), xbp=((0.08, 0.05, 0.04),), xcp=((0.145, 0.12, 0.10),),
        )
    return AtmoSidecar(
        bands=("blue", "green", "red"),
        scenes={
            "S2A_36RUU_20220717_0_L1C": scene(0.12),   # clean
            "S2A_36RTU_20220720_0_L1C": scene(0.30),   # hazy -> gated out at frac 0.6
        },
    )


# --- fake earthengine-api -------------------------------------------------


@dataclass
class _Rec:
    coarse: dict = field(default_factory=dict)


class _Img:
    def __init__(self, image_id, rec, bands=()):
        self.image_id = image_id
        self.rec = rec
        self.bands = tuple(bands)
        self.coarse = False

    def select(self, band):
        return _Img(self.image_id, self.rec, (band,) if isinstance(band, str) else tuple(band))

    def reduceResolution(self, **k):
        n = _Img(self.image_id, self.rec, self.bands)
        n.coarse = True
        return n

    def reproject(self, **k):
        return self


class _Coll:
    def __init__(self, scenes):
        self._scenes = scenes

    def filterBounds(self, *_):
        return self

    def filterDate(self, *_):
        return self

    def sort(self, *_):
        return self

    def aggregate_array(self, key):
        field = "system_index" if key == "system:index" else "timestamp_ms"
        return _List([s[field] for s in self._scenes])


class _List:
    def __init__(self, v):
        self._v = list(v)

    def getInfo(self):
        return list(self._v)


class _Data:
    def __init__(self, rec):
        self.rec = rec

    def computePixels(self, request):  # coarse cs (expression path)
        expr = request["expression"]
        d = request["grid"]["dimensions"]
        return self.rec.coarse.get(expr.image_id, np.full((d["height"], d["width"]), 0.9, "float32"))

    def getPixels(self, request):  # raw TOA / cs (assetId path)
        bands = request["bandIds"]
        d = request["grid"]["dimensions"]
        h, w = int(d["height"]), int(d["width"])
        out = np.zeros((h, w), dtype=np.dtype([(b, "float32") for b in bands]))
        for b in bands:
            out[b] = 0.9 if b == "cs" else 3000.0   # cs clear; TOA DN -> *1e-4 = 0.3
        return out


class _EE:
    def __init__(self, rec, scenes):
        self.data = _Data(rec)
        self.Reducer = type("R", (), {"mean": staticmethod(lambda: "mean")})
        self.Geometry = type("G", (), {"BBox": staticmethod(lambda *a: ("bbox", a))})
        self._scenes = scenes

    def Number(self, v):
        return type("N", (), {"getInfo": lambda self: v})()

    def Image(self, image_id):
        return _Img(image_id, None)

    def ImageCollection(self, _):
        return _Coll(self._scenes)

    def Initialize(self, *a, **k):
        return None


GEE_SCENES = [
    {"system_index": "20220717T082611_20220717T083855_T36RUU", "timestamp_ms": 1_657_000_000_000},
    {"system_index": "20220720T082611_20220720T083855_T36RTU", "timestamp_ms": 1_658_000_000_000},
]


def _source(**kw):
    return S2L1CGeeSource(
        temporal_ranges=(("2022-07-01", "2022-08-01"),),
        atmosphere=_sidecar(),
        chunk_size=2,
        scout_factor=2,
        ee_module=_EE(_Rec(), GEE_SCENES),
        **kw,
    )


def _grid():
    return GridSpec.from_bounds((0, 0, 4, 4), "EPSG:32636", 1, wgs84_bounds=(0, 0, 4, 4))


def test_clean_day_gate_keeps_only_low_aod_scenes():
    scenes = _source(low_aod_frac=0.6).list_scenes(grid=_grid())
    # frac 0.6 of 2 sidecar scenes -> keep the 0.12 (36RUU) only; 0.30 gated out
    assert [s.system_index for s in scenes] == ["20220717T082611_20220717T083855_T36RUU"]
    assert scenes[0].maiac_aod == pytest.approx(0.12)


def test_fetch_selected_corrects_toa_and_ranks_by_aod():
    grid = _grid()
    layout = ChunkLayout.from_grid(grid, chunk_size=2)
    source = _source(low_aod_frac=0.6)
    stats = [SceneChunkStats(scene_index=0, chunk_id=c, usable_fraction=1.0, mean_clear=0.9)
             for c in range(4)]
    plan = select(layout=layout, stats=stats, policy=SelectionPolicy(top_k=1))

    obs = source.fetch_selected(
        grid=grid, plan=plan, band_names=("s2_b02_blue",), scene_index=0, chunk_id=0
    )
    assert obs is not None
    assert obs.data.shape == (1, 2, 2)
    # TOA 0.3 corrected by blue 6S coeffs (xap1.27,xbp0.08,xcp0.145)
    expected = correct_reflectance(0.3, 1.27, 0.08, 0.145)
    assert np.allclose(obs.data, expected, atol=1e-5)
    # quality packs AOD (0.12) high, cs (0.9) low: 24*200 + round(0.1*199)=20
    assert int(obs.quality[0, 0]) == 24 * 200 + 20
    assert obs.source_id == "20220717T082611_20220717T083855_T36RUU"
