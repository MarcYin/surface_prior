"""Tests for the bestpixel L1C custom-AC pipeline (Rust correct_toa core +
pure-Python GEE orchestration). Requires the installed bestpixel extension; the
GEE calls are driven by a fake `ee` module so no network/credentials are used.
"""
from __future__ import annotations

import json

import bestpixel as bp
import numpy as np
import pytest
from bestpixel.atmosphere import AtmoSidecar, correct_reflectance

# --- Rust correct_toa core ------------------------------------------------


def test_correct_toa_roundtrip_and_nodata():
    xap, xbp, xcp = 1.27, 0.08, 0.145
    boa = 0.2
    y = boa / (1.0 - xcp * boa)
    toa = (y + xbp) / xap
    arr = np.array([[[toa, np.nan, -5.0]]], dtype=np.float32)  # (1 band, 1, 3)
    out = np.asarray(bp.correct_toa(arr, [xap], [xbp], [xcp]))
    assert out[0, 0, 0] == pytest.approx(boa, abs=1e-5)
    assert np.isnan(out[0, 0, 1])          # nodata passes through
    assert out[0, 0, 2] == 0.0             # negative clamps to 0


def test_correct_toa_rejects_bad_coeff_length():
    arr = np.zeros((2, 1, 1), dtype=np.float32)
    with pytest.raises(RuntimeError):
        bp.correct_toa(arr, [1.0], [0.0], [0.0])   # 1 coeff, 2 bands


# --- atmosphere sidecar ---------------------------------------------------


def _sidecar_payload():
    base = {
        "maiac_aod": 0.0, "wvp": 2.0, "tcwv_nodes": [2.0],
        "xap": [[1.27, 1.1]], "xbp": [[0.08, 0.05]], "xcp": [[0.145, 0.12]],
    }
    scenes = {}
    for sid, aod in [("S2A_36RTU_20220720_0_L1C", 0.30),
                     ("S2A_36RUU_20220717_0_L1C", 0.12)]:
        scenes[sid] = {**base, "maiac_aod": aod}
    return {"bands": ["blue", "green"], "scenes": scenes}


def test_sidecar_select_and_lookup(tmp_path):
    p = tmp_path / "sc.json"
    p.write_text(json.dumps(_sidecar_payload()))
    sc = AtmoSidecar.load(p)
    assert sc.select_low_aod(0.5) == ["S2A_36RUU_20220717_0_L1C"]
    # GEE system:index resolves to the same scene by (tile, date)
    assert sc.lookup("20220717T08_x_T36RUU").maiac_aod == pytest.approx(0.12)


# --- end-to-end build_l1c_composite with a fake ee ------------------------


class _Img:
    def __init__(self, sysidx=None):
        self._sysidx = sysidx

    def get(self, key):
        return self._sysidx


class _Coll:
    def __init__(self):
        self._d0 = None
        self._tile = None

    def filterBounds(self, *_):
        return self

    def filterDate(self, d0, d1):
        self._d0 = d0
        return self

    def filter(self, f):
        self._tile = f[2]
        return self

    def first(self):
        ymd = str(self._d0).replace("-", "")
        return _Img(f"{ymd}T000000_{ymd}T001000_T{self._tile}")


class _List:
    def __init__(self, values):
        self._v = list(values)

    def getInfo(self):
        return list(self._v)


class _Data:
    def getPixels(self, request):
        bands = request["bandIds"]
        d = request["grid"]["dimensions"]
        h, w = int(d["height"]), int(d["width"])
        out = np.zeros((h, w), dtype=np.dtype([(b, "float32") for b in bands]))
        for b in bands:
            out[b] = 0.9 if b == "cs" else 3000.0   # cs clear; TOA DN -> 0.3
        return out


class _EE:
    data = _Data()
    Filter = type("F", (), {"eq": staticmethod(lambda field, val: ("eq", field, val))})
    Geometry = type("G", (), {"Rectangle": staticmethod(lambda b: ("rect", tuple(b)))})

    def Image(self, x):
        return x if isinstance(x, _Img) else _Img()

    def ImageCollection(self, _):
        return _Coll()

    def List(self, values):
        return _List(values)


def test_build_l1c_composite_end_to_end(tmp_path):
    p = tmp_path / "sc.json"
    p.write_text(json.dumps(_sidecar_payload()))
    out = bp.build_l1c_composite(
        bbox=(31.0, 29.9, 31.1, 30.0),
        datetime=("2022-07-01", "2022-08-01"),
        sidecar=str(p),
        resolution=60, epsg=32636, low_aod_frac=0.5,
        bands=["blue", "green"], chunk=512, scout_factor=8,
        out=str(tmp_path / "c.tif"), ee_module=_EE(),
    )
    assert out["scenes"] == ["S2A_36RUU_20220717_0_L1C"]   # only the clean day
    blue = out["bands"]["blue"]
    expected = correct_reflectance(0.3, 1.27, 0.08, 0.145)
    valid = np.isfinite(blue)
    assert valid.mean() == pytest.approx(1.0)              # cs=0.9 clear everywhere
    assert np.allclose(blue[valid], expected, atol=1e-5)
    assert (out["count"] >= 1).all()
    # GeoTIFF written and round-trips
    import rasterio
    with rasterio.open(tmp_path / "c.tif") as ds:
        assert ds.count == 2
        assert int(np.nanmean(ds.read(1))) == pytest.approx(expected * 1e4, abs=2)
