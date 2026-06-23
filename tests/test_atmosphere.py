import json

import numpy as np
import pytest

from surface_priors.atmosphere import (
    AtmoSidecar,
    SceneAtmosphere,
    correct_reflectance,
)


def _scene(aod=0.13, wvp=2.0, tcwv=(1.0, 3.0)):
    # two bands, linear-in-tcwv coeffs
    return SceneAtmosphere(
        maiac_aod=aod,
        wvp=wvp,
        tcwv_nodes=tuple(tcwv),
        xap=((1.0, 1.2), (2.0, 2.2)),
        xbp=((0.0, 0.1), (0.0, 0.1)),
        xcp=((0.0, 0.0), (0.0, 0.0)),
    )


def test_coeffs_interpolates_and_clamps():
    s = _scene()
    assert s.coeffs(0, 2.0)[0] == pytest.approx(1.5)   # midpoint
    assert s.coeffs(0, 0.0)[0] == pytest.approx(1.0)   # clamp low
    assert s.coeffs(0, 9.0)[0] == pytest.approx(2.0)   # clamp high


def test_correct_reflectance_inverts_forward():
    xap, xbp, xcp = 1.27, 0.080, 0.145
    for boa in (0.02, 0.08, 0.2, 0.4):
        y = boa / (1.0 - xcp * boa)
        toa = (y + xbp) / xap
        assert correct_reflectance(toa, xap, xbp, xcp) == pytest.approx(boa, abs=1e-6)


def _sidecar_payload():
    base = {
        "maiac_aod": 0.0, "wvp": 2.0, "tcwv_nodes": [2.0],
        "xap": [[1.27, 1.1]], "xbp": [[0.08, 0.05]], "xcp": [[0.145, 0.12]],
    }
    scenes = {}
    for sid, aod in [
        ("S2A_36RTU_20220720_0_L1C", 0.30),
        ("S2A_36RUU_20220717_0_L1C", 0.12),
        ("S2B_36RTU_20220715_0_L1C", 0.20),
    ]:
        scenes[sid] = {**base, "maiac_aod": aod}
    return {"bands": ["blue", "green"], "scenes": scenes}


def test_load_and_select_low_aod(tmp_path):
    p = tmp_path / "sc.json"
    p.write_text(json.dumps(_sidecar_payload()))
    sc = AtmoSidecar.load(p)
    assert sc.bands == ("blue", "green")
    assert sc.band_index("green") == 1
    sel = sc.select_low_aod(0.67)   # lowest 2 of 3
    assert sel == ["S2A_36RUU_20220717_0_L1C", "S2B_36RTU_20220715_0_L1C"]


def test_lookup_matches_gee_system_index_by_tile_date(tmp_path):
    p = tmp_path / "sc.json"
    p.write_text(json.dumps(_sidecar_payload()))
    sc = AtmoSidecar.load(p)
    # exact STAC id
    assert sc.lookup("S2A_36RUU_20220717_0_L1C").maiac_aod == pytest.approx(0.12)
    # GEE system:index for the same tile+date resolves via (tile, date)
    atm = sc.lookup("20220717T082611_20220717T083855_T36RUU")
    assert atm is not None and atm.maiac_aod == pytest.approx(0.12)
    assert sc.lookup("nonsense") is None


def test_correct_stack_matches_scalar_path(tmp_path):
    p = tmp_path / "sc.json"
    p.write_text(json.dumps(_sidecar_payload()))
    sc = AtmoSidecar.load(p)
    atm = sc.lookup("S2A_36RUU_20220717_0_L1C")
    toa = np.array([[[0.2, 0.3]], [[0.25, 0.35]]], dtype="float32")  # (2band,1,2)
    out = sc.correct(atm, ["blue", "green"], toa)
    # band 0 matches the scalar correction with the scene's coeffs
    xap, xbp, xcp = atm.coeffs(0, atm.wvp)
    assert out[0, 0, 0] == pytest.approx(correct_reflectance(0.2, xap, xbp, xcp), abs=1e-6)
    assert out.shape == toa.shape
