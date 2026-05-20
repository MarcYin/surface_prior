import json

import numpy as np
import rasterio

from surface_priors.persistence import CompositeStore, stable_json_hash
from surface_priors.types import GridSpec, PriorComposite


def test_store_writes_tiled_deflate_geotiffs_and_stac_item(tmp_path):
    grid = GridSpec.from_bounds((0, 0, 2, 2), "EPSG:4326", 1)
    composite = PriorComposite(
        product_id="prior-fixture",
        grid=grid,
        band_names=("iso", "vol"),
        data=np.array(
            [
                [[0.1, 0.2], [0.3, np.nan]],
                [[0.4, 0.5], [0.6, 0.7]],
            ],
            dtype="float32",
        ),
        uncertainty=np.array(
            [
                [[5, 201], [200, np.nan]],
                [[0, 50], [100, 150]],
            ],
            dtype="float32",
        ),
        quality=np.zeros((2, 2), dtype="uint16"),
        sample_index=np.zeros((2, 2), dtype="int16"),
        selected_observation=np.zeros((2, 2), dtype="int16"),
        observation_count=np.ones((2, 2), dtype="uint16"),
    )
    request = {
        "product_id": "prior-fixture",
        "wgs84_bounds": [0, 0, 2, 2],
        "native_bounds": [0, 0, 2, 2],
        "native_crs": "EPSG:4326",
        "composite_period": "2024-07",
    }
    request_hash = stable_json_hash(request)

    store = CompositeStore(tmp_path)
    product = store.save(request_hash=request_hash, request=request, composite=composite)

    stac_path = tmp_path / request_hash / "stac-item.json"
    assert stac_path.exists()
    stac_item = json.loads(stac_path.read_text(encoding="utf-8"))
    assert stac_item["type"] == "Feature"
    assert stac_item["properties"]["surface:schema_version"] == "surface-priors/v1"
    assert stac_item["properties"]["surface:prior_type"] == "brdf"
    assert stac_item["properties"]["surface:asset_layout"] == "single-band-geotiff-per-band"
    assert stac_item["properties"]["surface:band_names"] == ["iso", "vol"]
    assert stac_item["properties"]["surface:composite_period"] == "2024-07"
    assert stac_item["assets"]["prior_iso"]["href"] == "assets/prior/2024-07/iso.tif"
    assert stac_item["assets"]["prior_iso"]["surface:asset_kind"] == "prior"
    assert stac_item["assets"]["prior_iso"]["surface:band_index"] == 0
    assert stac_item["assets"]["prior_iso"]["surface:composite_period"] == "2024-07"
    assert len(stac_item["assets"]["prior_iso"]["raster:bands"]) == 1
    assert stac_item["assets"]["uncertainty_vol"]["href"] == "assets/uncertainty/2024-07/vol.tif"
    assert stac_item["assets"]["uncertainty_vol"]["surface:asset_kind"] == "uncertainty"
    assert stac_item["assets"]["uncertainty_vol"]["surface:band_index"] == 1
    assert len(stac_item["assets"]["uncertainty_vol"]["raster:bands"]) == 1
    assert product.stac_item["id"] == "prior-fixture"
    assert product.composite.attrs["composite_period"] == "2024-07"

    with rasterio.open(tmp_path / request_hash / "assets" / "prior" / "2024-07" / "iso.tif") as dataset:
        assert dataset.count == 1
        assert dataset.dtypes == ("uint16",)
        assert dataset.nodata == 65535
        assert dataset.compression.value.lower() == "deflate"
        assert dataset.block_shapes[0] == (512, 512)
        assert dataset.overviews(1) == []
        assert dataset.scales == (0.0001,)
        assert dataset.descriptions == ("iso",)
        assert dataset.read(1).tolist() == [[1000, 2000], [3000, 65535]]

    with rasterio.open(
        tmp_path / request_hash / "assets" / "uncertainty" / "2024-07" / "iso.tif"
    ) as dataset:
        assert dataset.count == 1
        assert dataset.dtypes == ("uint8",)
        assert dataset.nodata == 255
        assert dataset.compression.value.lower() == "deflate"
        assert dataset.block_shapes[0] == (512, 512)
        assert dataset.overviews(1) == []
        assert dataset.descriptions == ("iso_relative_uncertainty",)
        assert dataset.read(1).tolist() == [[5, 255], [200, 255]]

    loaded = store.load(request_hash, request=request)

    assert loaded.composite.band_names == ("iso", "vol")
    assert loaded.composite.attrs["composite_period"] == "2024-07"
    assert loaded.composite.data.shape == (2, 2, 2)
    np.testing.assert_allclose(loaded.composite.data[1], [[0.4, 0.5], [0.6, 0.7]])
    np.testing.assert_allclose(loaded.composite.uncertainty[1], [[0, 50], [100, 150]])


def test_store_skips_uncertainty_when_composite_has_no_finite_values(tmp_path):
    grid = GridSpec.from_bounds((0, 0, 2, 2), "EPSG:4326", 1)
    composite = PriorComposite(
        product_id="no-uncertainty",
        grid=grid,
        band_names=("iso",),
        data=np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype="float32"),
        uncertainty=np.full((1, 2, 2), np.nan, dtype="float32"),
        quality=np.zeros((2, 2), dtype="uint16"),
        sample_index=np.zeros((2, 2), dtype="int16"),
        selected_observation=np.zeros((2, 2), dtype="int16"),
        observation_count=np.ones((2, 2), dtype="uint16"),
    )
    request = {
        "product_id": "no-uncertainty",
        "wgs84_bounds": [0, 0, 2, 2],
        "native_bounds": [0, 0, 2, 2],
        "native_crs": "EPSG:4326",
    }
    request_hash = stable_json_hash(request)

    store = CompositeStore(tmp_path)
    product = store.save(request_hash=request_hash, request=request, composite=composite)

    # No uncertainty directory and no uncertainty assets in the STAC item.
    uncertainty_root = tmp_path / request_hash / "assets" / "uncertainty"
    assert not uncertainty_root.exists()
    assert all(
        not key.startswith("uncertainty_") for key in product.stac_item["assets"]
    )

    # Round-trip via store.load — uncertainty array must come back as all-NaN.
    loaded = store.load(request_hash, request=request)
    assert loaded.composite.uncertainty.shape == (1, 2, 2)
    assert np.isnan(loaded.composite.uncertainty).all()
