import numpy as np

from surface_priors.composite import PriorCompositor
from surface_priors.types import GridSpec, Observation


def test_best_pixel_prefers_quality_then_sample_index_without_time():
    grid = GridSpec.from_bounds((0, 0, 2, 2), "EPSG:4326", 1)
    bands = ("iso",)
    first = Observation(
        data=np.array([[[10, 10], [10, 10]]], dtype="float32") / 100,
        quality=np.array([[1, 0], [0, 0]], dtype="uint16"),
        uncertainty=np.array([[[5, 5], [5, 5]]], dtype="float32"),
        sample_index=np.array([[5, 5], [5, 8]], dtype="int16"),
        band_names=bands,
        source_id="first",
    )
    second = Observation(
        data=np.array([[[20, 20], [20, 20]]], dtype="float32") / 100,
        quality=np.array([[0, 0], [0, 0]], dtype="uint16"),
        uncertainty=np.array([[[6, 6], [6, 6]]], dtype="float32"),
        sample_index=np.array([[9, 9], [2, 8]], dtype="int16"),
        band_names=bands,
        source_id="second",
    )

    composite = PriorCompositor().compose(
        product_id="fixture",
        grid=grid,
        band_names=bands,
        observations=(first, second),
    )

    assert composite.data.shape == (1, 2, 2)
    assert composite.data[0, 0, 0] == np.float32(0.20)
    assert composite.data[0, 0, 1] == np.float32(0.10)
    assert composite.data[0, 1, 0] == np.float32(0.20)
    assert composite.data[0, 1, 1] == np.float32(0.10)
    assert composite.uncertainty[0, 0, 0] == np.float32(6)
    assert np.all(composite.observation_count == 2)


def test_empty_composite_has_schema_arrays():
    grid = GridSpec.from_bounds((0, 0, 2, 2), "EPSG:4326", 1)

    composite = PriorCompositor().compose(
        product_id="empty",
        grid=grid,
        band_names=("iso", "vol"),
        observations=(),
    )

    assert composite.data.shape == (2, 2, 2)
    assert np.isnan(composite.data).all()
    assert np.all(composite.selected_observation == -1)


def test_best_pixel_default_does_not_emit_stack_std_uncertainty():
    """Default mode picks a single winner; the stack std around it is not a
    coherent uncertainty estimate, so the compositor leaves it as NaN."""
    grid = GridSpec.from_bounds((0, 0, 2, 2), "EPSG:4326", 1)
    bands = ("iso",)
    first = Observation(
        data=np.array([[[10, 10], [10, 10]]], dtype="float32") / 100,
        quality=np.array([[0, 0], [0, 0]], dtype="uint16"),
        band_names=bands,
    )
    second = Observation(
        data=np.array([[[20, 20], [20, 20]]], dtype="float32") / 100,
        quality=np.array([[1, 1], [1, 1]], dtype="uint16"),
        band_names=bands,
    )

    composite = PriorCompositor().compose(
        product_id="default-no-fallback",
        grid=grid,
        band_names=bands,
        observations=(first, second),
    )

    assert np.isnan(composite.uncertainty).all()
    # Data still picks the cleanest scene.
    np.testing.assert_allclose(composite.data[0], 0.10)


def test_emit_uncertainty_true_runs_stack_std_fallback():
    grid = GridSpec.from_bounds((0, 0, 2, 2), "EPSG:4326", 1)
    bands = ("iso",)
    first = Observation(
        data=np.array([[[10, 10], [10, 10]]], dtype="float32") / 100,
        quality=np.array([[0, 0], [0, 0]], dtype="uint16"),
        band_names=bands,
    )
    second = Observation(
        data=np.array([[[20, 20], [20, 20]]], dtype="float32") / 100,
        quality=np.array([[1, 1], [1, 1]], dtype="uint16"),
        band_names=bands,
    )

    composite = PriorCompositor(emit_uncertainty=True).compose(
        product_id="legacy-fallback",
        grid=grid,
        band_names=bands,
        observations=(first, second),
    )

    assert np.isfinite(composite.uncertainty).all()
    # std/|mean| * 100 between [0.10, 0.20] referenced to the best-pixel 0.10.
    expected = (np.std([0.10, 0.20]) / 0.10) * 100.0
    np.testing.assert_allclose(composite.uncertainty[0], expected, rtol=1e-3)

