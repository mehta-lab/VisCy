"""Integration tests for LOT batch-correction fitting."""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.lot_correction.lot_correction import (
    _coerce_obs_for_zarr,
    _is_string_dtype,
    _pool_embeddings,
    apply_lot_correction,
    fit_lot_correction,
    load_lot_pipeline,
    save_lot_pipeline,
)


def _make_adata(n: int, d: int, seed: int, shift: float = 0.0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, d)) + shift).astype(np.float32)
    obs = pd.DataFrame(
        {"fov_name": [f"A/{i % 3}/0" for i in range(n)]},
        index=pd.Index([f"cell_{i}" for i in range(n)], dtype=object),
    )
    obs["fov_name"] = obs["fov_name"].astype(object)
    return ad.AnnData(X=X, obs=obs)


def _write_zarr(adata: ad.AnnData, path: Path) -> None:
    ad.settings.allow_write_nullable_strings = True
    adata.var.index = adata.var.index.astype(object)
    adata.write_zarr(path, convert_strings_to_categoricals=False)


def test_pool_embeddings_concatenates_rows():
    a = _make_adata(10, 8, seed=0)
    b = _make_adata(15, 8, seed=1)
    pooled = _pool_embeddings([a, b])
    assert pooled.shape == (25, 8)
    assert pooled.dtype == np.float32


def test_pool_embeddings_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        _pool_embeddings([])


def test_pool_embeddings_rejects_mismatched_features():
    with pytest.raises(ValueError, match="same feature dimension"):
        _pool_embeddings([_make_adata(5, 8, seed=0), _make_adata(5, 7, seed=1)])


def test_fit_pools_multiple_datasets():
    source = [_make_adata(40, 16, seed=1), _make_adata(30, 16, seed=2)]
    target = [_make_adata(50, 16, seed=3, shift=2.0)]
    pipeline = fit_lot_correction(source, target, n_pca=5, ns_lot=100, random_seed=0)

    assert pipeline["pca"] is not None
    assert pipeline["pca"].n_components_ == 5
    assert pipeline["n_pca"] == 5
    assert 0.0 < pipeline["pca_variance_explained"] <= 100.0
    # PCA fit on pooled 70 source + 50 target cells.
    assert pipeline["pca"].n_samples_ == 120


def test_fit_records_channel():
    source = [_make_adata(40, 16, seed=1)]
    target = [_make_adata(50, 16, seed=3, shift=2.0)]
    pipeline = fit_lot_correction(source, target, channel="Phase3D", n_pca=5, ns_lot=100, random_seed=0)
    assert pipeline["channel"] == "Phase3D"


def test_channel_defaults_to_none():
    source = [_make_adata(40, 16, seed=1)]
    target = [_make_adata(50, 16, seed=3, shift=2.0)]
    pipeline = fit_lot_correction(source, target, n_pca=5, ns_lot=100, random_seed=0)
    assert pipeline["channel"] is None


def test_fit_without_pca():
    source = [_make_adata(40, 16, seed=1)]
    target = [_make_adata(50, 16, seed=3, shift=2.0)]
    pipeline = fit_lot_correction(source, target, n_pca=None, ns_lot=100, random_seed=0)

    assert pipeline["pca"] is None
    assert pipeline["n_pca"] is None
    assert pipeline["pca_variance_explained"] is None


def test_fit_raises_on_too_few_cells():
    source = [_make_adata(3, 16, seed=1)]
    target = [_make_adata(50, 16, seed=3)]
    with pytest.raises(ValueError, match="Too few reference cells"):
        fit_lot_correction(source, target, n_pca=5, random_seed=0)


@pytest.mark.parametrize("n_pca", [5, None])
def test_fit_save_load_then_apply_to_many(tmp_path, n_pca):
    """Fit once, save, reload, then apply the same pipeline to several zarrs."""
    source = [_make_adata(40, 16, seed=1)]
    target = [_make_adata(50, 16, seed=3, shift=2.0)]
    pipeline = fit_lot_correction(source, target, channel="Phase3D", n_pca=n_pca, ns_lot=100, random_seed=0)

    pipeline_path = tmp_path / "pipeline.pkl"
    save_lot_pipeline(pipeline, pipeline_path)
    loaded = load_lot_pipeline(pipeline_path)

    assert loaded["n_pca"] == (n_pca if n_pca is not None else None)
    assert (loaded["pca"] is not None) == (n_pca is not None)
    assert loaded["channel"] == "Phase3D"

    ad.settings.allow_write_nullable_strings = True
    expected_dim = n_pca if n_pca is not None else 16
    for i in range(3):
        input_zarr = tmp_path / f"input_{i}.zarr"
        input_adata = _make_adata(20, 16, seed=100 + i)
        input_adata.var.index = input_adata.var.index.astype(object)
        input_adata.write_zarr(input_zarr, convert_strings_to_categoricals=False)
        output_zarr = tmp_path / f"corrected_{i}.zarr"
        apply_lot_correction(input_zarr, loaded, output_zarr)
        out = ad.read_zarr(output_zarr)
        assert out.shape == (20, expected_dim)
        assert out.obsm["X_pre_lot"].shape == (20, expected_dim)
        expected_pre = loaded["scaler"].transform(input_adata.X)
        if loaded["pca"] is not None:
            expected_pre = loaded["pca"].transform(expected_pre)
        np.testing.assert_allclose(out.obsm["X_pre_lot"], expected_pre, rtol=1e-5)
        assert out.uns["lot_correction"]["channel"] == "Phase3D"


def test_apply_rejects_same_input_and_output(tmp_path):
    source = [_make_adata(40, 16, seed=1)]
    target = [_make_adata(50, 16, seed=2)]
    pipeline = fit_lot_correction(source, target, n_pca=5, ns_lot=50, random_seed=0)
    input_zarr = tmp_path / "input.zarr"
    _write_zarr(source[0], input_zarr)

    with pytest.raises(ValueError, match="must be different"):
        apply_lot_correction(input_zarr, pipeline, input_zarr, overwrite=True)

    assert ad.read_zarr(input_zarr).shape == source[0].shape


def test_apply_preserves_existing_output_on_failure(tmp_path):
    source = [_make_adata(40, 16, seed=1)]
    target = [_make_adata(50, 16, seed=2)]
    pipeline = fit_lot_correction(source, target, n_pca=5, ns_lot=50, random_seed=0)

    bad_input = _make_adata(20, 15, seed=3)
    input_zarr = tmp_path / "bad_input.zarr"
    _write_zarr(bad_input, input_zarr)

    previous = _make_adata(7, 3, seed=4)
    output_zarr = tmp_path / "output.zarr"
    _write_zarr(previous, output_zarr)
    previous_x = previous.X.copy()

    with pytest.raises(ValueError):
        apply_lot_correction(input_zarr, pipeline, output_zarr, overwrite=True)

    preserved = ad.read_zarr(output_zarr)
    assert preserved.shape == previous.shape
    np.testing.assert_array_equal(preserved.X, previous_x)


def test_apply_replaces_existing_output_after_success(tmp_path):
    source = [_make_adata(40, 16, seed=1)]
    target = [_make_adata(50, 16, seed=2)]
    pipeline = fit_lot_correction(source, target, n_pca=5, ns_lot=50, random_seed=0)

    input_zarr = tmp_path / "input.zarr"
    _write_zarr(source[0], input_zarr)
    output_zarr = tmp_path / "output.zarr"
    _write_zarr(_make_adata(7, 3, seed=4), output_zarr)

    apply_lot_correction(input_zarr, pipeline, output_zarr, overwrite=True)

    assert ad.read_zarr(output_zarr).shape == (40, 5)
    assert not list(tmp_path.glob(f".{output_zarr.name}.*"))


def test_coerce_obs_for_zarr_handles_categorical_string():
    """Categorical-over-string obs must coerce to a zarr-writable object dtype.

    Real embedding zarrs store fov_name/experiment/marker/perturbation as
    ``category`` whose categories are the pandas string extension dtype. Zarr
    cannot serialize string-extension-backed category values, so the coercion
    used by apply must drop them to plain object (recategorizing is not enough —
    the categories stay string-extension-backed). Reproduces the apply-time
    ``IORegistryError: No method registered for writing ArrowStringArray``.
    """
    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(6)]))
    obs["experiment"] = pd.Series(["exp_a"] * 6, index=obs.index, dtype="string").astype("category")
    obs["marker"] = pd.Series(["SEC61B"] * 6, index=obs.index, dtype="string").astype("category")
    obs["track_id"] = np.arange(6, dtype=np.int32)
    assert _is_string_dtype(obs["experiment"].dtype.categories.dtype)

    coerced = _coerce_obs_for_zarr(obs)

    # String-backed categoricals became plain object; numeric column untouched.
    assert coerced["experiment"].dtype == object
    assert coerced["marker"].dtype == object
    assert coerced["track_id"].dtype == np.int32
    assert list(coerced["experiment"]) == ["exp_a"] * 6

    # The coerced frame round-trips through zarr, which the categorical did not.
    adata = ad.AnnData(X=np.zeros((6, 3), dtype=np.float32), obs=coerced)
    adata.var.index = adata.var.index.astype(object)
    with tempfile.TemporaryDirectory() as d:
        adata.write_zarr(Path(d) / "roundtrip.zarr", convert_strings_to_categoricals=False)
