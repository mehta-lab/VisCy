"""Tests for dimensionality reduction CLI command and configuration."""

import anndata as ad
import numpy as np
import pytest
from dimensionality_reduction.config import (
    DimensionalityReductionConfig,
    PCAConfig,
    PHATEConfig,
    UMAPConfig,
)
from dimensionality_reduction.reduce_dimensionality import (
    _run_pca,
    _run_phate,
    _run_umap,
)
from pydantic import ValidationError


@pytest.fixture
def synthetic_zarr(tmp_path):
    """Create a synthetic AnnData zarr for testing reductions."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 64
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    adata = ad.AnnData(X=X)
    zarr_path = tmp_path / "embeddings.zarr"
    adata.write_zarr(zarr_path)
    return str(zarr_path)


class TestDimensionalityReductionConfig:
    def test_valid_config_pca(self, synthetic_zarr):
        cfg = DimensionalityReductionConfig(
            input_path=synthetic_zarr,
            pca=PCAConfig(n_components=10),
        )
        assert cfg.pca.n_components == 10
        assert cfg.umap is None
        assert cfg.phate is None

    def test_valid_config_all_methods(self, synthetic_zarr):
        cfg = DimensionalityReductionConfig(
            input_path=synthetic_zarr,
            pca=PCAConfig(),
            umap=UMAPConfig(),
            phate=PHATEConfig(),
        )
        assert cfg.pca is not None
        assert cfg.umap is not None
        assert cfg.phate is not None

    def test_missing_methods_raises(self, synthetic_zarr):
        with pytest.raises(ValidationError, match="At least one reduction method"):
            DimensionalityReductionConfig(input_path=synthetic_zarr)

    def test_missing_input_path_raises(self):
        with pytest.raises(ValidationError, match="Input path not found"):
            DimensionalityReductionConfig(
                input_path="/nonexistent/path.zarr",
                pca=PCAConfig(),
            )

    def test_output_path_defaults_none(self, synthetic_zarr):
        cfg = DimensionalityReductionConfig(
            input_path=synthetic_zarr,
            pca=PCAConfig(),
        )
        assert cfg.output_path is None

    def test_overwrite_keys_default_false(self, synthetic_zarr):
        cfg = DimensionalityReductionConfig(
            input_path=synthetic_zarr,
            pca=PCAConfig(),
        )
        assert cfg.overwrite_keys is False


class TestRunPCA:
    def test_pca_default(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 32)).astype(np.float32)
        cfg = PCAConfig()
        key, result = _run_pca(features, cfg)
        assert key == "X_pca"
        assert result.shape[0] == 50
        assert result.shape[1] == 32  # all components kept

    def test_pca_n_components(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 32)).astype(np.float32)
        cfg = PCAConfig(n_components=5)
        key, result = _run_pca(features, cfg)
        assert key == "X_pca"
        assert result.shape == (50, 5)

    def test_pca_no_normalize(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 32)).astype(np.float32)
        cfg = PCAConfig(normalize_features=False, n_components=10)
        key, result = _run_pca(features, cfg)
        assert key == "X_pca"
        assert result.shape == (50, 10)


class TestRunUMAP:
    @pytest.fixture(autouse=True)
    def _skip_no_umap(self):
        pytest.importorskip("umap")

    def test_umap_default(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 32)).astype(np.float32)
        cfg = UMAPConfig()
        key, result = _run_umap(features, cfg)
        assert key == "X_umap"
        assert result.shape == (50, 2)

    def test_umap_small_dataset_guard(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((10, 16)).astype(np.float32)
        cfg = UMAPConfig(n_neighbors=15)
        key, result = _run_umap(features, cfg)
        assert key == "X_umap"
        assert result.shape == (10, 2)


class TestRunPHATE:
    @pytest.fixture(autouse=True)
    def _skip_no_phate(self):
        pytest.importorskip("phate")

    def test_phate_default(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 32)).astype(np.float32)
        cfg = PHATEConfig()
        key, result = _run_phate(features, cfg)
        assert key == "X_phate"
        assert result.shape == (50, 2)

    def test_phate_small_dataset_guard(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((4, 16)).astype(np.float32)
        cfg = PHATEConfig(knn=5)
        key, result = _run_phate(features, cfg)
        assert key == "X_phate"
        assert result.shape == (4, 2)


class TestCLIIntegration:
    def test_pca_end_to_end(self, synthetic_zarr, tmp_path):
        from click.testing import CliRunner
        from dimensionality_reduction.reduce_dimensionality import main

        output_path = str(tmp_path / "output.zarr")
        config_content = f"input_path: {synthetic_zarr}\noutput_path: {output_path}\npca:\n  n_components: 10\n"
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        runner = CliRunner()
        result = runner.invoke(main, ["-c", str(config_path)])
        assert result.exit_code == 0, result.output

        adata = ad.read_zarr(output_path)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (100, 10)

    def test_overwrite_keys_protection(self, synthetic_zarr, tmp_path):
        from click.testing import CliRunner
        from dimensionality_reduction.reduce_dimensionality import main

        # Pre-populate X_pca
        adata = ad.read_zarr(synthetic_zarr)
        adata.obsm["X_pca"] = np.zeros((100, 5))
        adata.write_zarr(synthetic_zarr)

        config_content = f"input_path: {synthetic_zarr}\npca:\n  n_components: 10\n"
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        runner = CliRunner()
        result = runner.invoke(main, ["-c", str(config_path)])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_overwrite_keys_allowed(self, synthetic_zarr, tmp_path):
        from click.testing import CliRunner
        from dimensionality_reduction.reduce_dimensionality import main

        # Pre-populate X_pca
        adata = ad.read_zarr(synthetic_zarr)
        adata.obsm["X_pca"] = np.zeros((100, 5))
        adata.write_zarr(synthetic_zarr)

        config_content = f"input_path: {synthetic_zarr}\noverwrite_keys: true\npca:\n  n_components: 10\n"
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        runner = CliRunner()
        result = runner.invoke(main, ["-c", str(config_path)])
        assert result.exit_code == 0, result.output

        adata = ad.read_zarr(synthetic_zarr)
        assert adata.obsm["X_pca"].shape == (100, 10)

    def test_writes_back_to_input_when_no_output(self, synthetic_zarr, tmp_path):
        from click.testing import CliRunner
        from dimensionality_reduction.reduce_dimensionality import main

        config_content = f"input_path: {synthetic_zarr}\npca:\n  n_components: 5\n"
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        runner = CliRunner()
        result = runner.invoke(main, ["-c", str(config_path)])
        assert result.exit_code == 0, result.output

        adata = ad.read_zarr(synthetic_zarr)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (100, 5)
