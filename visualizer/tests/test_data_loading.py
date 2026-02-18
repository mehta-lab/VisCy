"""Tests for data loading and PHATE computation."""

from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from visualizer.config import DatasetConfig, MultiDatasetConfig
from visualizer.data_loading import (
    compute_joint_phate,
    load_and_prepare_data,
    load_multiple_datasets,
    validate_feature_compatibility,
)


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object."""
    n_obs = 100
    n_features = 10

    adata = ad.AnnData(
        X=np.random.randn(n_obs, n_features),
        obs=pd.DataFrame(
            {
                "track_id": np.random.randint(1, 20, n_obs),
                "fov_name": np.random.choice(["A/1/0", "A/2/0", "B/1/0"], n_obs),
                "t": np.random.randint(0, 10, n_obs),
                "y": np.random.randint(0, 100, n_obs),
                "x": np.random.randint(0, 100, n_obs),
                "id": range(n_obs),
            }
        ),
        obsm={"X_phate": np.random.randn(n_obs, 2)},
    )

    return adata


@pytest.fixture
def mock_adata_with_annotation():
    """Create a mock AnnData object with annotations."""
    n_obs = 100
    n_features = 10

    adata = ad.AnnData(
        X=np.random.randn(n_obs, n_features),
        obs=pd.DataFrame(
            {
                "track_id": np.random.randint(1, 20, n_obs),
                "fov_name": np.random.choice(["A/1/0", "A/2/0"], n_obs),
                "t": np.random.randint(0, 10, n_obs),
                "y": np.random.randint(0, 100, n_obs),
                "x": np.random.randint(0, 100, n_obs),
                "id": range(n_obs),
                "infection_status": np.random.choice(["infected", "uninfected"], n_obs),
            }
        ),
        obsm={"X_phate": np.random.randn(n_obs, 2)},
    )

    return adata


class TestLoadAndPrepareData:
    """Tests for load_and_prepare_data function."""

    @patch("visualizer.data_loading.read_zarr")
    def test_load_basic(self, mock_read_zarr, mock_adata):
        """Test basic data loading."""
        mock_read_zarr.return_value = mock_adata

        adata, plot_df, track_options, has_annotations = load_and_prepare_data(
            adata_path=Path("/path/to/adata.zarr")
        )

        assert isinstance(adata, ad.AnnData)
        assert isinstance(plot_df, pd.DataFrame)
        assert isinstance(track_options, list)
        assert isinstance(has_annotations, bool)
        assert "PHATE1" in plot_df.columns
        assert "PHATE2" in plot_df.columns
        assert "track_key" in plot_df.columns

    @patch("visualizer.data_loading.read_zarr")
    def test_load_with_dataset_id(self, mock_read_zarr, mock_adata):
        """Test loading with custom dataset_id."""
        mock_read_zarr.return_value = mock_adata

        adata, plot_df, _, _ = load_and_prepare_data(
            adata_path=Path("/path/to/adata.zarr"), dataset_id="custom_dataset"
        )

        assert "dataset_id" in adata.obs.columns
        assert all(adata.obs["dataset_id"] == "custom_dataset")
        assert all(plot_df["dataset_id"] == "custom_dataset")

    @patch("visualizer.data_loading.read_zarr")
    def test_load_with_fov_filter(self, mock_read_zarr, mock_adata):
        """Test FOV filtering."""
        mock_read_zarr.return_value = mock_adata
        initial_count = len(mock_adata)

        adata, plot_df, _, _ = load_and_prepare_data(
            adata_path=Path("/path/to/adata.zarr"), fov_filter=["A/1"]
        )

        assert len(adata) <= initial_count
        assert all("A/1" in str(fov) for fov in adata.obs["fov_name"])

    @patch("visualizer.data_loading.load_annotation_anndata")
    @patch("visualizer.data_loading.read_zarr")
    def test_load_with_annotations(
        self, mock_read_zarr, mock_load_annotation, mock_adata_with_annotation, tmp_path
    ):
        """Test loading with annotations."""
        mock_read_zarr.return_value = mock_adata_with_annotation
        mock_load_annotation.return_value = mock_adata_with_annotation

        annotation_csv = tmp_path / "annotations.csv"
        annotation_csv.write_text("id,infection_status\n1,infected\n2,uninfected")

        adata, plot_df, _, has_annotations = load_and_prepare_data(
            adata_path=Path("/path/to/adata.zarr"),
            annotation_csv=annotation_csv,
            annotation_column="infection_status",
        )

        assert has_annotations is True
        if "annotation" in plot_df.columns:
            assert "annotation" in plot_df.columns

    @patch("visualizer.data_loading.read_zarr")
    def test_load_missing_phate(self, mock_read_zarr):
        """Test error when PHATE embeddings are missing."""
        mock_adata_no_phate = ad.AnnData(
            X=np.random.randn(10, 5),
            obs=pd.DataFrame(
                {
                    "track_id": range(10),
                    "fov_name": ["A/1/0"] * 10,
                    "t": range(10),
                    "y": range(10),
                    "x": range(10),
                    "id": range(10),
                }
            ),
        )
        mock_read_zarr.return_value = mock_adata_no_phate

        with pytest.raises(ValueError, match="PHATE embeddings not found"):
            load_and_prepare_data(adata_path=Path("/path/to/adata.zarr"))

    @patch("visualizer.data_loading.read_zarr")
    def test_track_key_format(self, mock_read_zarr, mock_adata):
        """Test that track_key has the correct format."""
        mock_read_zarr.return_value = mock_adata

        _, plot_df, track_options, _ = load_and_prepare_data(
            adata_path=Path("/path/to/adata.zarr"), dataset_id="dataset1"
        )

        assert "track_key" in plot_df.columns
        sample_key = plot_df["track_key"].iloc[0]
        assert "dataset1" in sample_key
        assert "/" in sample_key


class TestValidateFeatureCompatibility:
    """Tests for validate_feature_compatibility function."""

    def test_compatible_datasets(self):
        """Test validation with compatible datasets."""
        adata1 = ad.AnnData(X=np.random.randn(50, 10))
        adata2 = ad.AnnData(X=np.random.randn(60, 10))
        adata3 = ad.AnnData(X=np.random.randn(70, 10))

        validate_feature_compatibility([adata1, adata2, adata3])

    def test_incompatible_datasets(self):
        """Test validation fails with incompatible datasets."""
        adata1 = ad.AnnData(X=np.random.randn(50, 10))
        adata2 = ad.AnnData(X=np.random.randn(60, 15))

        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            validate_feature_compatibility([adata1, adata2])

    def test_single_dataset(self):
        """Test validation with single dataset (should pass)."""
        adata1 = ad.AnnData(X=np.random.randn(50, 10))

        validate_feature_compatibility([adata1])


class TestComputeJointPhate:
    """Tests for compute_joint_phate function."""

    @patch("visualizer.data_loading.compute_phate")
    def test_compute_phate_basic(self, mock_compute_phate):
        """Test basic PHATE computation."""
        n_obs = 100
        n_features = 10
        adata = ad.AnnData(X=np.random.randn(n_obs, n_features))

        mock_embedding = np.random.randn(n_obs, 2)
        mock_compute_phate.return_value = (None, mock_embedding)

        result = compute_joint_phate(adata)

        assert "X_phate" in result.obsm
        assert result.obsm["X_phate"].shape == (n_obs, 2)
        mock_compute_phate.assert_called_once()

    @patch("visualizer.data_loading.compute_phate")
    def test_compute_phate_custom_params(self, mock_compute_phate):
        """Test PHATE computation with custom parameters."""
        n_obs = 100
        n_features = 10
        adata = ad.AnnData(X=np.random.randn(n_obs, n_features))

        mock_embedding = np.random.randn(n_obs, 3)
        mock_compute_phate.return_value = (None, mock_embedding)

        result = compute_joint_phate(
            adata, n_components=3, knn=10, decay=50, scale_embeddings=True
        )

        assert "X_phate" in result.obsm
        mock_compute_phate.assert_called_once_with(
            adata, n_components=3, knn=10, decay=50, scale_embeddings=True
        )


class TestLoadMultipleDatasets:
    """Tests for load_multiple_datasets function."""

    @patch("visualizer.data_loading.load_and_prepare_data")
    @patch("visualizer.data_loading.compute_joint_phate")
    def test_load_multiple_datasets_basic(
        self, mock_compute_phate, mock_load_and_prepare
    ):
        """Test loading multiple datasets."""
        mock_adata1 = ad.AnnData(
            X=np.random.randn(50, 10),
            obs=pd.DataFrame(
                {
                    "track_id": range(50),
                    "fov_name": ["A/1/0"] * 50,
                    "t": range(50),
                    "y": range(50),
                    "x": range(50),
                    "id": range(50),
                    "dataset_id": ["dataset1"] * 50,
                }
            ),
            obsm={"X_phate": np.random.randn(50, 2)},
        )

        mock_adata2 = ad.AnnData(
            X=np.random.randn(60, 10),
            obs=pd.DataFrame(
                {
                    "track_id": range(60),
                    "fov_name": ["B/1/0"] * 60,
                    "t": range(60),
                    "y": range(60),
                    "x": range(60),
                    "id": range(60),
                    "dataset_id": ["dataset2"] * 60,
                }
            ),
            obsm={"X_phate": np.random.randn(60, 2)},
        )

        mock_load_and_prepare.side_effect = [
            (mock_adata1, pd.DataFrame(), [], False),
            (mock_adata2, pd.DataFrame(), [], False),
        ]

        mock_compute_phate.return_value = ad.AnnData(
            X=np.random.randn(110, 10),
            obs=pd.DataFrame(
                {
                    "track_id": list(range(50)) + list(range(60)),
                    "fov_name": ["A/1/0"] * 50 + ["B/1/0"] * 60,
                    "t": list(range(50)) + list(range(60)),
                    "y": list(range(50)) + list(range(60)),
                    "x": list(range(50)) + list(range(60)),
                    "id": list(range(50)) + list(range(60)),
                    "dataset_id": ["dataset1"] * 50 + ["dataset2"] * 60,
                }
            ),
            obsm={"X_phate": np.random.randn(110, 2)},
        )

        config = MultiDatasetConfig(
            datasets=[
                DatasetConfig(
                    adata_path=Path("/path/to/adata1.zarr"),
                    data_path=Path("/path/to/data1.zarr"),
                    dataset_id="dataset1",
                ),
                DatasetConfig(
                    adata_path=Path("/path/to/adata2.zarr"),
                    data_path=Path("/path/to/data2.zarr"),
                    dataset_id="dataset2",
                ),
            ],
            phate_kwargs={"n_components": 2, "knn": 5},
        )

        adata_joint, plot_df, track_options, has_annotations = load_multiple_datasets(
            config
        )

        assert isinstance(adata_joint, ad.AnnData)
        assert isinstance(plot_df, pd.DataFrame)
        assert isinstance(track_options, list)

    @patch("visualizer.data_loading.load_and_prepare_data")
    def test_load_single_dataset_no_phate_kwargs(self, mock_load_and_prepare):
        """Test loading single dataset without recomputing PHATE."""
        mock_adata = ad.AnnData(
            X=np.random.randn(50, 10),
            obs=pd.DataFrame(
                {
                    "track_id": range(50),
                    "fov_name": ["A/1/0"] * 50,
                    "t": range(50),
                    "y": range(50),
                    "x": range(50),
                    "id": range(50),
                    "dataset_id": ["dataset1"] * 50,
                }
            ),
            obsm={"X_phate": np.random.randn(50, 2)},
        )

        mock_load_and_prepare.return_value = (mock_adata, pd.DataFrame(), [], False)

        config = MultiDatasetConfig(
            datasets=[
                DatasetConfig(
                    adata_path=Path("/path/to/adata.zarr"),
                    data_path=Path("/path/to/data.zarr"),
                    dataset_id="dataset1",
                )
            ],
            phate_kwargs=None,
        )

        adata_joint, plot_df, track_options, has_annotations = load_multiple_datasets(
            config
        )

        assert "X_phate" in adata_joint.obsm
        assert "track_key" in plot_df.columns
