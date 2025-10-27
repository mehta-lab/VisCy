from __future__ import annotations

import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from iohub import open_ome_zarr
from pytest import TempPathFactory

from viscy.representation.evaluation import (
    load_annotation_anndata,
)
from viscy.representation.evaluation.annotation import convert


@pytest.fixture(scope="function")
def xr_embeddings_dataset(
    tracks_hcs_dataset: Path, tmp_path_factory: TempPathFactory
) -> Path:
    """
    Provides a mock xarray embeddings dataset with tracking information from tracks_hcs_dataset.

    Parameters
    ----------
    tracks_hcs_dataset : Path
        Path to the HCS dataset with tracking CSV files.

    Returns
    -------
    Path
        Path to the zarr store containing the embeddings dataset.
    """
    dataset_path = tmp_path_factory.mktemp("xr_embeddings.zarr")

    all_tracks = []

    dataset = open_ome_zarr(tracks_hcs_dataset)

    for fov_name, _ in dataset.positions():
        tracks_csv_path = tracks_hcs_dataset / fov_name / "tracks.csv"
        tracks_df = pd.read_csv(tracks_csv_path)
        tracks_df["fov_name"] = fov_name
        all_tracks.append(tracks_df)

    # Combine all tracks
    tracks_df = pd.concat(all_tracks, ignore_index=True)

    n_samples = len(tracks_df)
    n_features = 32

    rng = np.random.default_rng(42)

    # Generate synthetic features (embeddings)
    features = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    # Create coordinates (PCA, UMAP, PHATE, projections)
    pca_coords = rng.normal(size=(n_samples, 2)).astype(np.float32)
    umap_coords = rng.normal(size=(n_samples, 2)).astype(np.float32)
    phate_coords = rng.normal(size=(n_samples, 2)).astype(np.float32)
    projections = rng.normal(size=(n_samples, 2)).astype(
        np.float32
    )  # 2 projection dims

    # Create the xarray dataset
    ds = xr.Dataset(
        data_vars={
            "features": (["sample", "feature"], features),
        },
        coords={
            "fov_name": ("sample", tracks_df["fov_name"]),
            "track_id": ("sample", tracks_df["track_id"]),
            "t": ("sample", tracks_df["t"]),
            "id": ("sample", tracks_df["id"]),
            "parent_track_id": ("sample", tracks_df["parent_track_id"]),
            "parent_id": ("sample", tracks_df["parent_id"]),
            "y": ("sample", tracks_df["y"]),
            "x": ("sample", tracks_df["x"]),
            "PCA_0": ("sample", pca_coords[:, 0]),
            "PCA_1": ("sample", pca_coords[:, 1]),
            "UMAP_0": ("sample", umap_coords[:, 0]),
            "UMAP_1": ("sample", umap_coords[:, 1]),
            "PHATE_0": ("sample", phate_coords[:, 0]),
            "PHATE_1": ("sample", phate_coords[:, 1]),
            "projections": (["sample", "projection"], projections),
            "sample": np.arange(n_samples),
            "feature": np.arange(n_features),
            "projection": np.arange(projections.shape[1]),
        },
    )

    # Save to zarr
    ds.to_zarr(dataset_path)

    return dataset_path


@pytest.fixture(scope="function")
def anndata_embeddings(
    xr_embeddings_dataset: Path, tmp_path_factory: TempPathFactory
) -> Path:
    """
    Provides an AnnData zarr store created from xarray embeddings dataset.

    Parameters
    ----------
    xr_embeddings_dataset : Path
        Path to the xarray embeddings dataset.

    Returns
    -------
    Path
        Path to the AnnData zarr store.
    """
    rng = np.random.default_rng(42)

    # Create output path for AnnData
    adata_path = tmp_path_factory.mktemp("anndata_embeddings.zarr")

    # Load the xarray dataset
    embeddings_ds = xr.open_zarr(xr_embeddings_dataset)

    # Extract features as X matrix
    n_samples = len(embeddings_ds.coords["sample"])

    X = rng.normal(size=(n_samples, 32)).astype(np.float32)

    obs_data = {
        "id": embeddings_ds.coords["id"].values,
        "fov_name": embeddings_ds.coords["fov_name"].values.astype(str),
        "track_id": embeddings_ds.coords["track_id"].values,
        "parent_track_id": embeddings_ds.coords["parent_track_id"].values,
        "parent_id": embeddings_ds.coords["parent_id"].values,
        "t": embeddings_ds.coords["t"].values,
        "y": embeddings_ds.coords["y"].values,
        "x": embeddings_ds.coords["x"].values,
    }
    obs_df = pd.DataFrame(obs_data)

    # Get the number of samples from the dataset

    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        obsm={
            "X_projections": rng.normal(size=(n_samples, 3)).astype(np.float32),
            "X_pca": rng.normal(size=(n_samples, 3)).astype(np.float32),
            "X_umap": rng.uniform(-10, 10, size=(n_samples, 3)).astype(np.float32),
            "X_phate": rng.normal(scale=0.5, size=(n_samples, 3)).astype(np.float32),
        },
    )

    # Write to zarr
    adata.write_zarr(adata_path)

    return adata_path


def test_convert_xarray_annotation_to_anndata(xr_embeddings_dataset, tmp_path):
    """Test that convert_xarray_annotation_to_anndata correctly converts xarray to AnnData."""
    # Load the xarray dataset
    embeddings_ds = xr.open_zarr(xr_embeddings_dataset)

    # Define output path
    output_path = tmp_path / "test_converted.zarr"

    # Convert to AnnData using the function we're testing
    adata_result = convert(
        embeddings_ds=embeddings_ds,
        output_path=output_path,
        overwrite=True,
        return_anndata=True,
    )
    # Second conversion with overwrite=False should raise FileExistsError
    with pytest.raises(
        FileExistsError, match=f"Output path {output_path} already exists"
    ):
        convert(
            embeddings_ds=embeddings_ds,
            output_path=output_path,
            overwrite=False,
            return_anndata=False,
        )

    assert isinstance(adata_result, ad.AnnData)

    assert output_path.exists()

    adata_loaded = ad.read_zarr(output_path)

    np.testing.assert_allclose(adata_loaded.X, embeddings_ds["features"].values)

    # Verify obs columns
    expected_obs_columns = [
        "id",
        "fov_name",
        "track_id",
        "parent_track_id",
        "parent_id",
        "t",
        "y",
        "x",
    ]
    for col in expected_obs_columns:
        assert col in adata_loaded.obs.columns
        if col == "fov_name":
            assert list(adata_loaded.obs[col]) == list(
                embeddings_ds.coords[col].values.astype(str)
            )
        else:
            np.testing.assert_allclose(
                adata_loaded.obs[col].values, embeddings_ds.coords[col].values
            )

    # Verify obsm (embeddings)
    assert all(
        embedding_key in adata_loaded.obsm
        for embedding_key in ["X_pca", "X_umap", "X_phate", "X_projections"]
    )

    # Check projections
    np.testing.assert_allclose(
        adata_loaded.obsm["X_projections"], embeddings_ds.coords["projections"].values
    )

    # Check PCA
    np.testing.assert_allclose(
        adata_loaded.obsm["X_pca"][:, 0], embeddings_ds.coords["PCA_0"].values
    )
    np.testing.assert_allclose(
        adata_loaded.obsm["X_pca"][:, 1], embeddings_ds.coords["PCA_1"].values
    )

    # Check UMAP
    np.testing.assert_allclose(
        adata_loaded.obsm["X_umap"][:, 0], embeddings_ds.coords["UMAP_0"].values
    )
    np.testing.assert_allclose(
        adata_loaded.obsm["X_umap"][:, 1], embeddings_ds.coords["UMAP_1"].values
    )

    # Check PHATE
    np.testing.assert_allclose(
        adata_loaded.obsm["X_phate"][:, 0], embeddings_ds.coords["PHATE_0"].values
    )
    np.testing.assert_allclose(
        adata_loaded.obsm["X_phate"][:, 1], embeddings_ds.coords["PHATE_1"].values
    )


def test_load_annotation_anndata(tracks_hcs_dataset, anndata_embeddings, tmp_path):
    """Test that load_annotation_anndata correctly loads annotations from an AnnData object."""
    # Load the AnnData object
    adata = ad.read_zarr(anndata_embeddings)

    A11_annotations_path = tracks_hcs_dataset / "A" / "1" / "1" / "tracks.csv"

    A11_annotations_df = pd.read_csv(A11_annotations_path)

    rng = np.random.default_rng(42)
    A11_annotations_df["fov_name"] = "A/1/1"
    A11_annotations_df["infection_state"] = rng.choice(
        [-1, 0, 1], size=len(A11_annotations_df)
    )

    # Save the modified annotations to a new CSV file
    annotations_path = tmp_path / "test_annotations.csv"
    A11_annotations_df.to_csv(annotations_path, index=False)

    # Test the function with the new CSV file
    result = load_annotation_anndata(adata, str(annotations_path), "infection_state")

    assert len(result) == 2  # Only 2 observations from A/1/1 have annotations

    expected_values = A11_annotations_df["infection_state"].values
    actual_values = result.values
    np.testing.assert_array_equal(actual_values, expected_values)

    assert result.index.names == ["fov_name", "id"]
    assert all(result.index.get_level_values("fov_name") == "A/1/1")


def test_cli_convert_to_anndata(xr_embeddings_dataset, tmp_path):
    """Test CLI command: viscy convert_to_anndata"""
    output_path = tmp_path / "cli_output.zarr"

    # First conversion should succeed
    result = subprocess.run(
        [
            "viscy",
            "convert_to_anndata",
            "--embeddings_path",
            str(xr_embeddings_dataset),
            "--output_anndata_path",
            str(output_path),
            "--overwrite",
            "false",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert output_path.exists()

    # Verify basic correctness
    adata = ad.read_zarr(output_path)
    assert isinstance(adata, ad.AnnData)
    assert len(adata) > 0

    # Second run with overwrite=false should fail
    result_retry = subprocess.run(
        [
            "viscy",
            "convert_to_anndata",
            "--embeddings_path",
            str(xr_embeddings_dataset),
            "--output_anndata_path",
            str(output_path),
            "--overwrite",
            "false",
        ],
        capture_output=True,
        text=True,
    )

    assert result_retry.returncode != 0, (
        "Should fail when file exists with overwrite=false"
    )
    assert (
        "exists" in result_retry.stderr.lower()
        or "FileExistsError" in result_retry.stderr
    )
