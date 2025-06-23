from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from iohub import open_ome_zarr

from viscy.representation.evaluation.feature import (
    CellFeatures,
    DynamicFeatures,
)


@pytest.mark.parametrize("channel_idx", [0, 1])
def test_cell_features_with_labels_hcs(
    small_hcs_dataset, small_hcs_labels, channel_idx
):
    """Test CellFeatures with labels HCS dataset."""
    data_path = small_hcs_dataset
    with open_ome_zarr(data_path) as dataset:
        _, position = next(dataset.positions())
        image_array = position["0"]

    with open_ome_zarr(small_hcs_labels) as labels_dataset:
        _, position = next(labels_dataset.positions())
        labels_array = position["0"]

    # Extract patch from center
    patch_size = 16
    t, z = 0, 0
    y_center, x_center = image_array.shape[-2] // 2, image_array.shape[-1] // 2
    half_patch = patch_size // 2

    y_slice = slice(y_center - half_patch, y_center + half_patch)
    x_slice = slice(x_center - half_patch, x_center + half_patch)

    image_patch = image_array[t, channel_idx, z, y_slice, x_slice]
    labels_patch = labels_array[t, channel_idx, z, y_slice, x_slice]

    cf = CellFeatures(
        image=image_patch.astype(np.float32), segmentation_mask=labels_patch
    )
    features_df = cf.compute_all_features()

    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 1

    assert "mean_intensity" in features_df.columns
    assert "contrast" in features_df.columns
    assert "zernike_std" in features_df.columns
    assert "area" in features_df.columns

    for col in features_df.columns:
        value = features_df[col].iloc[0]
        if col in ["kurtosis", "skewness"]:
            patch_std = np.std(image_patch)
            if patch_std < 1e-10:
                # For constant images, kurtosis and skewness should be NaN
                assert np.isnan(value), (
                    f"Feature {col} should be NaN for constant image (std={patch_std})"
                )
            else:
                # For non-constant images, values should be finite and reasonable
                assert np.isfinite(value), (
                    f"Feature {col} is not finite for non-constant image (std={patch_std})"
                )
                assert -10 < value < 10, (
                    f"Feature {col} = {value} seems unreasonable for random data"
                )
        else:
            assert np.isfinite(value), f"Feature {col} is not finite: {value}"


@pytest.mark.parametrize("fov_path", ["A/1/0", "A/1/1", "A/2/0"])
def test_dynamic_features_with_tracks_hcs(tracks_hcs_dataset, fov_path):
    """Test DynamicFeatures with tracks HCS dataset."""

    tracks_path = Path(tracks_hcs_dataset) / fov_path / "tracks.csv"
    if not tracks_path.exists():
        pytest.skip(f"Tracks file not found at {tracks_path}")

    tracks_df = pd.read_csv(tracks_path)

    if len(tracks_df) == 0:
        pytest.skip("No tracks found in dataset")

    # Test with first track
    first_track_id = tracks_df["track_id"].iloc[0]

    df = DynamicFeatures(tracks_df)
    features_df = df.compute_all_features(first_track_id)

    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 1

    # Check expected columns
    expected_cols = {
        "mean_velocity",
        "max_velocity",
        "min_velocity",
        "std_velocity",
        "total_distance",
        "net_displacement",
        "directional_persistence",
        "mean_angular_velocity",
        "max_angular_velocity",
        "std_angular_velocity",
        "instantaneous_velocity",
    }
    assert set(features_df.columns) == expected_cols

    for col in features_df.columns:
        if col != "instantaneous_velocity":
            assert np.isfinite(features_df[col].iloc[0]), f"Feature {col} is not finite"
