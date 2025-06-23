import numpy as np
import pandas as pd
import pytest
from skimage import measure

from viscy.representation.evaluation.feature import CellFeatures, DynamicFeatures


@pytest.fixture
def simple_image():
    """Create a simple test image with a known pattern"""
    axis = [0.0, 0.2, 0.4, 0.2, 0.0]
    x, y = np.meshgrid(axis, axis)

    image = x + y

    return image


@pytest.fixture
def simple_mask():
    """Create a simple binary mask."""
    from skimage.morphology import disk

    mask = disk(2)
    return mask


def test_intensity_features(simple_image):
    """Test computation of intensity-based features with known input."""
    cell_features = CellFeatures(simple_image)
    cell_features.compute_intensity_features()

    features = cell_features.intensity_features

    assert np.isclose(features["mean_intensity"], 0.32, atol=1e-6)
    assert np.isclose(
        features["std_dev"], 0.21166010488516723, atol=1e-6
    )  # Actual std dev
    assert np.isclose(features["min_intensity"], 0.0)
    assert np.isclose(features["max_intensity"], 0.8)

    assert not np.isnan(features["kurtosis"])
    assert not np.isnan(features["skewness"])
    assert features["spectral_entropy"] > 0
    assert features["iqr"] > 0


def test_texture_features(simple_image):
    """Test computation of texture features with known input."""
    cell_features = CellFeatures(simple_image)
    cell_features.compute_texture_features()

    features = cell_features.texture_features

    assert features["contrast"] >= 0
    assert features["dissimilarity"] >= 0
    assert 0 <= features["homogeneity"] <= 1

    assert features["spectral_entropy"] > 0
    assert features["entropy"] > 0
    assert features["texture"] >= 0


def test_morphology_features(simple_image, simple_mask):
    """Test computation of morphological features with known input."""
    # Convert mask to labeled image
    labeled_mask = measure.label(simple_mask.astype(int))

    cell_features = CellFeatures(simple_image, labeled_mask)
    cell_features.compute_morphology_features()

    features = cell_features.morphology_features

    assert features["area"] == 13  # Number of True pixels in mask
    assert features["perimeter"] > 0
    assert features["perimeter_area_ratio"] > 0

    assert 0 <= features["eccentricity"] <= 1

    assert features["intensity_localization"] > 0
    assert features["masked_intensity"] > 0
    assert features["masked_area"] == 13


def test_symmetry_descriptor(simple_image):
    """Test computation of symmetry features with known input."""
    cell_features = CellFeatures(simple_image)
    cell_features.compute_symmetry_descriptor()

    features = cell_features.symmetry_descriptor

    assert features["zernike_std"] >= 0
    assert not np.isnan(features["zernike_mean"])

    assert features["radial_intensity_gradient"] < 0


def test_all_features(simple_image, simple_mask):
    """Test computation of all features together."""
    labeled_mask = measure.label(simple_mask.astype(int))

    cell_features = CellFeatures(simple_image, labeled_mask)
    features_df = cell_features.compute_all_features()

    # Test that all feature types are present
    assert "mean_intensity" in features_df.columns
    assert "contrast" in features_df.columns
    assert "area" in features_df.columns
    assert "zernike_std" in features_df.columns

    # Test that no features are NaN
    assert not features_df.isna().any().any()


def test_edge_cases():
    """Test behavior with edge cases."""
    constant_image = np.ones((5, 5))
    cell_features = CellFeatures(constant_image)
    cell_features.compute_intensity_features()

    features = cell_features.intensity_features
    assert features["std_dev"] == 0
    assert np.isnan(features["kurtosis"])  # Kurtosis is undefined for constant values
    assert np.isnan(features["skewness"])  # Skewness is undefined for constant values

    empty_mask = np.zeros((5, 5), dtype=int)
    cell_features = CellFeatures(constant_image, empty_mask)
    with pytest.raises(AssertionError):
        cell_features.compute_morphology_features()


def test_normalization(simple_image, simple_mask):
    """Test that features are invariant to intensity scaling."""
    # Convert mask to labeled image
    labeled_mask = measure.label(simple_mask.astype(int))

    # Compute features for original image
    cell_features1 = CellFeatures(simple_image, labeled_mask)
    features1 = cell_features1.compute_all_features()

    # Compute features for scaled image
    scaled_image = simple_image * 2.0
    cell_features2 = CellFeatures(scaled_image, labeled_mask)
    features2 = cell_features2.compute_all_features()

    # Compare features that should be invariant to scaling
    assert np.allclose(features1["eccentricity"], features2["eccentricity"])
    assert np.allclose(
        features1["perimeter_area_ratio"], features2["perimeter_area_ratio"]
    )
    assert np.allclose(features1["zernike_std"], features2["zernike_std"])


@pytest.fixture
def simple_track():
    """Create a simple track with known properties."""
    # Create a track moving in a straight line
    t = np.array([0, 1, 2, 3, 4])
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 0, 0, 0, 0])
    track_id = np.array(["1"] * 5)

    return pd.DataFrame({"track_id": track_id, "t": t, "x": x, "y": y})


def test_velocity_features(simple_track):
    """Test computation of velocity features with known input."""
    dynamic_features = DynamicFeatures(simple_track)
    features_df = dynamic_features.compute_all_features("1")

    # Test velocity features
    assert "mean_velocity" in features_df.columns
    assert "max_velocity" in features_df.columns
    assert "min_velocity" in features_df.columns
    assert "std_velocity" in features_df.columns

    # For straight line motion at constant speed:
    assert np.isclose(features_df["mean_velocity"].iloc[0], 0.8, atol=1e-6)
    assert np.isclose(
        features_df["std_velocity"].iloc[0], 0.4, atol=1e-6
    )  # Actual std dev


def test_displacement_features(simple_track):
    """Test computation of displacement features with known input."""
    dynamic_features = DynamicFeatures(simple_track)
    features_df = dynamic_features.compute_all_features("1")

    assert "total_distance" in features_df.columns
    assert "net_displacement" in features_df.columns
    assert "directional_persistence" in features_df.columns

    # For straight line motion:
    assert np.isclose(
        features_df["total_distance"].iloc[0], 4.0, atol=1e-6
    )  # Total distance
    assert np.isclose(
        features_df["net_displacement"].iloc[0], 4.0, atol=1e-6
    )  # Net displacement
    assert np.isclose(
        features_df["directional_persistence"].iloc[0], 1.0, atol=1e-6
    )  # Perfect persistence


def test_angular_features(simple_track):
    """Test computation of angular features with known input."""
    dynamic_features = DynamicFeatures(simple_track)
    features_df = dynamic_features.compute_all_features("1")

    assert "mean_angular_velocity" in features_df.columns
    assert "max_angular_velocity" in features_df.columns
    assert "std_angular_velocity" in features_df.columns

    # For straight line motion:
    assert np.isclose(
        features_df["mean_angular_velocity"].iloc[0], 1.4142136207497351e-05, atol=1e-6
    )  # Actual small value
    assert np.isclose(features_df["std_angular_velocity"].iloc[0], 0.0, atol=1e-6)


def test_tracking_edge_cases():
    """Test behavior with edge cases in tracking data."""
    # Test with single point
    single_point = pd.DataFrame({"track_id": ["1"], "t": [0], "x": [0], "y": [0]})
    dynamic_features = DynamicFeatures(single_point)
    features_df = dynamic_features.compute_all_features("1")

    assert np.isclose(features_df["mean_velocity"].iloc[0], 0.0, atol=1e-6)
    assert np.isclose(features_df["total_distance"].iloc[0], 0.0, atol=1e-6)
    assert np.isclose(features_df["mean_angular_velocity"].iloc[0], 0.0, atol=1e-6)

    two_points = pd.DataFrame(
        {"track_id": ["1", "1"], "t": [0, 1], "x": [0, 1], "y": [0, 1]}
    )
    dynamic_features = DynamicFeatures(two_points)
    features_df = dynamic_features.compute_all_features("1")

    assert np.isclose(
        features_df["mean_velocity"].iloc[0], 0.7071067811865476, atol=1e-6
    )
    assert np.isclose(features_df["mean_angular_velocity"].iloc[0], 0.0, atol=1e-6)


def test_tracking_invalid_data():
    """Test behavior with invalid tracking data."""
    # Test with missing columns
    invalid_data = pd.DataFrame({"track_id": ["1"], "t": [0]})
    with pytest.raises(ValueError):
        DynamicFeatures(invalid_data)

    # Test with non-numeric coordinates
    invalid_data = pd.DataFrame(
        {"track_id": ["1"], "t": [0], "x": ["invalid"], "y": [0]}
    )
    with pytest.raises(ValueError):
        DynamicFeatures(invalid_data)
