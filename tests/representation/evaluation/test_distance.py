import numpy as np
import pytest
import xarray as xr

from viscy.representation.evaluation.distance import (
    calculate_cosine_similarity_cell,
    compute_track_displacement,
)


@pytest.fixture
def sample_embedding_dataset():
    """Create a sample embedding dataset for testing."""
    n_samples = 10
    n_features = 5

    features = np.random.rand(n_samples, n_features)
    fov_names = ["fov1"] * 5 + ["fov2"] * 5
    track_ids = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    time_points = [0, 1, 2, 0, 1, 0, 1, 2, 0, 1]

    dataset = xr.Dataset(
        {
            "features": (["sample", "features"], features),
            "fov_name": (["sample"], fov_names),
            "track_id": (["sample"], track_ids),
            "t": (["sample"], time_points),
        }
    )
    return dataset


def test_calculate_cosine_similarity_cell(sample_embedding_dataset):
    """Test cosine similarity calculation for a single track."""
    time_points, similarities = calculate_cosine_similarity_cell(
        sample_embedding_dataset, "fov1", 1
    )

    assert len(time_points) == len(similarities)
    assert len(time_points) == 3
    assert np.isclose(similarities[0], 1.0, atol=1e-6)
    assert all(-1 <= sim <= 1 for sim in similarities)


@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean", "sqeuclidean"])
def test_compute_track_displacement(sample_embedding_dataset, distance_metric):
    """Test track displacement computation with different metrics."""
    result = compute_track_displacement(
        sample_embedding_dataset, distance_metric=distance_metric
    )

    assert isinstance(result, dict)
    assert all(isinstance(tau, int) for tau in result.keys())
    assert all(isinstance(displacements, list) for displacements in result.values())
    assert all(
        all(isinstance(d, (int, float)) and d >= 0 for d in displacements)
        for displacements in result.values()
    )


def test_compute_track_displacement_numerical():
    """Test compute_track_displacement with known embeddings and expected results."""
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    dataset = xr.Dataset(
        {
            "features": (["sample", "features"], features),
            "fov_name": (["sample"], ["fov1", "fov1", "fov1"]),
            "track_id": (["sample"], [1, 1, 1]),
            "t": (["sample"], [0, 1, 2]),
        }
    )
    result_euclidean = compute_track_displacement(dataset, distance_metric="euclidean")

    assert 1 in result_euclidean
    assert 2 in result_euclidean
    assert len(result_euclidean[1]) == 2
    assert len(result_euclidean[2]) == 1

    result_sqeuclidean = compute_track_displacement(
        dataset, distance_metric="sqeuclidean"
    )
    expected_tau1 = [2.0, 1.0]
    expected_tau2 = [1.0]

    assert np.allclose(sorted(result_sqeuclidean[1]), sorted(expected_tau1), atol=1e-10)
    assert np.allclose(result_sqeuclidean[2], expected_tau2, atol=1e-10)

    result_cosine = compute_track_displacement(dataset, distance_metric="cosine")
    expected_cosine_tau1 = [1.0, 1 - 1 / np.sqrt(2)]
    expected_cosine_tau2 = [1 - 1 / np.sqrt(2)]

    assert np.allclose(
        sorted(result_cosine[1]), sorted(expected_cosine_tau1), atol=1e-10
    )
    assert np.allclose(result_cosine[2], expected_cosine_tau2, atol=1e-10)


def test_compute_track_displacement_empty_dataset():
    """Test behavior with empty dataset."""
    empty_dataset = xr.Dataset(
        {
            "features": (["sample", "features"], np.empty((0, 5))),
            "fov_name": (["sample"], []),
            "track_id": (["sample"], []),
            "t": (["sample"], []),
        }
    )

    result = compute_track_displacement(empty_dataset)
    assert result == {}
