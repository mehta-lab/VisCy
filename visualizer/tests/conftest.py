"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_tracks_df():
    """Create a sample tracks DataFrame for testing."""
    return pd.DataFrame(
        {
            "track_id": [1, 1, 1, 2, 2, 2, 3, 3],
            "t": [0, 1, 2, 0, 1, 2, 0, 1],
            "fov_name": [
                "A/1/0",
                "A/1/0",
                "A/1/0",
                "A/1/0",
                "A/1/0",
                "A/1/0",
                "A/2/0",
                "A/2/0",
            ],
            "y": [10, 11, 12, 20, 21, 22, 30, 31],
            "x": [100, 101, 102, 200, 201, 202, 300, 301],
            "PHATE1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "PHATE2": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            "annotation": [
                "uninfected",
                "uninfected",
                "uninfected",
                "infected",
                "infected",
                "infected",
                "uninfected",
                "uninfected",
            ],
            "dataset_id": [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )


@pytest.fixture
def sample_multi_dataset_tracks_df():
    """Create a sample multi-dataset tracks DataFrame."""
    df1 = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "t": [0, 1, 0, 1],
            "fov_name": ["A/1/0", "A/1/0", "A/1/0", "A/1/0"],
            "y": [10, 11, 20, 21],
            "x": [100, 101, 200, 201],
            "PHATE1": [0.1, 0.2, 0.3, 0.4],
            "PHATE2": [1.1, 1.2, 1.3, 1.4],
            "annotation": ["uninfected", "uninfected", "infected", "infected"],
            "dataset_id": [0, 0, 0, 0],
        }
    )

    df2 = pd.DataFrame(
        {
            "track_id": [10, 10, 20, 20],
            "t": [0, 1, 0, 1],
            "fov_name": ["B/1/0", "B/1/0", "B/1/0", "B/1/0"],
            "y": [30, 31, 40, 41],
            "x": [300, 301, 400, 401],
            "PHATE1": [0.5, 0.6, 0.7, 0.8],
            "PHATE2": [1.5, 1.6, 1.7, 1.8],
            "annotation": ["uninfected", "uninfected", "infected", "infected"],
            "dataset_id": [1, 1, 1, 1],
        }
    )

    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def sample_embedding():
    """Create a sample PHATE embedding array."""
    return np.random.randn(100, 2)


@pytest.fixture
def sample_features():
    """Create a sample feature matrix."""
    return np.random.randn(100, 10)


@pytest.fixture
def sample_fov_names():
    """Create a sample list of FOV names."""
    return ["A/1/0", "A/1/0", "A/1/0", "A/2/0", "A/2/0", "B/1/0", "B/1/0", "B/1/0"]


@pytest.fixture
def mock_zarr_array():
    """Create a mock zarr array for testing."""

    class MockZarrArray:
        def __init__(self, shape=(10, 1, 256, 256)):
            self.shape = shape
            self._data = np.random.randint(0, 255, size=shape, dtype=np.uint8)

        def __getitem__(self, key):
            return self._data[key]

    return MockZarrArray()


@pytest.fixture
def mock_zarr_store(mock_zarr_array):
    """Create a mock zarr store for testing."""

    class MockZarrStore:
        def __init__(self):
            self.Phase3D = mock_zarr_array

        def __getitem__(self, key):
            if key == "Phase3D":
                return self.Phase3D
            raise KeyError(f"Key {key} not found")

    return MockZarrStore()
