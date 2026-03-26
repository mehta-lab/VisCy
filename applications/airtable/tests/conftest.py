"""Shared fixtures for airtable_utils tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Sample Airtable API response records
# ---------------------------------------------------------------------------

SAMPLE_AIRTABLE_RECORDS = [
    {
        "id": "rec001",
        "fields": {
            "dataset": "dataset_alpha",
            "well_id": "A/1",
            "fov": "000000",
            "cell_type": {"name": "HEK293T"},
            "cell_state": {"name": "healthy"},
            "cell_line": [{"name": "HEK293T-H2B-mCherry"}],
            "organelle": {"name": "nucleus"},
            "perturbation": {"name": "DMSO"},
            "hours_post_perturbation": 24.0,
            "moi": None,
            "time_interval_min": 5.0,
            "seeding_density": 50000,
            "treatment_concentration_nm": 100.0,
            "channel_0_name": "Phase3D",
            "channel_0_marker": {"name": "Membrane"},
            "channel_1_name": "raw GFP EX488 EM525-45",
            "channel_1_marker": {"name": "Endoplasmic Reticulum"},
            "channel_2_name": None,
            "channel_2_marker": None,
            "channel_3_name": None,
            "channel_3_marker": None,
            "data_path": "/hpc/datasets/alpha.zarr",
            "fluorescence_modality": {"name": "widefield"},
            "t_shape": 50,
            "c_shape": 2,
            "z_shape": 30,
            "y_shape": 2048,
            "x_shape": 2048,
        },
    },
    {
        "id": "rec002",
        "fields": {
            "dataset": "dataset_beta",
            "well_id": "B/2",
            "fov": "000001",
            "cell_type": "A549",
            "cell_state": "infected",
            "cell_line": None,
            "organelle": "mitochondria",
            "perturbation": "ZIKV",
            "hours_post_perturbation": 48.0,
            "moi": 0.5,
            "time_interval_min": 10.0,
            "seeding_density": None,
            "treatment_concentration_nm": None,
            "channel_0_name": "BF_LED_Matrix_Full",
            "channel_0_marker": None,
            "channel_1_name": "nuclei_prediction",
            "channel_1_marker": {"name": "Nucleus"},
            "channel_2_name": None,
            "channel_2_marker": None,
            "channel_3_name": None,
            "channel_3_marker": None,
            "data_path": "/hpc/datasets/beta.zarr",
            "fluorescence_modality": None,
            "t_shape": 100,
            "c_shape": 2,
            "z_shape": 15,
            "y_shape": 1024,
            "x_shape": 1024,
        },
    },
]

DATASET_NAMES_RECORDS = [
    {"id": "rec001", "fields": {"dataset": "dataset_alpha"}},
    {"id": "rec002", "fields": {"dataset": "dataset_beta"}},
    {"id": "rec003", "fields": {"dataset": "dataset_alpha"}},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_airtable_records():
    """Return sample Airtable API response records."""
    return SAMPLE_AIRTABLE_RECORDS


@pytest.fixture()
def dataset_names_records():
    """Return sample dataset-names-only records."""
    return DATASET_NAMES_RECORDS


@pytest.fixture()
def mock_env(monkeypatch):
    """Set required Airtable environment variables."""
    monkeypatch.setenv("AIRTABLE_API_KEY", "patFAKEKEY123")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "appFAKEBASE456")


@pytest.fixture()
def mock_table():
    """Return a MagicMock that stands in for ``pyairtable.Table``."""
    return MagicMock()


@pytest.fixture()
def mock_api(mock_table):
    """Patch ``pyairtable.Api`` so it returns ``mock_table`` on ``.table()``."""
    with patch("airtable_utils.database.Api") as api_cls:
        api_instance = MagicMock()
        api_instance.table.return_value = mock_table
        api_cls.return_value = api_instance
        yield api_cls


@pytest.fixture()
def airtable_datasets(mock_env, mock_api, mock_table):
    """Return an ``AirtableDatasets`` instance backed by mocks."""
    from airtable_utils.database import AirtableDatasets

    ds = AirtableDatasets()
    return ds
