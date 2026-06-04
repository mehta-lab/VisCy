"""Tests for airtable_utils.database."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestAirtableDatasetsInit:
    """Test AirtableDatasets constructor and env var handling."""

    def test_init_with_env_vars(self, mock_env, mock_api):
        """Constructor succeeds when both env vars are set."""
        from airtable_utils.database import AirtableDatasets

        AirtableDatasets()
        # Api was called with the fake key
        mock_api.assert_called_once_with("patFAKEKEY123")
        # .table() is called twice: once for Datasets, once for Marker Registry
        mock_api.return_value.table.assert_any_call("appFAKEBASE456", "Datasets")

    def test_init_raises_when_api_key_missing(self, monkeypatch):
        """ValueError is raised when AIRTABLE_API_KEY is not set."""
        monkeypatch.delenv("AIRTABLE_API_KEY", raising=False)
        monkeypatch.setenv("AIRTABLE_BASE_ID", "appFAKEBASE456")

        from airtable_utils.database import AirtableDatasets

        with patch("airtable_utils.database.Api"):
            with pytest.raises(ValueError, match="AIRTABLE_API_KEY"):
                AirtableDatasets()

    def test_init_raises_when_base_id_missing(self, monkeypatch):
        """ValueError is raised when AIRTABLE_BASE_ID is not set."""
        monkeypatch.setenv("AIRTABLE_API_KEY", "patFAKEKEY123")
        monkeypatch.delenv("AIRTABLE_BASE_ID", raising=False)

        from airtable_utils.database import AirtableDatasets

        with patch("airtable_utils.database.Api"):
            with pytest.raises(ValueError, match="AIRTABLE_BASE_ID"):
                AirtableDatasets()

    def test_init_raises_when_both_missing(self, monkeypatch):
        """ValueError is raised when both env vars are missing."""
        monkeypatch.delenv("AIRTABLE_API_KEY", raising=False)
        monkeypatch.delenv("AIRTABLE_BASE_ID", raising=False)

        from airtable_utils.database import AirtableDatasets

        with patch("airtable_utils.database.Api"):
            with pytest.raises(ValueError):
                AirtableDatasets()

    def test_init_raises_when_api_key_empty(self, monkeypatch):
        """ValueError is raised when AIRTABLE_API_KEY is set to empty string."""
        monkeypatch.setenv("AIRTABLE_API_KEY", "")
        monkeypatch.setenv("AIRTABLE_BASE_ID", "appFAKEBASE456")

        from airtable_utils.database import AirtableDatasets

        with patch("airtable_utils.database.Api"):
            with pytest.raises(ValueError, match="AIRTABLE_API_KEY"):
                AirtableDatasets()

    def test_no_constructor_params_accepted(self):
        """Constructor does not accept api_key or base_id parameters."""
        import inspect

        from airtable_utils.database import AirtableDatasets

        sig = inspect.signature(AirtableDatasets.__init__)
        params = list(sig.parameters.keys())
        # Only 'self' should be a parameter
        assert params == ["self"], (
            f"Expected only 'self', got {params}. api_key/base_id must not be constructor parameters."
        )


# ---------------------------------------------------------------------------
# get_unique_datasets
# ---------------------------------------------------------------------------


class TestGetUniqueDatasets:
    """Test AirtableDatasets.get_unique_datasets()."""

    def test_returns_sorted_unique_names(self, airtable_datasets, mock_table, dataset_names_records):
        mock_table.all.return_value = dataset_names_records
        result = airtable_datasets.get_unique_datasets()
        mock_table.all.assert_called_once_with(fields=["dataset"])
        assert result == ["dataset_alpha", "dataset_beta"]

    def test_empty_table_returns_empty_list(self, airtable_datasets, mock_table):
        mock_table.all.return_value = []
        result = airtable_datasets.get_unique_datasets()
        assert result == []

    def test_skips_records_without_dataset_field(self, airtable_datasets, mock_table):
        mock_table.all.return_value = [
            {"id": "rec001", "fields": {"dataset": "alpha"}},
            {"id": "rec002", "fields": {}},  # missing dataset
            {"id": "rec003", "fields": {"dataset": "beta"}},
        ]
        result = airtable_datasets.get_unique_datasets()
        assert result == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# get_dataset_records
# ---------------------------------------------------------------------------


class TestGetDatasetRecords:
    """Test AirtableDatasets.get_dataset_records()."""

    def test_returns_dataset_records(self, airtable_datasets, mock_table, sample_airtable_records):
        mock_table.all.return_value = [sample_airtable_records[0]]
        result = airtable_datasets.get_dataset_records("dataset_alpha")
        mock_table.all.assert_called_once_with(formula="{dataset} = 'dataset_alpha'")
        assert len(result) == 1
        assert result[0].dataset == "dataset_alpha"
        assert result[0].well_id == "A/1"
        assert result[0].record_id == "rec001"

    def test_empty_result(self, airtable_datasets, mock_table):
        mock_table.all.return_value = []
        result = airtable_datasets.get_dataset_records("nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# list_records
# ---------------------------------------------------------------------------


class TestListRecords:
    """Test AirtableDatasets.list_records()."""

    def test_returns_dataframe(self, airtable_datasets, mock_table, sample_airtable_records):
        mock_table.all.return_value = sample_airtable_records
        df = airtable_datasets.list_records()
        mock_table.all.assert_called_once_with()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df["dataset"]) == ["dataset_alpha", "dataset_beta"]

    def test_with_filter_formula(self, airtable_datasets, mock_table, sample_airtable_records):
        mock_table.all.return_value = [sample_airtable_records[0]]
        formula = "{cell_type} = 'HEK293T'"
        df = airtable_datasets.list_records(filter_formula=formula)
        mock_table.all.assert_called_once_with(formula=formula)
        assert len(df) == 1

    def test_without_filter_formula(self, airtable_datasets, mock_table):
        mock_table.all.return_value = []
        df = airtable_datasets.list_records(filter_formula=None)
        mock_table.all.assert_called_once_with()
        assert len(df) == 0

    def test_dataframe_columns(self, airtable_datasets, mock_table, sample_airtable_records):
        mock_table.all.return_value = [sample_airtable_records[0]]
        df = airtable_datasets.list_records()
        expected_cols = {
            "dataset",
            "well_id",
            "fov",
            "cell_type",
            "cell_state",
            "cell_line",
            "marker",
            "organelle",
            "perturbation",
            "hours_post_perturbation",
            "moi",
            "time_interval_min",
            "seeding_density",
            "treatment_concentration_nm",
            "channel_names",
            "channel_markers",
            *(f"channel_{i}_{attr}" for i in range(8) for attr in ("name", "marker")),
            "data_path",
            "tracks_path",
            "fluorescence_modality",
            "microscope",
            "labelfree_modality",
            "treatment",
            "hours_post_treatment",
            "t_shape",
            "c_shape",
            "z_shape",
            "y_shape",
            "x_shape",
            "pixel_size_xy_um",
            "pixel_size_z_um",
            "record_id",
        }
        assert set(df.columns) == expected_cols


# ---------------------------------------------------------------------------
# batch_delete
# ---------------------------------------------------------------------------


class TestBatchDelete:
    """Test AirtableDatasets.batch_delete()."""

    def test_delegates_to_table(self, airtable_datasets, mock_table):
        mock_table.batch_delete.return_value = [{"id": "rec001", "deleted": True}]
        result = airtable_datasets.batch_delete(["rec001"])
        mock_table.batch_delete.assert_called_once_with(["rec001"])
        assert result == [{"id": "rec001", "deleted": True}]

    def test_passes_multiple_ids(self, airtable_datasets, mock_table):
        ids = ["rec001", "rec002", "rec003"]
        mock_table.batch_delete.return_value = []
        airtable_datasets.batch_delete(ids)
        mock_table.batch_delete.assert_called_once_with(ids)
