"""Tests for Airtable database module using mocks."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from viscy.airtable.database import AirtableManager, CollectionDataset, Collections
from viscy.airtable.schemas import DatasetRecord

# ============================================================================
# Tests for CollectionDataset dataclass
# ============================================================================


def test_collection_dataset_creation():
    """Test creating a CollectionDataset."""
    dataset = CollectionDataset(
        data_path="/hpc/data/plate.zarr",
        tracks_path="/hpc/tracks/plate.zarr",
        fov_names=["B/3/0", "B/3/1", "B/4/0"],
    )

    assert dataset.data_path == "/hpc/data/plate.zarr"
    assert dataset.tracks_path == "/hpc/tracks/plate.zarr"
    assert len(dataset) == 3
    assert len(dataset.fov_names) == 3


def test_collection_dataset_fov_paths():
    """Test generating FOV paths."""
    dataset = CollectionDataset(
        data_path="/hpc/data/plate.zarr",
        tracks_path="/hpc/tracks/plate.zarr",
        fov_names=["B/3/0", "B/3/1"],
    )

    fov_paths = dataset.fov_paths
    assert fov_paths == [
        "/hpc/data/plate.zarr/B/3/0",
        "/hpc/data/plate.zarr/B/3/1",
    ]


def test_collection_dataset_exists(tmp_path):
    """Test checking if paths exist."""
    # Create actual directories
    data_path = tmp_path / "data.zarr"
    tracks_path = tmp_path / "tracks.zarr"
    data_path.mkdir()
    tracks_path.mkdir()

    dataset = CollectionDataset(
        data_path=str(data_path),
        tracks_path=str(tracks_path),
        fov_names=["0", "1"],
    )

    assert dataset.exists() is True

    # Test with non-existent paths
    dataset_bad = CollectionDataset(
        data_path="/nonexistent/data.zarr",
        tracks_path="/nonexistent/tracks.zarr",
        fov_names=["0"],
    )

    assert dataset_bad.exists() is False


def test_collection_dataset_validate(tmp_path):
    """Test validation raises error for missing paths."""
    data_path = tmp_path / "data.zarr"
    data_path.mkdir()

    # Missing tracks path
    dataset = CollectionDataset(
        data_path=str(data_path),
        tracks_path="/nonexistent/tracks.zarr",
        fov_names=["0"],
    )

    with pytest.raises(FileNotFoundError, match="Tracks path not found"):
        dataset.validate()


# ============================================================================
# Tests for Collections dataclass
# ============================================================================


def test_collections_creation():
    """Test creating a Collections object."""
    datasets = [
        CollectionDataset("/data1.zarr", "/tracks1.zarr", ["0", "1"]),
        CollectionDataset("/data2.zarr", "/tracks2.zarr", ["0", "1", "2"]),
    ]

    collection = Collections(
        name="test_collection",
        version="0.0.1",
        datasets=datasets,
    )

    assert collection.name == "test_collection"
    assert collection.version == "0.0.1"
    assert len(collection) == 2
    assert collection.total_fovs == 5  # 2 + 3


def test_collections_iteration():
    """Test iterating over datasets in a collection."""
    datasets = [
        CollectionDataset("/data1.zarr", "/tracks1.zarr", ["0"]),
        CollectionDataset("/data2.zarr", "/tracks2.zarr", ["0", "1"]),
    ]

    collection = Collections("test", "0.0.1", datasets)

    dataset_list = list(collection)
    assert len(dataset_list) == 2
    assert dataset_list[0].data_path == "/data1.zarr"
    assert dataset_list[1].data_path == "/data2.zarr"


def test_collections_validate(tmp_path):
    """Test validation checks all dataset paths."""
    # Create valid paths for first dataset
    data1 = tmp_path / "data1.zarr"
    tracks1 = tmp_path / "tracks1.zarr"
    data1.mkdir()
    tracks1.mkdir()

    datasets = [
        CollectionDataset(str(data1), str(tracks1), ["0"]),
        CollectionDataset("/bad/data.zarr", "/bad/tracks.zarr", ["0"]),  # Invalid
    ]

    collection = Collections("test", "0.0.1", datasets)

    with pytest.raises(FileNotFoundError, match="Data path not found"):
        collection.validate()


# ============================================================================
# Tests for AirtableManager
# ============================================================================


@pytest.fixture
def mock_airtable_api():
    """Create a mock Airtable API."""
    api = Mock()
    api.table.return_value = Mock()
    return api


@pytest.fixture
def airtable_manager(mock_airtable_api):
    """Create an AirtableManager with mocked API."""
    with patch("viscy.airtable.database.Api", return_value=mock_airtable_api):
        with patch.dict("os.environ", {"AIRTABLE_API_KEY": "test_key"}):
            manager = AirtableManager(base_id="test_base_id")
    return manager


def test_airtable_manager_init_requires_api_key():
    """Test that AirtableManager requires an API key."""
    with patch.dict("os.environ", {}, clear=True):
        # Remove AIRTABLE_API_KEY from env
        with pytest.raises(ValueError, match="Airtable API key required"):
            AirtableManager(base_id="test_base")


def test_airtable_manager_init_with_explicit_key():
    """Test initializing with explicit API key."""
    with patch("viscy.airtable.database.Api") as mock_api_class:
        AirtableManager(base_id="test_base", api_key="explicit_key")

        # Verify Api was called with the explicit key
        mock_api_class.assert_called_once_with("explicit_key")


def test_airtable_manager_init_with_env_key():
    """Test initializing with API key from environment."""
    with patch("viscy.airtable.database.Api") as mock_api_class:
        with patch.dict("os.environ", {"AIRTABLE_API_KEY": "env_key"}):
            AirtableManager(base_id="test_base")

            mock_api_class.assert_called_once_with("env_key")


def test_register_dataset(airtable_manager):
    """Test registering a single dataset."""
    dataset = DatasetRecord(
        dataset_name="test_plate",
        well_id="B_3",
        fov_name="0",
        data_path="/hpc/data/test.zarr/B/3/0",
    )

    # Mock the create response
    airtable_manager.datasets_table.create.return_value = {"id": "rec123"}

    record_id = airtable_manager.register_dataset(dataset)

    assert record_id == "rec123"
    airtable_manager.datasets_table.create.assert_called_once()

    # Verify the data passed to Airtable
    call_args = airtable_manager.datasets_table.create.call_args[0][0]
    assert call_args["Dataset"] == "test_plate"
    assert call_args["Well ID"] == "B_3"
    assert call_args["FOV"] == "0"


def test_register_datasets_multiple(airtable_manager):
    """Test registering multiple datasets."""
    datasets = [
        DatasetRecord(
            dataset_name="plate",
            well_id="B_3",
            fov_name="0",
            data_path="/hpc/data/plate.zarr/B/3/0",
        ),
        DatasetRecord(
            dataset_name="plate",
            well_id="B_3",
            fov_name="1",
            data_path="/hpc/data/plate.zarr/B/3/1",
        ),
    ]

    # Mock create responses
    airtable_manager.datasets_table.create.side_effect = [
        {"id": "rec123"},
        {"id": "rec456"},
    ]

    record_ids = airtable_manager.register_datasets(datasets)

    assert record_ids == ["rec123", "rec456"]
    assert airtable_manager.datasets_table.create.call_count == 2


def test_create_collection_from_datasets(airtable_manager):
    """Test creating a collection from dataset FOV IDs."""
    # Mock list_collections to return empty (no duplicates)
    with patch.object(
        airtable_manager, "list_collections", return_value=pd.DataFrame()
    ):
        # Mock the datasets_table.all() response for FOV lookup
        def mock_all(formula=None):
            if formula == "{FOV_ID}='plate_B_3_0'":
                return [{"id": "rec1", "fields": {"FOV_ID": "plate_B_3_0"}}]
            elif formula == "{FOV_ID}='plate_B_3_1'":
                return [{"id": "rec2", "fields": {"FOV_ID": "plate_B_3_1"}}]
            return []

        airtable_manager.datasets_table.all.side_effect = mock_all

        # Mock the collections_table.create response
        airtable_manager.collections_table.create.return_value = {"id": "col123"}

        collection_id = airtable_manager.create_collection_from_datasets(
            collection_name="test_collection",
            fov_ids=["plate_B_3_0", "plate_B_3_1"],
            version="0.0.1",
            purpose="training",
        )

        assert collection_id == "col123"

        # Verify the collection was created with correct data
        airtable_manager.collections_table.create.assert_called_once()
        call_args = airtable_manager.collections_table.create.call_args[0][0]

        assert call_args["name"] == "test_collection"
        assert call_args["version"] == "0.0.1"
        assert call_args["purpose"] == "training"
        assert call_args["datasets"] == ["rec1", "rec2"]  # Linked record IDs


def test_create_collection_validates_version_format(airtable_manager):
    """Test that collection creation validates version format."""
    with pytest.raises(ValueError, match="semantic version format"):
        airtable_manager.create_collection_from_datasets(
            collection_name="test",
            fov_ids=["fov1"],
            version="invalid_version",  # Should be like "0.0.1"
        )


def test_list_datasets_returns_pydantic_models(airtable_manager):
    """Test listing datasets returns Pydantic models."""
    # Mock the all() response
    airtable_manager.datasets_table.all.return_value = [
        {
            "id": "rec123",
            "fields": {
                "Dataset": "test_plate",
                "Well ID": "B_3",
                "FOV": "0",
                "Data path": "/hpc/data/test.zarr/B/3/0",
            },
        }
    ]

    datasets = airtable_manager.list_datasets(as_pydantic=True)

    assert len(datasets) == 1
    assert isinstance(datasets[0], DatasetRecord)
    assert datasets[0].dataset_name == "test_plate"
    assert datasets[0].well_id == "B_3"


def test_list_datasets_returns_dicts(airtable_manager):
    """Test listing datasets can return raw dictionaries."""
    airtable_manager.datasets_table.all.return_value = [
        {
            "id": "rec123",
            "fields": {"Dataset": "test_plate"},
        }
    ]

    datasets = airtable_manager.list_datasets(as_dataframe=False, as_pydantic=False)

    assert len(datasets) == 1
    assert isinstance(datasets[0], dict)
    assert datasets[0]["Dataset"] == "test_plate"


def test_get_collection_data_paths(airtable_manager):
    """Test getting data paths from a collection."""
    # Mock list_collections to return collection info
    collections_df = pd.DataFrame(
        [
            {
                "id": "col123",
                "name": "test_collection",
                "version": "0.0.1",
                "datasets": ["rec1", "rec2"],
            }
        ]
    )

    with patch.object(
        airtable_manager, "list_collections", return_value=collections_df
    ):
        # Mock datasets_table.get() for individual record fetches
        def mock_get(record_id):
            if record_id == "rec1":
                return {
                    "id": "rec1",
                    "fields": {
                        "Dataset": "plate",
                        "Data path": "/hpc/data/plate.zarr",
                        "Well ID": "B_3",
                        "FOV": "0",
                    },
                }
            elif record_id == "rec2":
                return {
                    "id": "rec2",
                    "fields": {
                        "Dataset": "plate",
                        "Data path": "/hpc/data/plate.zarr",
                        "Well ID": "B_3",
                        "FOV": "1",
                    },
                }
            return None

        airtable_manager.datasets_table.get.side_effect = mock_get

        collection = airtable_manager.get_dataset_paths(
            collection_name="test_collection", version="0.0.1"
        )

        assert isinstance(collection, Collections)
        assert collection.name == "test_collection"
        assert collection.version == "0.0.1"
        assert collection.total_fovs == 2


def test_log_model_training(airtable_manager):
    """Test logging model training to Models table."""
    airtable_manager.models_table.create.return_value = {"id": "model123"}

    # Mock collections_table.get() for update operation
    airtable_manager.collections_table.get.return_value = {
        "id": "col123",
        "fields": {"models_trained": ""},
    }

    model_id = airtable_manager.log_model_training(
        collection_id="col123",
        wandb_run_id="run456",
        model_name="contrastive-v1",
        checkpoint_path="/hpc/models/model.ckpt",
        trained_by="test_user",
        metrics={"val_loss": 0.15},
    )

    assert model_id == "model123"

    # Verify the model record was created correctly
    airtable_manager.models_table.create.assert_called_once()
    call_args = airtable_manager.models_table.create.call_args[0][0]

    assert call_args["collection"] == ["col123"]
    assert call_args["wandb_run_id"] == "run456"  # Fixed: was mlflow_run_id
    assert call_args["model_name"] == "contrastive-v1"
    assert call_args["checkpoint_path"] == "/hpc/models/model.ckpt"
    assert call_args["val_loss"] == 0.15


def test_list_collections(airtable_manager):
    """Test listing all collections."""
    airtable_manager.collections_table.all.return_value = [
        {
            "id": "col1",
            "fields": {
                "name": "collection_1",
                "version": "0.0.1",
                "purpose": "training",
            },
        },
        {
            "id": "col2",
            "fields": {
                "name": "collection_2",
                "version": "0.0.2",
                "purpose": "validation",
            },
        },
    ]

    # Test returning DataFrame (default)
    collections_df = airtable_manager.list_collections()

    assert isinstance(collections_df, pd.DataFrame)
    assert len(collections_df) == 2
    assert collections_df.iloc[0]["name"] == "collection_1"
    assert collections_df.iloc[1]["purpose"] == "validation"

    # Test returning list of dicts
    collections_list = airtable_manager.list_collections(as_dataframe=False)

    assert isinstance(collections_list, list)
    assert len(collections_list) == 2
    assert collections_list[0]["name"] == "collection_1"
    assert collections_list[1]["purpose"] == "validation"


def test_get_models_for_collection(airtable_manager):
    """Test getting all models trained on a specific collection."""
    airtable_manager.models_table.all.return_value = [
        {
            "id": "model1",
            "fields": {
                "model_name": "model-v1",
                "collection": ["col123"],
                "trained_date": "2026-01-12",
            },
        },
        {
            "id": "model2",
            "fields": {
                "model_name": "model-v2",
                "collection": ["col123"],
                "trained_date": "2026-01-13",
            },
        },
    ]

    # Test returning DataFrame (default, sorted by trained_date descending)
    models_df = airtable_manager.get_models_for_collection(collection_id="col123")

    assert isinstance(models_df, pd.DataFrame)
    assert len(models_df) == 2
    # Latest model comes first due to descending sort
    assert models_df.iloc[0]["model_name"] == "model-v2"
    assert models_df.iloc[0]["trained_date"] == "2026-01-13"
    assert models_df.iloc[1]["model_name"] == "model-v1"
    assert models_df.iloc[1]["trained_date"] == "2026-01-12"

    # Test returning list of dicts
    models_list = airtable_manager.get_models_for_collection(
        collection_id="col123", as_dataframe=False
    )

    assert isinstance(models_list, list)
    assert len(models_list) == 2
    # List doesn't get sorted automatically, so order is preserved from input
    assert models_list[0]["model_name"] == "model-v1"
    assert models_list[1]["model_name"] == "model-v2"
