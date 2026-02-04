"""FOV-level dataset airtable_db with Airtable."""

import getpass
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from natsort import natsorted
from pyairtable import Api

from viscy.airtable.schemas import DatasetRecord


@dataclass
class CollectionDataset:
    """
    Dataset paths for one HCS plate/zarr store.

    A collection may contain multiple stores, each returned as a separate CollectionDataset.
    """

    data_path: str
    tracks_path: str
    fov_names: list[str]

    def __len__(self) -> int:
        return len(self.fov_names)

    @property
    def fov_paths(self) -> list[str]:
        """Full paths to each FOV: {data_path}/{fov_name}."""
        return [f"{self.data_path}/{fov}" for fov in self.fov_names]

    def exists(self) -> bool:
        """Check if data_path and tracks_path exist."""
        return Path(self.data_path).exists() and Path(self.tracks_path).exists()

    def validate(self) -> None:
        """Raise FileNotFoundError if paths don't exist."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        if not Path(self.tracks_path).exists():
            raise FileNotFoundError(f"Tracks path not found: {self.tracks_path}")


@dataclass
class Collections:
    """All datasets for a collection, potentially across multiple HCS plates."""

    name: str
    version: str
    datasets: list[CollectionDataset]

    def __iter__(self):
        """Iterate over datasets."""
        return iter(self.datasets)

    def __len__(self):
        """Total number of HCS plates."""
        return len(self.datasets)

    @property
    def total_fovs(self) -> int:
        """Total FOVs across all plates."""
        return sum(len(ds) for ds in self.datasets)

    def validate(self) -> None:
        """Validate all dataset paths exist."""
        for ds in self.datasets:
            ds.validate()


# TODO: update the usage examples in the docstrings
# TODO: update the headers to match the Airtable columns and potentially move to separate file
# TODO: Convert these to Pydantic models, so we can easily dump and load from/to Airtable

DATASETS_INDEX = [
    "Dataset",
    "Well ID",
    "FOV",
    "Cell type",
    "Cell state",
    "Cell line",
    "Organelle",
    "Channel-0",
    "Channel-1",
    "Channel-2",
    "Data path",
    "Fluorescence modality",
    "OrganelleBox Infectomics",
    "FOV_ID",
]

MODELS_INDEX = [
    "model_name",
    "model_family",
    "trained_date",
    "trained_by",
    "tensorboard_log",
    "mlflow_run_id",
]

MANIFESTS_INDEX = [
    "name",
    "version",
    "datasets",
    "project",
    "purpose",
    "created_by",
    "created_time",
]


class AirtableManager:
    """
    Unified interface to Airtable for dataset, collection, and model management.

    Use this to:
    - Register individual FOVs from HCS plates
    - Create and manage dataset collections (collections of FOVs)
    - Track model training on collections
    - Query datasets, collections, and models

    Parameters
    ----------
    base_id : str
        Airtable base ID
    api_key : str | None
        Airtable API key. If None, reads from AIRTABLE_API_KEY env var.

    Examples
    --------
    >>> airtable_db = AirtableManager(base_id="appXXXXXXXXXXXXXX")
    >>>
    >>> # Create collection from FOV selection
    >>> collection_id = airtable_db.create_collection_from_datasets(
    ...     collection_name="RPE1_infection_v2",
    ...     fov_ids=["FOV_001", "FOV_002", "FOV_004"],
    ...     version="0.0.1",
    ...     purpose="training"
    ... )
    >>>
    >>> # Track model training
    >>> airtable_db.log_model_training(
    ...     collection_id=collection_id,
    ...     mlflow_run_id="run_123",
    ...     model_name="my_model",
    ... )
    >>>
    >>> # Get all FOV paths for a collection
    >>> fov_paths = airtable_db.get_collection_data_paths("RPE1_infection_v2")
    >>> print(fov_paths)
    >>> # ['/hpc/data/rpe1.zarr/B/3/0', '/hpc/data/rpe1.zarr/B/3/1', ...]
    """

    def __init__(
        self,
        base_id: str,
        api_key: str | None = None,
    ):
        api_key = api_key or os.getenv("AIRTABLE_API_KEY")
        if not api_key:
            raise ValueError("Airtable API key required (set AIRTABLE_API_KEY)")

        self.api = Api(api_key)
        self.base_id = base_id
        self.datasets_table = self.api.table(base_id, "Datasets")
        self.collections_table = self.api.table(base_id, "Collections")
        self.models_table = self.api.table(base_id, "Models")

    def register_dataset(self, dataset: DatasetRecord) -> str:
        """
        Register a single dataset record (FOV) in Airtable.

        Parameters
        ----------
        dataset : DatasetRecord
            Dataset record with FOV metadata

        Returns
        -------
        str
            Airtable record ID

        Examples
        --------
        >>> from viscy.airtable.schemas import DatasetRecord
        >>> dataset = DatasetRecord(
        ...     fov_id="plate_B_3_0",
        ...     dataset_name="plate",
        ...     well_id="B_3",
        ...     fov_name="0",
        ...     data_path="/hpc/data/plate.zarr/B/3/0"
        ... )
        >>> record_id = airtable_db.register_dataset(dataset)
        """
        record_dict = dataset.to_airtable_dict()
        created = self.datasets_table.create(record_dict)
        return created["id"]

    def register_datasets(self, datasets: list[DatasetRecord]) -> list[str]:
        """
        Register multiple dataset records (FOVs) in Airtable.

        Parameters
        ----------
        datasets : list[DatasetRecord]
            List of dataset records to register

        Returns
        -------
        list[str]
            List of Airtable record IDs

        Examples
        --------
        >>> from viscy.airtable.schemas import DatasetRecord
        >>> datasets = [
        ...     DatasetRecord(
        ...         fov_id="plate_B_3_0",
        ...         dataset_name="plate",
        ...         well_id="B_3",
        ...         fov_name="0",
        ...         data_path="/hpc/data/plate.zarr/B/3/0"
        ...     ),
        ...     DatasetRecord(
        ...         fov_id="plate_B_3_1",
        ...         dataset_name="plate",
        ...         well_id="B_3",
        ...         fov_name="1",
        ...         data_path="/hpc/data/plate.zarr/B/3/1"
        ...     ),
        ... ]
        >>> record_ids = airtable_db.register_datasets(datasets)
        """
        record_ids = []
        for dataset in datasets:
            record_dict = dataset.to_airtable_dict()
            created = self.datasets_table.create(record_dict)
            record_ids.append(created["id"])
        return record_ids

    def create_collection_from_datasets(
        self,
        collection_name: str,
        fov_ids: list[str],
        version: str,
        purpose: str = "training",
        project_name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Create a collection (collection) from a list of FOV IDs.

        Parameters
        ----------
        collection_name : str
            Name for this collection

        fov_ids : list[str]
            List of FOV_ID values from Datasets table (e.g., ["plate1_B_3_0", "plate1_B_3_1"])
        version : str
            Semantic version (e.g., "0.0.1", "0.1.0", "1.0.0")
        purpose : str
            Purpose of this collection ("training", "validation", "test")
        project_name : str | None
            Project Name (e.g OrganelleBox, DynaCLR, etc.)
        description : str | None
            Human-readable description

        Returns
        -------
        str
            Airtable collection record ID

        Examples
        --------
        >>> airtable_db.create_collection_from_datasets(
        ...     collection_name="2024_11_07_A549_SEC61_DENV_wells_B1_B2",
        ...     project_name="OrganelleBox",
        ...     fov_ids=["2024_11_07_A549_SEC61_DENV_B1_0", "2024_11_07_A549_SEC61_DENV_B1_1"],
        ...     version="0.0.1",
        ...     purpose="training",
        ...     project_name="OrganelleBox",
        ...     description="High-quality dataset records from wells B3-B4"
        ... )
        """
        # Validate semantic version format
        import re

        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise ValueError(
                f"Version must be semantic version format (e.g., '0.0.1', '1.0.0'), got: '{version}'"
            )

        # Check if collection with same name + version exists (use DataFrame)
        df_collections = self.list_collections()

        # Only check for duplicates if table is not empty and has required columns
        if (
            len(df_collections) > 0
            and "name" in df_collections.columns
            and "version" in df_collections.columns
        ):
            existing = df_collections[
                (df_collections["name"] == collection_name)
                & (df_collections["version"] == version)
            ]

            if len(existing) > 0:
                raise ValueError(
                    f"Collections '{collection_name}' version '{version}' already exists. "
                    f"To create a new version, increment the version number (e.g., '0.0.2')."
                )

            existing_versions = df_collections[
                df_collections["name"] == collection_name
            ]
            if len(existing_versions) > 0:
                versions = sorted(existing_versions["version"].tolist())
                print(
                    f"ℹ Collections '{collection_name}' existing versions: {versions}"
                )
                print(f"  Creating new version: '{version}'")

        # Get Airtable record IDs for these FOV IDs (ensure unique)
        dataset_record_ids = []
        seen_fov_ids = set()

        for fov_id in fov_ids:
            if fov_id in seen_fov_ids:
                continue  # Skip duplicates

            formula = f"{{FOV_ID}}='{fov_id}'"
            records = self.datasets_table.all(formula=formula)
            if records:
                dataset_record_ids.append(records[0]["id"])
                seen_fov_ids.add(fov_id)
            else:
                raise ValueError(f"FOV ID '{fov_id}' not found in Datasets table")

        # Remove any duplicate record IDs (shouldn't happen, but extra safety)
        dataset_record_ids = list(dict.fromkeys(dataset_record_ids))

        # Create collection record
        collection_record = {
            "name": collection_name,
            "datasets": dataset_record_ids,  # Linked records (unique)
            "version": version,  # Semantic version (required)
            "purpose": purpose,
            "created_by": getpass.getuser(),
        }
        if project_name:
            collection_record["project"] = project_name
        if description:
            collection_record["description"] = description

        created = self.collections_table.create(collection_record)
        return created["id"]

    def create_collection_from_query(
        self,
        collection_name: str,
        version: str,
        source_dataset: str | None = None,
        well_ids: list[str] | None = None,
        exclude_fov_ids: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Create a collection by filtering dataset records with pandas.

        Parameters
        ----------
        collection_name : str
            Name for this collection
        version : str
            Semantic version (e.g., "0.0.1") - REQUIRED
        source_dataset : str | None
            Filter by source dataset name (from 'Dataset' field)
        well_ids : list[str] | None
            Filter by well identifiers (e.g., ["B_3", "B_4"])
        exclude_fov_ids : list[str] | None
            FOV_ID values to exclude
        **kwargs
            Additional arguments for create_collection_from_datasets

        Returns
        -------
        str
            Airtable collection record ID

        Examples
        --------
        >>> # Create collection from specific wells in a dataset
        >>> airtable_db.create_collection_from_query(
        ...     collection_name="RPE1_infection_training",
        ...     version="0.0.1",
        ...     source_dataset="RPE1_plate1",
        ...     well_ids=["B_3", "B_4"],
        ...     exclude_fov_ids=["RPE1_plate1_B_3_2"]
        ... )
        """
        # Get all dataset records as DataFrame
        df = self.list_datasets()

        # Apply filters with pandas
        if source_dataset:
            df = df[df["Dataset"] == source_dataset]

        if well_ids:
            df = df[df["Well ID"].isin(well_ids)]

        # Exclude specified FOVs
        if exclude_fov_ids:
            df = df[~df["FOV_ID"].isin(exclude_fov_ids)]

        fov_ids = df["FOV_ID"].tolist()

        print(f"Found {len(fov_ids)} dataset records matching criteria")

        # Create collection
        return self.create_collection_from_datasets(
            collection_name=collection_name, version=version, fov_ids=fov_ids, **kwargs
        )

    def get_collection_data_paths(
        self, collection_name: str, version: str | None = None
    ) -> list[str]:
        """
        Get list of data paths for a collection.

        Parameters
        ----------
        collection_name : str
            Collections name
        version : str | None
            Specific version (if None, returns latest)

        Returns
        -------
        list[str]
            List of data paths

        Examples
        --------
        >>> paths = airtable_db.get_collection_data_paths("RPE1_infection_v2")
        >>> print(paths)
        >>> # ['/hpc/data/rpe1.zarr/B/3/0', '/hpc/data/rpe1.zarr/B/3/1', ...]
        """
        # Get all collections as DataFrame
        df_collections = self.list_collections()

        if len(df_collections) == 0 or "name" not in df_collections.columns:
            raise ValueError(
                f"Collections '{collection_name}' not found (table is empty)"
            )

        # Filter by name
        filtered = df_collections[df_collections["name"] == collection_name]

        if len(filtered) == 0:
            raise ValueError(f"Collections '{collection_name}' not found")

        # Filter by version if specified, otherwise get latest
        if version:
            if "version" not in df_collections.columns:
                raise ValueError("Version field not found in Collections table")
            filtered = filtered[filtered["version"] == version]
            if len(filtered) == 0:
                raise ValueError(
                    f"Collections '{collection_name}' version '{version}' not found"
                )
        else:
            # Get latest version (sort by created_time if column exists)
            if "created_time" in filtered.columns:
                filtered = filtered.sort_values("created_time", ascending=False)

        # Get the first (or only) matching collection
        collection_row = filtered.iloc[0]

        # Get linked dataset record IDs
        dataset_record_ids = collection_row.get("datasets", [])
        if not dataset_record_ids or len(dataset_record_ids) == 0:
            return []

        # Fetch data paths
        data_paths = []
        for dataset_id in dataset_record_ids:
            dataset_record = self.datasets_table.get(dataset_id)
            data_paths.append(dataset_record["fields"]["Data path"])

        return data_paths

    def get_collection(
        self, collection_name: str, version: str | None = None
    ) -> dict[str, Any]:
        """
        Get full collection information including data paths.

        Parameters
        ----------
        collection_name : str
            Collections name
        version : str | None
            Specific version

        Returns
        -------
        dict
            Collections info with data paths and metadata
        """
        # Get all collections as DataFrame
        df_collections = self.list_collections()

        if len(df_collections) == 0 or "name" not in df_collections.columns:
            raise ValueError(
                f"Collections '{collection_name}' not found (table is empty)"
            )

        # Filter by name
        filtered = df_collections[df_collections["name"] == collection_name]

        if len(filtered) == 0:
            raise ValueError(f"Collections '{collection_name}' not found")

        # Filter by version if specified, otherwise get latest
        if version:
            if "version" not in df_collections.columns:
                raise ValueError("Version field not found in Collections table")
            filtered = filtered[filtered["version"] == version]
            if len(filtered) == 0:
                raise ValueError(
                    f"Collections '{collection_name}' version '{version}' not found"
                )
        else:
            # Get latest version (sort by created_time if column exists)
            if "created_time" in filtered.columns:
                filtered = filtered.sort_values("created_time", ascending=False)

        # Get the first (or only) matching collection
        collection_row = filtered.iloc[0]
        collection = collection_row.to_dict()

        # Add data paths
        collection["data_paths"] = self.get_collection_data_paths(
            collection_name, version
        )

        return collection

    def list_collections(
        self, purpose: str | None = None, as_dataframe: bool = True
    ) -> pd.DataFrame | list[dict]:
        """
        List all collections.

        Parameters
        ----------
        purpose : str | None
            Filter by purpose ("training", "validation", "test")
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            Collections records as DataFrame or list of dicts

        Examples
        --------
        >>> airtable_db.list_collections(purpose="training")
        >>> # Returns DataFrame with columns: id, name, version, purpose, ...
        """
        # Fetch all collections (try sorting, but don't fail if field doesn't exist)
        try:
            records = self.collections_table.all(sort=["-created_time"])
        except Exception:
            # If sort fails (field might not exist), fetch without sorting
            records = self.collections_table.all()

        data = [{"id": r["id"], **r["fields"]} for r in records]

        # Convert to DataFrame or list
        if as_dataframe:
            df = pd.DataFrame(data)
            # Sort by created_time if column exists
            if len(df) > 0 and "created_time" in df.columns:
                df = df.sort_values("created_time", ascending=False)
            # Filter by purpose if specified
            if purpose and len(df) > 0 and "purpose" in df.columns:
                df = df[df["purpose"] == purpose]
            return df
        else:
            # Filter list if purpose specified
            if purpose:
                data = [d for d in data if d.get("purpose") == purpose]
            return data

    def list_datasets(
        self,
        as_dataframe: bool = True,
        as_pydantic: bool = False,
        skip_invalid: bool = True,
    ) -> pd.DataFrame | list[dict] | list[DatasetRecord]:
        """
        Get all dataset records (FOVs) as a DataFrame, list of dicts, or Pydantic models.

        Use pandas for filtering - much simpler and more powerful than
        building Airtable formulas.

        Parameters
        ----------
        as_dataframe : bool
            If True, return pandas DataFrame. Ignored if as_pydantic is True.
        as_pydantic : bool
            If True, return list of DatasetRecord objects. Takes precedence over as_dataframe.
        skip_invalid : bool
            If True and as_pydantic=True, skip records that fail validation instead of raising error.
            Default is True to handle legacy/incomplete records gracefully.

        Returns
        -------
        pd.DataFrame | list[dict] | list[DatasetRecord]
            All dataset records

        Examples
        --------
        >>> # Get all datasets as DataFrame
        >>> df = airtable_db.list_datasets()
        >>>
        >>> # Filter with pandas (simple and powerful!)
        >>> filtered = df[df['Dataset'] == 'RPE1_plate1']
        >>> filtered = df[df['Well ID'].isin(['B_3', 'B_4'])]
        >>> filtered = df[~df['FOV_ID'].isin(['RPE1_plate1_B_3_2'])]
        >>>
        >>> # Get as Pydantic models for type safety
        >>> datasets = airtable_db.list_datasets(as_pydantic=True)
        >>> for dataset in datasets:
        ...     print(dataset.fov_id, dataset.data_path)
        >>>
        >>> # Group and analyze
        >>> df.groupby('Dataset').size()
        >>> df.groupby('Well ID').size()
        """
        records = self.datasets_table.all()

        if as_pydantic:
            parsed_records = []
            for r in records:
                try:
                    parsed_records.append(DatasetRecord.from_airtable_record(r))
                except Exception:
                    if skip_invalid:
                        # Skip invalid records silently
                        continue
                    else:
                        raise
            return parsed_records

        data = [{"id": r["id"], **r["fields"]} for r in records]

        if as_dataframe:
            return pd.DataFrame(data)
        return data

    def delete_collection(self, collection_id: str) -> bool:
        """
        Delete a collection record from Airtable.

        Parameters
        ----------
        collection_id : str
            Airtable record ID of the collection to delete

        Returns
        -------
        bool
            True if deletion was successful

        Examples
        --------
        >>> collection_id = airtable_db.create_collection_from_datasets(...)
        >>> airtable_db.delete_collection(collection_id)
        >>> print(f"Deleted collection: {collection_id}")
        """
        self.collections_table.delete(collection_id)
        return True

    def log_model_training(
        self,
        collection_id: str,
        wandb_run_id: str,
        model_name: str | None = None,
        metrics: dict[str, float] | None = None,
        checkpoint_path: str | None = None,
        trained_by: str | None = None,
    ) -> str:
        """
        Log that a model was trained using a collection.

        Creates entry in Models table and updates Collections table.

        Parameters
        ----------
        collection_id : str
            Airtable record ID of collection used
        wandb_run_id : str
            W&B run ID for experiment tracking
        model_name : str | None
            Human-readable model name
        metrics : dict | None
            Training metrics (e.g., {"val_loss": 0.15, "dice": 0.92})
        checkpoint_path : str | None
            Path to saved model checkpoint
        trained_by : str | None
            Username of person who trained the model

        Returns
        -------
        str
            Airtable record ID of created model entry

        Examples
        --------
        >>> collection_id = airtable_db.create_collection_from_datasets(...)
        >>> model_id = airtable_db.log_model_training(
        ...     collection_id=collection_id,
        ...     wandb_run_id="20260107-152420",
        ...     model_name="contrastive-a549:v1",
        ...     metrics={"val_loss": 0.15},
        ...     trained_by="eduardo.hirata"
        ... )
        """
        # Create model record
        model_record = {
            "model_name": model_name or f"model_{datetime.now():%Y%m%d_%H%M%S}",
            "collection": [collection_id],  # Link to collection
            "wandb_run_id": wandb_run_id,
            "trained_date": datetime.now().isoformat(),
        }

        if metrics:
            model_record.update(metrics)

        if checkpoint_path:
            model_record["checkpoint_path"] = checkpoint_path

        if trained_by:
            model_record["trained_by"] = trained_by

        created = self.models_table.create(model_record)

        # Update collection record to track usage
        collection = self.collections_table.get(collection_id)
        models_trained_str = collection["fields"].get("models_trained", "")

        # Handle models_trained as comma-separated string
        if models_trained_str:
            models_list = [m.strip() for m in models_trained_str.split(",")]
            models_list.append(wandb_run_id)
            new_models_str = ", ".join(models_list)
        else:
            new_models_str = wandb_run_id

        self.collections_table.update(
            collection_id,
            {"models_trained": new_models_str, "last_used": datetime.now().isoformat()},
        )

        return created["id"]

    def get_models_for_collection(
        self, collection_id: str, as_dataframe: bool = True
    ) -> pd.DataFrame | list[dict]:
        """
        Get all models trained on a specific collection.

        Parameters
        ----------
        collection_id : str
            Airtable record ID of collection
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            Model records as DataFrame or list of dicts

        Examples
        --------
        >>> models_df = airtable_db.get_models_for_collection(collection_id)
        >>> print(models_df[["model_name", "mlflow_run_id", "trained_date"]])
        """
        # Get all models as DataFrame
        records = self.models_table.all()
        data = [{"id": r["id"], **r["fields"]} for r in records]

        if as_dataframe:
            df = pd.DataFrame(data)
            if len(df) == 0:
                return df

            # Filter by collection_id using pandas
            # The 'collection' field contains a list of linked record IDs
            df_filtered = df[
                df["collection"].apply(
                    lambda x: collection_id in x if isinstance(x, list) else False
                )
            ]

            # Sort by trained_date if column exists
            if "trained_date" in df_filtered.columns:
                df_filtered = df_filtered.sort_values("trained_date", ascending=False)

            return df_filtered
        else:
            # Filter list
            filtered = [d for d in data if collection_id in d.get("collection", [])]
            return filtered

    def list_models(self, as_dataframe: bool = True) -> pd.DataFrame | list[dict]:
        """
        List all models in the airtable_db.

        Parameters
        ----------
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            All model records

        Examples
        --------
        >>> models_df = airtable_db.list_models()
        >>> print(models_df.groupby("model_name").size())
        """
        records = self.models_table.all()
        data = [{"id": r["id"], **r["fields"]} for r in records]

        if as_dataframe:
            df = pd.DataFrame(data)
            # Sort by trained_date if column exists
            if len(df) > 0 and "trained_date" in df.columns:
                df = df.sort_values("trained_date", ascending=False)
            return df
        return data

    def get_dataset_paths(
        self,
        collection_name: str,
        version: str,
    ) -> Collections:
        """
        Get zarr store paths and FOV names for a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection
        version : str
            Semantic version of the collection

        Returns
        -------
        Collections
            Collections object containing list of CollectionDataset (one per HCS plate)

        Examples
        --------
        >>> collection = airtable_db.get_dataset_paths("my_collection", "0.0.1")
        >>> print(f"{collection.name} v{collection.version}: {collection.total_fovs} FOVs")

        >>> # Use with TripletDataModule
        >>> for ds in collection:
        ...     data_module = TripletDataModule(
        ...         data_path=ds.data_path,
        ...         tracks_path=ds.tracks_path,
        ...         include_fov_names=ds.fov_names,
        ...     )
        """
        # Get collection record IDs
        dataset_record_ids = self._get_collection_dataset_ids(collection_name, version)
        if not dataset_record_ids:
            return Collections(name=collection_name, version=version, datasets=[])

        dataset_records = [
            self.datasets_table.get(dataset_id)["fields"]
            for dataset_id in dataset_record_ids
        ]

        stores: dict[str, list[str]] = {}
        for fields in dataset_records:
            data_path = fields["Data path"]
            fov_name = f"{fields['Well ID']}/{fields['FOV']}"

            if data_path not in stores:
                stores[data_path] = []
            stores[data_path].append(fov_name)

        datasets = [
            CollectionDataset(
                data_path=data_path,
                tracks_path=self._derive_tracks_path(data_path),
                fov_names=natsorted(fov_names),
            )
            for data_path, fov_names in stores.items()
        ]

        return Collections(name=collection_name, version=version, datasets=datasets)

    @staticmethod
    def _derive_tracks_path(data_path: str) -> str:
        """
        Derive tracks path from data path.

        Pattern:
        - Data: {base}/2-assemble/{name}.zarr
        - Tracks: {base}/1-preprocess/label-free/3-track/{name}_cropped.zarr
        """
        # Replace 2-assemble with 1-preprocess/label-free/3-track
        tracks_path = data_path.replace(
            "/2-assemble/", "/1-preprocess/label-free/3-track/"
        )
        # Replace .zarr with _cropped.zarr
        if tracks_path.endswith(".zarr"):
            tracks_path = tracks_path[:-5] + "_cropped.zarr"
        return tracks_path

    def _get_collection_dataset_ids(
        self, collection_name: str, version: str
    ) -> list[str]:
        """Get linked dataset record IDs for a collection."""
        df_collections = self.list_collections()

        if len(df_collections) == 0 or "name" not in df_collections.columns:
            raise ValueError(
                f"Collections '{collection_name}' not found (table is empty)"
            )

        filtered = df_collections[df_collections["name"] == collection_name]
        if len(filtered) == 0:
            raise ValueError(f"Collections '{collection_name}' not found")

        if "version" not in df_collections.columns:
            raise ValueError("Version field not found in Collections table")

        filtered = filtered[filtered["version"] == version]
        if len(filtered) == 0:
            raise ValueError(
                f"Collections '{collection_name}' version '{version}' not found"
            )

        collection_row = filtered.iloc[0]
        dataset_record_ids = collection_row.get("datasets", [])

        if not dataset_record_ids or len(dataset_record_ids) == 0:
            return []

        return dataset_record_ids

    def update_record(
        self,
    ):
        # TODO: to update the tracks path column
        raise NotImplementedError("Not implemented yet")
