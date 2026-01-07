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


@dataclass
class ManifestDataset:
    """
    Dataset paths for one HCS plate/zarr store.

    A manifest may contain multiple stores, each returned as a separate ManifestDataset.
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
class Manifest:
    """All datasets for a manifest, potentially across multiple HCS plates."""

    name: str
    version: str
    datasets: list[ManifestDataset]

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
    Unified interface to Airtable for dataset, manifest, and model management.

    Use this to:
    - Register individual FOVs from HCS plates
    - Create and manage dataset manifests (collections of FOVs)
    - Track model training on manifests
    - Query datasets, manifests, and models

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
    >>> # Create manifest from FOV selection
    >>> manifest_id = airtable_db.create_manifest_from_datasets(
    ...     manifest_name="RPE1_infection_v2",
    ...     fov_ids=["FOV_001", "FOV_002", "FOV_004"],
    ...     version="0.0.1",
    ...     purpose="training"
    ... )
    >>>
    >>> # Track model training
    >>> airtable_db.log_model_training(
    ...     manifest_id=manifest_id,
    ...     mlflow_run_id="run_123",
    ...     model_name="my_model",
    ... )
    >>>
    >>> # Get all FOV paths for a manifest
    >>> fov_paths = airtable_db.get_manifest_data_paths("RPE1_infection_v2")
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
        self.manifests_table = self.api.table(base_id, "Manifest")
        self.models_table = self.api.table(base_id, "Models")

    def register_dataset(
        self,
        fov_id: str,
        dataset_name: str,
        well_id: str,
        fov_name: str,
        data_path: str,
    ) -> str:
        """
        Register a single dataset record (FOV) in Airtable.

        Parameters
        ----------
        fov_id : str
            Human-readable identifier (e.g., "RPE1_plate1_B_3_0")
        dataset_name : str
            Name of the dataset/plate this FOV belongs to
        well_id : str
            Well identifier as row_column (e.g., "B_3")
        fov_name : str
            FOV index within well (e.g., "0", "1", "2")
        data_path : str
            Full path to FOV (e.g., "/hpc/data/plate.zarr/B/3/0")

        Returns
        -------
        str
            Airtable record ID
        """
        record = {
            "FOV_ID": fov_id,
            "Dataset": dataset_name,
            "Well ID": well_id,
            "FOV": fov_name,
            "Data path": data_path,
        }

        created = self.datasets_table.create(record)
        return created["id"]

    def create_manifest_from_datasets(
        self,
        manifest_name: str,
        fov_ids: list[str],
        version: str,
        purpose: str = "training",
        project_name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Create a manifest (collection) from a list of FOV IDs.

        Parameters
        ----------
        manifest_name : str
            Name for this manifest

        fov_ids : list[str]
            List of FOV_ID values from Datasets table (e.g., ["plate1_B_3_0", "plate1_B_3_1"])
        version : str
            Semantic version (e.g., "0.0.1", "0.1.0", "1.0.0")
        purpose : str
            Purpose of this manifest ("training", "validation", "test")
        project_name : str | None
            Project Name (e.g OrganelleBox, DynaCLR, etc.)
        description : str | None
            Human-readable description

        Returns
        -------
        str
            Airtable manifest record ID

        Examples
        --------
        >>> airtable_db.create_manifest_from_datasets(
        ...     manifest_name="2024_11_07_A549_SEC61_DENV_wells_B1_B2",
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

        # Check if manifest with same name + version exists (use DataFrame)
        df_manifests = self.list_manifests()

        # Only check for duplicates if table is not empty and has required columns
        if (
            len(df_manifests) > 0
            and "name" in df_manifests.columns
            and "version" in df_manifests.columns
        ):
            existing = df_manifests[
                (df_manifests["name"] == manifest_name)
                & (df_manifests["version"] == version)
            ]

            if len(existing) > 0:
                raise ValueError(
                    f"Manifest '{manifest_name}' version '{version}' already exists. "
                    f"To create a new version, increment the version number (e.g., '0.0.2')."
                )

            existing_versions = df_manifests[df_manifests["name"] == manifest_name]
            if len(existing_versions) > 0:
                versions = sorted(existing_versions["version"].tolist())
                print(f"ℹ Manifest '{manifest_name}' existing versions: {versions}")
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

        # Create manifest record
        manifest_record = {
            "name": manifest_name,
            "datasets": dataset_record_ids,  # Linked records (unique)
            "version": version,  # Semantic version (required)
            "purpose": purpose,
            "created_by": getpass.getuser(),
        }
        if project_name:
            manifest_record["project"] = project_name
        if description:
            manifest_record["description"] = description

        created = self.manifests_table.create(manifest_record)
        return created["id"]

    def create_manifest_from_query(
        self,
        manifest_name: str,
        version: str,
        source_dataset: str | None = None,
        well_ids: list[str] | None = None,
        exclude_fov_ids: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Create a manifest by filtering dataset records with pandas.

        Parameters
        ----------
        manifest_name : str
            Name for this manifest
        version : str
            Semantic version (e.g., "0.0.1") - REQUIRED
        source_dataset : str | None
            Filter by source dataset name (from 'Dataset' field)
        well_ids : list[str] | None
            Filter by well identifiers (e.g., ["B_3", "B_4"])
        exclude_fov_ids : list[str] | None
            FOV_ID values to exclude
        **kwargs
            Additional arguments for create_manifest_from_datasets

        Returns
        -------
        str
            Airtable manifest record ID

        Examples
        --------
        >>> # Create manifest from specific wells in a dataset
        >>> airtable_db.create_manifest_from_query(
        ...     manifest_name="RPE1_infection_training",
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

        # Create manifest
        return self.create_manifest_from_datasets(
            manifest_name=manifest_name, version=version, fov_ids=fov_ids, **kwargs
        )

    def get_manifest_data_paths(
        self, manifest_name: str, version: str | None = None
    ) -> list[str]:
        """
        Get list of data paths for a manifest.

        Parameters
        ----------
        manifest_name : str
            Manifest name
        version : str | None
            Specific version (if None, returns latest)

        Returns
        -------
        list[str]
            List of data paths

        Examples
        --------
        >>> paths = airtable_db.get_manifest_data_paths("RPE1_infection_v2")
        >>> print(paths)
        >>> # ['/hpc/data/rpe1.zarr/B/3/0', '/hpc/data/rpe1.zarr/B/3/1', ...]
        """
        # Get all manifests as DataFrame
        df_manifests = self.list_manifests()

        if len(df_manifests) == 0 or "name" not in df_manifests.columns:
            raise ValueError(f"Manifest '{manifest_name}' not found (table is empty)")

        # Filter by name
        filtered = df_manifests[df_manifests["name"] == manifest_name]

        if len(filtered) == 0:
            raise ValueError(f"Manifest '{manifest_name}' not found")

        # Filter by version if specified, otherwise get latest
        if version:
            if "version" not in df_manifests.columns:
                raise ValueError("Version field not found in Manifest table")
            filtered = filtered[filtered["version"] == version]
            if len(filtered) == 0:
                raise ValueError(
                    f"Manifest '{manifest_name}' version '{version}' not found"
                )
        else:
            # Get latest version (sort by created_time if column exists)
            if "created_time" in filtered.columns:
                filtered = filtered.sort_values("created_time", ascending=False)

        # Get the first (or only) matching manifest
        manifest_row = filtered.iloc[0]

        # Get linked dataset record IDs
        dataset_record_ids = manifest_row.get("datasets", [])
        if not dataset_record_ids or len(dataset_record_ids) == 0:
            return []

        # Fetch data paths
        data_paths = []
        for dataset_id in dataset_record_ids:
            dataset_record = self.datasets_table.get(dataset_id)
            data_paths.append(dataset_record["fields"]["Data path"])

        return data_paths

    def get_manifest(
        self, manifest_name: str, version: str | None = None
    ) -> dict[str, Any]:
        """
        Get full manifest information including data paths.

        Parameters
        ----------
        manifest_name : str
            Manifest name
        version : str | None
            Specific version

        Returns
        -------
        dict
            Manifest info with data paths and metadata
        """
        # Get all manifests as DataFrame
        df_manifests = self.list_manifests()

        if len(df_manifests) == 0 or "name" not in df_manifests.columns:
            raise ValueError(f"Manifest '{manifest_name}' not found (table is empty)")

        # Filter by name
        filtered = df_manifests[df_manifests["name"] == manifest_name]

        if len(filtered) == 0:
            raise ValueError(f"Manifest '{manifest_name}' not found")

        # Filter by version if specified, otherwise get latest
        if version:
            if "version" not in df_manifests.columns:
                raise ValueError("Version field not found in Manifest table")
            filtered = filtered[filtered["version"] == version]
            if len(filtered) == 0:
                raise ValueError(
                    f"Manifest '{manifest_name}' version '{version}' not found"
                )
        else:
            # Get latest version (sort by created_time if column exists)
            if "created_time" in filtered.columns:
                filtered = filtered.sort_values("created_time", ascending=False)

        # Get the first (or only) matching manifest
        manifest_row = filtered.iloc[0]
        manifest = manifest_row.to_dict()

        # Add data paths
        manifest["data_paths"] = self.get_manifest_data_paths(manifest_name, version)

        return manifest

    def list_manifests(
        self, purpose: str | None = None, as_dataframe: bool = True
    ) -> pd.DataFrame | list[dict]:
        """
        List all manifests.

        Parameters
        ----------
        purpose : str | None
            Filter by purpose ("training", "validation", "test")
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            Manifest records as DataFrame or list of dicts

        Examples
        --------
        >>> airtable_db.list_manifests(purpose="training")
        >>> # Returns DataFrame with columns: id, name, version, purpose, ...
        """
        # Fetch all manifests (try sorting, but don't fail if field doesn't exist)
        try:
            records = self.manifests_table.all(sort=["-created_time"])
        except Exception:
            # If sort fails (field might not exist), fetch without sorting
            records = self.manifests_table.all()

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

    def list_datasets(self, as_dataframe: bool = True) -> pd.DataFrame | list[dict]:
        """
        Get all dataset records (FOVs) as a DataFrame (or list of dicts).

        Use pandas for filtering - much simpler and more powerful than
        building Airtable formulas.

        Parameters
        ----------
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            All dataset records

        Examples
        --------
        >>> # Get all datasets
        >>> df = airtable_db.list_datasets()
        >>>
        >>> # Filter with pandas (simple and powerful!)
        >>> filtered = df[df['Dataset'] == 'RPE1_plate1']
        >>> filtered = df[df['Well ID'].isin(['B_3', 'B_4'])]
        >>> filtered = df[~df['FOV_ID'].isin(['RPE1_plate1_B_3_2'])]
        >>>
        >>> # Group and analyze
        >>> df.groupby('Dataset').size()
        >>> df.groupby('Well ID').size()
        """
        records = self.datasets_table.all()
        data = [{"id": r["id"], **r["fields"]} for r in records]

        if as_dataframe:
            return pd.DataFrame(data)
        return data

    def delete_manifest(self, manifest_id: str) -> bool:
        """
        Delete a manifest record from Airtable.

        Parameters
        ----------
        manifest_id : str
            Airtable record ID of the manifest to delete

        Returns
        -------
        bool
            True if deletion was successful

        Examples
        --------
        >>> manifest_id = airtable_db.create_manifest_from_datasets(...)
        >>> airtable_db.delete_manifest(manifest_id)
        >>> print(f"Deleted manifest: {manifest_id}")
        """
        self.manifests_table.delete(manifest_id)
        return True

    def log_model_training(
        self,
        manifest_id: str,
        mlflow_run_id: str,
        model_name: str | None = None,
        metrics: dict[str, float] | None = None,
        checkpoint_path: str | None = None,
        trained_by: str | None = None,
    ) -> str:
        """
        Log that a model was trained using a manifest.

        Creates entry in Models table and updates Manifest table.

        Parameters
        ----------
        manifest_id : str
            Airtable record ID of manifest used
        mlflow_run_id : str
            MLflow run ID for experiment tracking
        model_name : str | None
            Human-readable model name
        metrics : dict | None
            Training metrics (e.g., {"accuracy": 0.89, "f1_score": 0.92})
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
        >>> manifest_id = airtable_db.create_manifest_from_datasets(...)
        >>> model_id = airtable_db.log_model_training(
        ...     manifest_id=manifest_id,
        ...     mlflow_run_id="run_abc123",
        ...     model_name="sec61_model_v1",
        ...     metrics={"val_loss": 0.15},
        ...     trained_by="researcher_name"
        ... )
        """
        # Create model record
        model_record = {
            "model_name": model_name or f"model_{datetime.now():%Y%m%d_%H%M%S}",
            "manifest": [manifest_id],  # Link to manifest
            "mlflow_run_id": mlflow_run_id,
            "trained_date": datetime.now().isoformat(),
        }

        if metrics:
            model_record.update(metrics)

        if checkpoint_path:
            model_record["checkpoint_path"] = checkpoint_path

        if trained_by:
            model_record["trained_by"] = trained_by

        created = self.models_table.create(model_record)

        # Update manifest record to track usage
        manifest = self.manifests_table.get(manifest_id)
        models_trained_str = manifest["fields"].get("models_trained", "")

        # Handle models_trained as comma-separated string
        if models_trained_str:
            models_list = [m.strip() for m in models_trained_str.split(",")]
            models_list.append(mlflow_run_id)
            new_models_str = ", ".join(models_list)
        else:
            new_models_str = mlflow_run_id

        self.manifests_table.update(
            manifest_id,
            {"models_trained": new_models_str, "last_used": datetime.now().isoformat()},
        )

        return created["id"]

    def get_models_for_manifest(
        self, manifest_id: str, as_dataframe: bool = True
    ) -> pd.DataFrame | list[dict]:
        """
        Get all models trained on a specific manifest.

        Parameters
        ----------
        manifest_id : str
            Airtable record ID of manifest
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            Model records as DataFrame or list of dicts

        Examples
        --------
        >>> models_df = airtable_db.get_models_for_manifest(manifest_id)
        >>> print(models_df[["model_name", "mlflow_run_id", "trained_date"]])
        """
        # Get all models as DataFrame
        records = self.models_table.all()
        data = [{"id": r["id"], **r["fields"]} for r in records]

        if as_dataframe:
            df = pd.DataFrame(data)
            if len(df) == 0:
                return df

            # Filter by manifest_id using pandas
            # The 'manifest' field contains a list of linked record IDs
            df_filtered = df[
                df["manifest"].apply(
                    lambda x: manifest_id in x if isinstance(x, list) else False
                )
            ]

            # Sort by trained_date if column exists
            if "trained_date" in df_filtered.columns:
                df_filtered = df_filtered.sort_values("trained_date", ascending=False)

            return df_filtered
        else:
            # Filter list
            filtered = [d for d in data if manifest_id in d.get("manifest", [])]
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
        manifest_name: str,
        version: str,
    ) -> Manifest:
        """
        Get zarr store paths and FOV names for a manifest.

        Parameters
        ----------
        manifest_name : str
            Name of the manifest
        version : str
            Semantic version of the manifest

        Returns
        -------
        Manifest
            Manifest object containing list of ManifestDataset (one per HCS plate)

        Examples
        --------
        >>> manifest = airtable_db.get_dataset_paths("my_manifest", "0.0.1")
        >>> print(f"{manifest.name} v{manifest.version}: {manifest.total_fovs} FOVs")

        >>> # Use with TripletDataModule
        >>> for ds in manifest:
        ...     data_module = TripletDataModule(
        ...         data_path=ds.data_path,
        ...         tracks_path=ds.tracks_path,
        ...         include_fov_names=ds.fov_names,
        ...     )
        """
        # Get manifest record IDs
        dataset_record_ids = self._get_manifest_dataset_ids(manifest_name, version)
        if not dataset_record_ids:
            return Manifest(name=manifest_name, version=version, datasets=[])

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
            ManifestDataset(
                data_path=data_path,
                tracks_path=self._derive_tracks_path(data_path),
                fov_names=natsorted(fov_names),
            )
            for data_path, fov_names in stores.items()
        ]

        return Manifest(name=manifest_name, version=version, datasets=datasets)

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

    def _get_manifest_dataset_ids(self, manifest_name: str, version: str) -> list[str]:
        """Get linked dataset record IDs for a manifest."""
        df_manifests = self.list_manifests()

        if len(df_manifests) == 0 or "name" not in df_manifests.columns:
            raise ValueError(f"Manifest '{manifest_name}' not found (table is empty)")

        filtered = df_manifests[df_manifests["name"] == manifest_name]
        if len(filtered) == 0:
            raise ValueError(f"Manifest '{manifest_name}' not found")

        if "version" not in df_manifests.columns:
            raise ValueError("Version field not found in Manifest table")

        filtered = filtered[filtered["version"] == version]
        if len(filtered) == 0:
            raise ValueError(
                f"Manifest '{manifest_name}' version '{version}' not found"
            )

        manifest_row = filtered.iloc[0]
        dataset_record_ids = manifest_row.get("datasets", [])

        if not dataset_record_ids or len(dataset_record_ids) == 0:
            return []

        return dataset_record_ids

    def update_record(
        self,
    ): ...  # inputs are the table, the record, maybe manifest, use the unique keys instead of the first column

    # TODO: to update the tracks path column
    raise NotImplementedError("Not implemented yet")
