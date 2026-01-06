"""FOV-level dataset registry with Airtable."""

import getpass
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from pyairtable import Api


class AirtableFOVRegistry:
    """
    Interface to Airtable for FOV-level dataset management.

    Use this to:
    - Register individual FOVs from HCS plates
    - Create dataset "tags" (collections of FOVs)
    - Query which FOVs are in each dataset
    - Generate training configs from dataset tags

    Parameters
    ----------
    base_id : str
        Airtable base ID
    api_key : str | None
        Airtable API key. If None, reads from AIRTABLE_API_KEY env var.

    Examples
    --------
    >>> registry = AirtableFOVRegistry(base_id="appXXXXXXXXXXXXXX")
    >>>
    >>> # Create dataset from FOV selection
    >>> registry.create_dataset_from_fovs(
    ...     dataset_name="RPE1_infection_v2",
    ...     fov_ids=["FOV_001", "FOV_002", "FOV_004"],
    ...     version="v2",
    ...     purpose="training"
    ... )
    >>>
    >>> # Get all FOV paths for a dataset
    >>> fov_paths = registry.get_dataset_fov_paths("RPE1_infection_v2")
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
        self.fovs_table = self.api.table(base_id, "FOVs")
        self.datasets_table = self.api.table(base_id, "Datasets")
        self.models_table = self.api.table(base_id, "Models")

    def register_fov(
        self,
        fov_id: str,
        plate_name: str,
        well_id: str,
        row: str,
        column: str,
        fov_name: str,
        fov_path: str,
        quality: str = "Good",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register a single FOV in Airtable.

        Parameters
        ----------
        fov_id : str
            Human-readable identifier (e.g., "RPE1_plate1_B_3_0")
        plate_name : str
            Name of the plate this FOV belongs to
        well_id : str
            Well identifier as row_column (e.g., "B_3")
        row : str
            Well row (e.g., "B")
        column : str
            Well column (e.g., "3")
        fov_name : str
            FOV index within well (e.g., "0", "1", "2")
        fov_path : str
            Full path to FOV (e.g., "/hpc/data/plate.zarr/B/3/0")
        quality : str
            Quality assessment ("Good", "Poor", "Contaminated", etc.)
        metadata : dict | None
            Additional metadata (cell_count, timestamp, etc.)

        Returns
        -------
        str
            Airtable record ID
        """
        record = {
            "fov_id": fov_id,
            "plate_name": plate_name,
            "well_id": well_id,
            "row": row,
            "column": column,
            "fov_name": fov_name,
            "fov_path": fov_path,
            "quality": quality,
        }

        if metadata:
            # Store as JSON string in notes field
            record["notes"] = json.dumps(metadata)

        created = self.fovs_table.create(record)
        return created["id"]

    def create_dataset_from_fovs(
        self,
        dataset_name: str,
        fov_ids: list[str],
        version: str,
        purpose: str = "training",
        description: str | None = None,
    ) -> str:
        """
        Create a dataset (tag) from a list of FOV IDs.

        Parameters
        ----------
        dataset_name : str
            Name for this dataset collection
        fov_ids : list[str]
            List of FOV IDs to include (e.g., ["FOV_001", "FOV_002"])
        version : str
            Semantic version (e.g., "0.0.1", "0.1.0", "1.0.0")
            REQUIRED - forces conscious versioning
        purpose : str
            Purpose of this dataset ("training", "validation", "test")
        description : str | None
            Human-readable description

        Returns
        -------
        str
            Airtable dataset record ID

        Examples
        --------
        >>> registry.create_dataset_from_fovs(
        ...     dataset_name="RPE1_clean_wells",
        ...     fov_ids=["FOV_001", "FOV_002", "FOV_004"],
        ...     version="0.0.1",
        ...     description="High-quality FOVs from wells B3-B4"
        ... )
        """
        # Validate semantic version format
        import re

        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise ValueError(
                f"Version must be semantic version format (e.g., '0.0.1', '1.0.0'), got: '{version}'"
            )

        # Check if dataset with same name + version exists (use DataFrame)
        df_datasets = self.list_datasets()

        if len(df_datasets) > 0:
            existing = df_datasets[
                (df_datasets["name"] == dataset_name)
                & (df_datasets["version"] == version)
            ]

            if len(existing) > 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' version '{version}' already exists. "
                    f"To create a new version, increment the version number (e.g., '0.0.2')."
                )

            # Show existing versions (helpful feedback)
            existing_versions = df_datasets[df_datasets["name"] == dataset_name]
            if len(existing_versions) > 0:
                versions = sorted(existing_versions["version"].tolist())
                print(f"ℹ Dataset '{dataset_name}' existing versions: {versions}")
                print(f"  Creating new version: '{version}'")

        # Get Airtable record IDs for these FOV IDs (ensure unique)
        fov_record_ids = []
        seen_fov_ids = set()

        for fov_id in fov_ids:
            if fov_id in seen_fov_ids:
                continue  # Skip duplicates

            formula = f"{{fov_id}}='{fov_id}'"
            records = self.fovs_table.all(formula=formula)
            if records:
                fov_record_ids.append(records[0]["id"])
                seen_fov_ids.add(fov_id)
            else:
                raise ValueError(f"FOV '{fov_id}' not found in FOVs table")

        # Remove any duplicate record IDs (shouldn't happen, but extra safety)
        fov_record_ids = list(dict.fromkeys(fov_record_ids))

        # Create dataset record
        dataset_record = {
            "name": dataset_name,
            "fovs": fov_record_ids,  # Linked records (unique)
            "version": version,  # Semantic version (required)
            "purpose": purpose,
            "created_date": datetime.now().isoformat(),
            "created_by": getpass.getuser(),
            "num_fovs": len(fov_record_ids),
        }

        if description:
            dataset_record["description"] = description

        created = self.datasets_table.create(dataset_record)
        return created["id"]

    def create_dataset_from_query(
        self,
        dataset_name: str,
        version: str,
        plate_name: str | None = None,
        well_ids: list[str] | None = None,
        quality: str | None = None,
        exclude_fov_ids: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Create a dataset by filtering FOVs with pandas.

        Parameters
        ----------
        dataset_name : str
            Name for this dataset
        version : str
            Semantic version (e.g., "0.0.1") - REQUIRED
        plate_name : str | None
            Filter by plate name
        well_ids : list[str] | None
            Filter by well identifiers (e.g., ["B_3", "B_4"])
        quality : str | None
            Filter by quality ("Good", "Poor", etc.)
        exclude_fov_ids : list[str] | None
            FOV IDs to exclude
        **kwargs
            Additional arguments for create_dataset_from_fovs

        Returns
        -------
        str
            Airtable dataset record ID

        Examples
        --------
        >>> # Create dataset from all good-quality FOVs in specific wells
        >>> registry.create_dataset_from_query(
        ...     dataset_name="RPE1_infection_training",
        ...     version="0.0.1",
        ...     plate_name="RPE1_plate1",
        ...     well_ids=["B_3", "B_4"],
        ...     quality="Good",
        ...     exclude_fov_ids=["RPE1_plate1_B_3_2"]
        ... )
        """
        # Get all FOVs as DataFrame
        df = self.list_fovs()

        # Apply filters with pandas
        if plate_name:
            df = df[df["plate_name"] == plate_name]

        if quality:
            df = df[df["quality"] == quality]

        if well_ids:
            df = df[df["well_id"].isin(well_ids)]

        # Exclude specified FOVs
        if exclude_fov_ids:
            df = df[~df["fov_id"].isin(exclude_fov_ids)]

        fov_ids = df["fov_id"].tolist()

        print(f"Found {len(fov_ids)} FOVs matching criteria")

        # Create dataset
        return self.create_dataset_from_fovs(
            dataset_name=dataset_name, version=version, fov_ids=fov_ids, **kwargs
        )

    def get_dataset_fov_paths(
        self, dataset_name: str, version: str | None = None
    ) -> list[str]:
        """
        Get list of FOV paths for a dataset.

        Parameters
        ----------
        dataset_name : str
            Dataset name
        version : str | None
            Specific version (if None, returns latest)

        Returns
        -------
        list[str]
            List of FOV paths

        Examples
        --------
        >>> paths = registry.get_dataset_fov_paths("RPE1_infection_v2")
        >>> print(paths)
        >>> # ['/hpc/data/rpe1.zarr/B/3/0', '/hpc/data/rpe1.zarr/B/3/1', ...]
        """
        # Get all datasets as DataFrame
        df_datasets = self.list_datasets()

        if len(df_datasets) == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Filter by name
        filtered = df_datasets[df_datasets["name"] == dataset_name]

        if len(filtered) == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Filter by version if specified, otherwise get latest
        if version:
            filtered = filtered[filtered["version"] == version]
            if len(filtered) == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' version '{version}' not found"
                )
        else:
            # Get latest version (sort by created_date)
            filtered = filtered.sort_values("created_date", ascending=False)

        # Get the first (or only) matching dataset
        dataset_row = filtered.iloc[0]

        # Get linked FOV record IDs
        fov_record_ids = dataset_row.get("fovs", [])
        if not fov_record_ids or len(fov_record_ids) == 0:
            return []

        # Fetch FOV paths
        fov_paths = []
        for fov_id in fov_record_ids:
            fov_record = self.fovs_table.get(fov_id)
            fov_paths.append(fov_record["fields"]["fov_path"])

        return fov_paths

    def get_dataset(
        self, dataset_name: str, version: str | None = None
    ) -> dict[str, Any]:
        """
        Get full dataset information including FOV details.

        Parameters
        ----------
        dataset_name : str
            Dataset name
        version : str | None
            Specific version

        Returns
        -------
        dict
            Dataset info with FOV paths and metadata
        """
        # Get all datasets as DataFrame
        df_datasets = self.list_datasets()

        if len(df_datasets) == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Filter by name
        filtered = df_datasets[df_datasets["name"] == dataset_name]

        if len(filtered) == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Filter by version if specified, otherwise get latest
        if version:
            filtered = filtered[filtered["version"] == version]
            if len(filtered) == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' version '{version}' not found"
                )
        else:
            # Get latest version (sort by created_date)
            filtered = filtered.sort_values("created_date", ascending=False)

        # Get the first (or only) matching dataset
        dataset_row = filtered.iloc[0]
        dataset = dataset_row.to_dict()

        # Add FOV paths
        dataset["fov_paths"] = self.get_dataset_fov_paths(dataset_name, version)

        return dataset

    def list_datasets(
        self, purpose: str | None = None, as_dataframe: bool = True
    ) -> pd.DataFrame | list[dict]:
        """
        List all datasets.

        Parameters
        ----------
        purpose : str | None
            Filter by purpose ("training", "validation", "test")
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            Dataset records as DataFrame or list of dicts

        Examples
        --------
        >>> registry.list_datasets(purpose="training")
        >>> # Returns DataFrame with columns: id, name, version, purpose, ...
        """
        # Fetch all datasets (sorted by most recent first)
        records = self.datasets_table.all(sort=["-created_date"])
        data = [{"id": r["id"], **r["fields"]} for r in records]

        # Convert to DataFrame or list
        if as_dataframe:
            df = pd.DataFrame(data)
            # Filter by purpose if specified
            if purpose and len(df) > 0:
                df = df[df["purpose"] == purpose]
            return df
        else:
            # Filter list if purpose specified
            if purpose:
                data = [d for d in data if d.get("purpose") == purpose]
            return data

    def list_fovs(self, as_dataframe: bool = True) -> pd.DataFrame | list[dict]:
        """
        Get all FOVs as a DataFrame (or list of dicts).

        Use pandas for filtering - much simpler and more powerful than
        building Airtable formulas.

        Parameters
        ----------
        as_dataframe : bool
            If True, return pandas DataFrame. If False, return list of dicts.

        Returns
        -------
        pd.DataFrame | list[dict]
            All FOV records

        Examples
        --------
        >>> # Get all FOVs
        >>> df = registry.list_fovs()
        >>>
        >>> # Filter with pandas (simple and powerful!)
        >>> filtered = df[df['plate_name'] == 'RPE1_plate1']
        >>> filtered = df[df['quality'] == 'Good']
        >>> filtered = df[df['row'] == 'B']
        >>> filtered = df[df['row'].isin(['B', 'C'])]
        >>> filtered = df[(df['row'] == 'B') & (df['column'] == '3')]
        >>>
        >>> # Exclude FOVs
        >>> filtered = df[~df['fov_id'].isin(['RPE1_plate1_B_3_2'])]
        >>>
        >>> # Group and analyze
        >>> df.groupby('plate_name').size()
        >>> df.groupby(['row', 'column']).size()
        """
        records = self.fovs_table.all()
        data = [{"id": r["id"], **r["fields"]} for r in records]

        if as_dataframe:
            return pd.DataFrame(data)
        return data
