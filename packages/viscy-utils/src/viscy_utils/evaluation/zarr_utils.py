"""Utilities for selectively updating AnnData zarr stores."""

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import zarr
from anndata.io import write_elem
from pandas.arrays import ArrowStringArray


def append_to_anndata_zarr(
    zarr_path: str | Path,
    *,
    obsm: dict[str, Any] | None = None,
    obs: pd.DataFrame | None = None,
    uns: dict | None = None,
) -> None:
    """Selectively write obs, obsm, or uns into an existing AnnData zarr store.

    Unlike ``adata.write_zarr()``, this only updates the specified slots
    without overwriting unrelated data (X, var, layers, etc.).

    Parameters
    ----------
    zarr_path : str | Path
        Path to an existing AnnData zarr store.
    obsm : dict[str, Any], optional
        Mapping of obsm keys to arrays. Each key is written to ``obsm/{key}``,
        replacing any existing entry.
    obs : pd.DataFrame, optional
        Observation metadata. Replaces the entire ``obs`` group.
    uns : dict, optional
        Mapping of uns keys to values. Each key is written to ``uns/{key}``,
        replacing any existing entry while preserving other uns keys.
    """
    store = zarr.open(str(zarr_path), mode="a", use_consolidated=False)
    ad.settings.allow_write_nullable_strings = True

    if obs is not None:
        # anndata's zarr writer cannot serialize pandas ArrowStringArray;
        # convert Arrow-backed string columns and index to plain object dtype.
        obs = obs.copy()
        for col in obs.columns:
            arr = obs[col].array
            if isinstance(arr, ArrowStringArray):
                obs[col] = obs[col].astype(object)
            elif isinstance(arr, pd.Categorical) and isinstance(arr.categories._values, ArrowStringArray):
                obs[col] = obs[col].cat.rename_categories(arr.categories.astype(object))
        if isinstance(obs.index._values, ArrowStringArray):
            obs.index = obs.index.astype(object)
        if "obs" in store:
            del store["obs"]
        write_elem(store, "obs", obs)

    if obsm is not None:
        for key, value in obsm.items():
            obsm_path = f"obsm/{key}"
            if obsm_path in store:
                del store[obsm_path]
            write_elem(store, obsm_path, value)

    if uns is not None:
        if "uns" not in store:
            store.create_group("uns")
        for key, value in uns.items():
            uns_path = f"uns/{key}"
            if uns_path in store:
                del store[uns_path]
            write_elem(store, uns_path, value)

    zarr.consolidate_metadata(str(zarr_path))


def merge_csv_into_obs(
    adata: ad.AnnData,
    csv_path: str | Path,
    merge_key: str | list[str] = "id",
    columns: list[str] | None = None,
    prefix: str = "",
) -> tuple[ad.AnnData, dict[str, int]]:
    """Merge columns from a CSV into the obs of an AnnData object.

    Only the required columns are read from the CSV, and rows are filtered
    to IDs present in obs before merging to minimize memory usage.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to merge into.
    csv_path : str | Path
        Path to a CSV file with column(s) matching ``merge_key``.
    merge_key : str or list[str]
        Column name(s) present in both ``adata.obs`` and the CSV to join on.
    columns : list[str], optional
        CSV columns to merge. If ``None``, all columns not already in obs
        (excluding the merge keys) are used.
    prefix : str
        Prefix to prepend to each new column name
        (e.g. ``"annotated_"``, ``"feature_"``).

    Returns
    -------
    adata : ad.AnnData
        The input AnnData with new columns added to ``.obs``.
    match_counts : dict[str, int]
        Mapping of each new column name to the number of matched (non-null) rows.

    Raises
    ------
    KeyError
        If ``merge_key`` is missing from obs or CSV, or if requested columns
        are not found in the CSV.
    ValueError
        If no new columns are found to merge.
    """
    keys = [merge_key] if isinstance(merge_key, str) else list(merge_key)

    # Determine columns to read before loading the full CSV
    if columns is not None:
        usecols = keys + list(columns)
    else:
        usecols = None

    csv_df = pd.read_csv(csv_path, usecols=usecols)

    for k in keys:
        if k not in csv_df.columns:
            raise KeyError(f"Merge key '{k}' not found in CSV columns: {list(csv_df.columns)}")
        if k not in adata.obs.columns:
            raise KeyError(f"Merge key '{k}' not found in obs columns: {list(adata.obs.columns)}")

    if columns is not None:
        missing = [c for c in columns if c not in csv_df.columns]
        if missing:
            raise KeyError(f"Columns not found in CSV: {missing}")
        append_columns = list(columns)
    else:
        existing = set(adata.obs.columns) | set(keys)
        append_columns = [c for c in csv_df.columns if c not in existing]

    if not append_columns:
        raise ValueError("No new columns to merge.")

    # Filter CSV to only rows with keys present in obs to save memory
    subset = csv_df[keys + append_columns].drop_duplicates(subset=keys)
    if len(keys) == 1:
        obs_keys = set(adata.obs[keys[0]])
        subset = subset[subset[keys[0]].isin(obs_keys)]
    else:
        obs_tuples = set(adata.obs[keys].itertuples(index=False, name=None))
        subset = subset[subset[keys].apply(tuple, axis=1).isin(obs_tuples)]

    merged = adata.obs.merge(subset, on=keys, how="left")

    match_counts = {}
    for col in append_columns:
        dest = f"{prefix}{col}"
        adata.obs[dest] = merged[col].values
        match_counts[dest] = int(merged[col].notna().sum())

    return adata, match_counts
