"""Utilities for loading and annotating DynaCLR embedding datasets (AnnData format)."""

from pathlib import Path

import pandas as pd
from anndata import read_zarr


def convert_to_dataframe(embeddings_dataset) -> pd.DataFrame:
    """
    Convert an AnnData embedding dataset to a flat DataFrame.

    The resulting DataFrame contains all ``obs`` columns plus one column per
    embedding dimension named ``feature_1 … feature_N``.

    Parameters
    ----------
    embeddings_dataset : anndata.AnnData
        Loaded from an embedding zarr store (e.g. via ``anndata.read_zarr``).

    Returns
    -------
    pd.DataFrame
    """
    obs_df = embeddings_dataset.obs.copy()
    feat_df = pd.DataFrame(
        embeddings_dataset.X,
        columns=[f"feature_{i + 1}" for i in range(embeddings_dataset.X.shape[1])],
        index=obs_df.index,
    )
    return pd.concat([obs_df, feat_df], axis=1)


def load_embeddings(embeddings_path: Path) -> pd.DataFrame:
    """
    Load an AnnData zarr store and return a flat feature DataFrame.

    Adds a ``well`` column derived from the first two parts of ``fov_name``.
    """
    adata = read_zarr(embeddings_path)
    df = convert_to_dataframe(adata)
    parts = df["fov_name"].str.split("/")
    df["well"] = parts.str[0].fillna("") + "/" + parts.str[1].fillna("")
    return df


def match_embeddings_to_annotations(
    embeddings_path: Path,
    annotations_path: Path,
    wells: list[str],
) -> pd.DataFrame:
    """
    Match embedding rows to manual annotation labels by (fov_name, track_id, t).

    Rows without a matching annotation are dropped.

    Parameters
    ----------
    embeddings_path : Path
        Path to the AnnData zarr store.
    annotations_path : Path
        Path to a CSV with columns: fov_name, track_id, t, infection_status.
    wells : list of str
        Only include rows whose ``well`` field is in this list.

    Returns
    -------
    pd.DataFrame  with an additional ``infection_status`` column.
    """
    adata = read_zarr(embeddings_path)
    annotations = pd.read_csv(annotations_path)

    df = convert_to_dataframe(adata)
    parts = df["fov_name"].str.split("/")
    df["well"] = parts.str[0].fillna("") + "/" + parts.str[1].fillna("")

    ann_parts = annotations["fov_name"].str.split("/")
    annotations["well"] = ann_parts.str[0].fillna("") + "/" + ann_parts.str[1].fillna("")

    df = df[df["well"].isin(wells)].reset_index(drop=True)
    annotations = annotations[annotations["well"].isin(wells)].copy()

    # Ensure compatible numeric types for merge keys
    for col in ["track_id", "t"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        annotations[col] = pd.to_numeric(annotations[col], errors="coerce")

    df["infection_status"] = None
    for idx in range(len(df)):
        row = df.iloc[idx]
        match = annotations[
            (annotations["track_id"] == row["track_id"])
            & (annotations["t"] == row["t"])
            & (annotations["fov_name"] == row["fov_name"])
        ]
        status = match.iloc[0]["infection_status"] if not match.empty else "unknown"
        df.at[df.index[idx], "infection_status"] = str(status)

    return df[df["infection_status"] != "unknown"].reset_index(drop=True)
