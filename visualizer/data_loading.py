"""
Data loading, validation, and PHATE computation for track visualization.

This module handles loading AnnData objects, applying FOV filters, loading
annotations, and computing PHATE embeddings for single or multiple datasets.

Functions
---------
load_and_prepare_data : Load single dataset with annotations and FOV filtering
validate_feature_compatibility : Check feature dimensions across datasets
compute_joint_phate : Compute PHATE embeddings using viscy module
load_multiple_datasets : Orchestrate multi-dataset loading and concatenation
"""

import logging
from pathlib import Path

import anndata as ad
import pandas as pd
from anndata import read_zarr

from viscy.representation.evaluation.annotation import load_annotation_anndata
from viscy.representation.evaluation.dimensionality_reduction import compute_phate

from .config import MultiDatasetConfig

logger = logging.getLogger(__name__)


def load_and_prepare_data(
    adata_path: Path,
    annotation_csv: Path | None = None,
    annotation_column: str | None = None,
    categories: dict | None = None,
    fov_filter: list[str] | None = None,
    dataset_id: str | None = None,
) -> tuple[ad.AnnData, pd.DataFrame, list, bool]:
    """
    Load AnnData with PHATE embeddings and optional annotations.

    Parameters
    ----------
    adata_path : Path
        Path to AnnData zarr store with PHATE embeddings.
    annotation_csv : Path, optional
        Path to CSV file with annotations. If None, no annotations are loaded.
    annotation_column : str, optional
        Column name in CSV for annotation values.
    categories : dict, optional
        Dictionary to remap annotation categories (e.g., {0: "uninfected", 1: "infected"}).
    fov_filter : list[str], optional
        List of FOV names or patterns to filter. If provided, only FOVs containing
        any of these strings will be included.
    dataset_id : str, optional
        Unique identifier for this dataset. If None, uses the zarr filename from adata_path.

    Returns
    -------
    adata : ad.AnnData
        AnnData object with optional annotations and dataset_id in obs.
    plot_df : pd.DataFrame
        DataFrame with PHATE coordinates and metadata for plotting, including dataset_id.
    track_options : list
        List of unique track identifiers for dropdown (format: "dataset_id/fov_name/track_id").
    has_annotations : bool
        Whether annotations were loaded.
    """
    if dataset_id is None:
        dataset_id = adata_path.name.replace(".zarr", "")
        logger.info(f"Using dataset_id: {dataset_id}")

    logger.info(f"Loading AnnData from {adata_path}")
    adata = read_zarr(adata_path)
    logger.info(f"Loaded {adata.shape[0]} observations with {adata.shape[1]} features")

    if fov_filter is not None and len(fov_filter) > 0:
        logger.info(f"Filtering FOVs by patterns: {fov_filter}")

        unique_fovs = adata.obs["fov_name"].unique()
        logger.info(f"Available FOVs (showing first 10): {list(unique_fovs[:10])}")
        logger.info(f"Total unique FOVs: {len(unique_fovs)}")

        initial_count = adata.shape[0]

        fov_names = adata.obs["fov_name"].astype(str)

        mask = pd.Series([False] * len(fov_names), index=fov_names.index)
        for pattern in fov_filter:
            mask |= fov_names.str.contains(pattern, regex=False)

        adata = adata[mask]
        logger.info(
            f"Filtered to {adata.shape[0]} observations from {len(adata.obs['fov_name'].unique())} FOVs "
            f"(removed {initial_count - adata.shape[0]} observations)"
        )

    has_annotations = False

    if annotation_csv is not None and annotation_csv.exists() and annotation_column:
        logger.info(f"Loading annotations from {annotation_csv}")
        adata = load_annotation_anndata(
            adata, str(annotation_csv), annotation_column, categories=categories
        )
        has_annotations = True

        initial_count = adata.shape[0]
        valid_mask = (adata.obs[annotation_column] != "unknown") & (
            adata.obs[annotation_column].notna()
        )
        adata = adata[valid_mask]
        logger.info(
            f"Filtered {initial_count - adata.shape[0]} invalid observations (unknown/NaN), {adata.shape[0]} remaining"
        )
    else:
        logger.info("No annotations provided, skipping annotation loading")

    if "X_phate" not in adata.obsm:
        raise ValueError("PHATE embeddings not found in AnnData.obsm['X_phate']")

    phate_coords = adata.obsm["X_phate"]
    logger.info(f"PHATE embedding shape: {phate_coords.shape}")

    adata.obs["dataset_id"] = dataset_id

    plot_df_dict = {
        "PHATE1": phate_coords[:, 0],
        "PHATE2": phate_coords[:, 1],
        "track_id": adata.obs["track_id"].values,
        "fov_name": adata.obs["fov_name"].values,
        "t": adata.obs["t"].values,
        "y": adata.obs["y"].values,
        "x": adata.obs["x"].values,
        "id": adata.obs["id"].values,
        "dataset_id": adata.obs["dataset_id"].values,
    }

    if has_annotations and annotation_column in adata.obs:
        plot_df_dict["annotation"] = adata.obs[annotation_column].values

    plot_df = pd.DataFrame(plot_df_dict, index=adata.obs.index)

    plot_df["track_key"] = (
        plot_df["dataset_id"].astype(str)
        + "/"
        + plot_df["fov_name"].astype(str)
        + "/"
        + plot_df["track_id"].astype(str)
    )
    track_options = sorted(plot_df["track_key"].unique())
    logger.info(f"Found {len(track_options)} unique tracks")

    unique_fovs = plot_df["fov_name"].unique()
    logger.info(f"FOV name format (first 5): {list(unique_fovs[:5])}")

    return adata, plot_df, track_options, has_annotations


def validate_feature_compatibility(adatas: list[ad.AnnData]) -> None:
    """
    Validate all datasets have compatible feature dimensions.

    Parameters
    ----------
    adatas : list[ad.AnnData]
        List of AnnData objects to validate.

    Raises
    ------
    ValueError
        If datasets have incompatible feature dimensions.
    """
    if len(adatas) < 2:
        return

    ref_n_features = adatas[0].X.shape[1]
    ref_idx = 0

    for idx, adata in enumerate(adatas[1:], start=1):
        n_features = adata.X.shape[1]
        if n_features != ref_n_features:
            raise ValueError(
                f"Feature dimension mismatch: dataset {ref_idx} has {ref_n_features} features, "
                f"but dataset {idx} has {n_features} features. "
                f"All datasets must have the same number of features for joint PHATE embedding."
            )

    logger.info(
        f"✓ Feature validation passed: all {len(adatas)} datasets have {ref_n_features} features"
    )


def compute_joint_phate(
    adata_joint: ad.AnnData,
    n_components: int = 2,
    knn: int = 5,
    decay: int = 40,
    scale_embeddings: bool = False,
    **phate_kwargs,
) -> ad.AnnData:
    """
    Compute PHATE embedding on joint dataset.

    Parameters
    ----------
    adata_joint : ad.AnnData
        Combined AnnData object with feature matrix in .X
    n_components : int, optional
        Number of PHATE components (default: 2).
    knn : int, optional
        Number of nearest neighbors (default: 5).
    decay : int, optional
        Decay parameter for PHATE (default: 40).
    scale_embeddings : bool, optional
        Whether to scale embeddings (default: False).
    **phate_kwargs
        Additional keyword arguments passed to compute_phate (e.g., random_state).

    Returns
    -------
    ad.AnnData
        AnnData with PHATE embedding in .obsm['X_phate'].
    """
    logger.info("Computing joint PHATE embedding...")
    logger.info(
        f"  Parameters: n_components={n_components}, knn={knn}, decay={decay}, scale={scale_embeddings}"
    )
    if phate_kwargs:
        logger.info(f"  Additional PHATE kwargs: {phate_kwargs}")

    _, phate_embedding = compute_phate(
        adata_joint,
        n_components=n_components,
        knn=knn,
        decay=decay,
        scale_embeddings=scale_embeddings,
        **phate_kwargs,
    )

    adata_joint.obsm["X_phate"] = phate_embedding

    logger.info(
        f"✓ PHATE embedding computed: shape {adata_joint.obsm['X_phate'].shape}"
    )

    return adata_joint


def load_multiple_datasets(
    config: MultiDatasetConfig,
) -> tuple[ad.AnnData, pd.DataFrame, list, bool]:
    """
    Load and concatenate multiple AnnData objects with joint PHATE embedding.

    Parameters
    ----------
    config : MultiDatasetConfig
        Configuration object specifying datasets and PHATE parameters.

    Returns
    -------
    adata_joint : ad.AnnData
        Combined AnnData with X_phate in obsm.
    plot_df : pd.DataFrame
        DataFrame with PHATE coordinates and metadata including dataset_id.
    track_options : list
        Unique track identifiers (format: "dataset_id/fov_name/track_id").
    has_annotations : bool
        Whether any dataset has annotations.
    """
    logger.info(f"Loading {len(config.datasets)} datasets for joint PHATE embedding...")

    adatas = []
    has_any_annotations = False

    for i, dataset_cfg in enumerate(config.datasets):
        dataset_id = dataset_cfg.dataset_id
        if dataset_id is None or dataset_id == "":
            dataset_id = dataset_cfg.data_path.name.replace(".zarr", "")
            logger.info(f"Auto-detected dataset_id: {dataset_id}")

        logger.info(f"\n--- Loading dataset {i}: {dataset_id} ---")

        adata, _, _, has_annot = load_and_prepare_data(
            adata_path=dataset_cfg.adata_path,
            annotation_csv=dataset_cfg.annotation_csv,
            annotation_column=dataset_cfg.annotation_column,
            categories=dataset_cfg.categories,
            fov_filter=dataset_cfg.fov_filter,
            dataset_id=dataset_id,
        )

        logger.info(f"  Dataset '{dataset_id}': {adata.shape[0]} observations")

        adata.obs.index = [f"{dataset_id}_{idx}" for idx in adata.obs.index]

        adatas.append(adata)
        has_any_annotations = has_any_annotations or has_annot

    validate_feature_compatibility(adatas)

    logger.info("\nConcatenating datasets...")
    adata_joint = ad.concat(adatas, axis=0, join="outer", merge="unique")
    logger.info(
        f"✓ Concatenated {len(adatas)} datasets: {adata_joint.shape[0]} total observations"
    )

    if len(adatas) == 1 and config.phate_kwargs is None:
        if "X_phate" in adata_joint.obsm:
            logger.info("✓ Using existing PHATE embeddings from AnnData")
        else:
            raise ValueError(
                "No PHATE embeddings found in AnnData. Set phate_kwargs to compute embeddings."
            )
    else:
        if len(adatas) > 1:
            logger.info("Computing joint PHATE embedding across all datasets...")
        else:
            logger.info("phate_kwargs specified, recomputing PHATE embeddings...")

        if "X_phate" in adata_joint.obsm:
            del adata_joint.obsm["X_phate"]

        phate_params = dict(config.phate_kwargs) if config.phate_kwargs else {}
        n_components = phate_params.pop("n_components", 2)
        knn = phate_params.pop("knn", 5)
        decay = phate_params.pop("decay", 40)
        scale_embeddings = phate_params.pop("scale_embeddings", False)
        adata_joint = compute_joint_phate(
            adata_joint,
            n_components=n_components,
            knn=knn,
            decay=decay,
            scale_embeddings=scale_embeddings,
            **phate_params,
        )

    phate_coords = adata_joint.obsm["X_phate"]

    plot_df_dict = {
        "PHATE1": phate_coords[:, 0],
        "PHATE2": phate_coords[:, 1],
        "track_id": adata_joint.obs["track_id"].values,
        "fov_name": adata_joint.obs["fov_name"].values,
        "t": adata_joint.obs["t"].values,
        "y": adata_joint.obs["y"].values,
        "x": adata_joint.obs["x"].values,
        "id": adata_joint.obs["id"].values,
        "dataset_id": adata_joint.obs["dataset_id"].values,
    }

    annotation_column = None
    for dataset_cfg in config.datasets:
        if dataset_cfg.annotation_column:
            annotation_column = dataset_cfg.annotation_column
            break

    if (
        has_any_annotations
        and annotation_column
        and annotation_column in adata_joint.obs
    ):
        plot_df_dict["annotation"] = adata_joint.obs[annotation_column].values

    plot_df = pd.DataFrame(plot_df_dict, index=adata_joint.obs.index)

    plot_df["track_key"] = (
        plot_df["dataset_id"].astype(str)
        + "/"
        + plot_df["fov_name"].astype(str)
        + "/"
        + plot_df["track_id"].astype(str)
    )
    track_options = sorted(plot_df["track_key"].unique())
    logger.info(f"✓ Found {len(track_options)} unique tracks across all datasets")

    return adata_joint, plot_df, track_options, has_any_annotations
