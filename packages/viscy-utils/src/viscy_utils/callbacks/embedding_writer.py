"""Callback for writing embeddings to zarr store."""

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from numpy.typing import NDArray
from xarray import Dataset, open_zarr

from viscy_data._typing import INDEX_COLUMNS

__all__ = [
    "read_embedding_dataset",
    "EmbeddingWriter",
    "write_embedding_dataset",
    "get_available_index_columns",
]
_logger = logging.getLogger("lightning.pytorch")


def get_available_index_columns(dataset: Dataset, dataset_path: str | None = None) -> list[str]:
    """Get available index columns from a dataset.

    Parameters
    ----------
    dataset : Dataset
        The xarray dataset to check for index columns.
    dataset_path : str, optional
        Path for logging purposes.

    Returns
    -------
    list[str]
        List of available index columns.
    """
    available_cols = [col for col in INDEX_COLUMNS if col in dataset.coords]
    missing_cols = set(INDEX_COLUMNS) - set(available_cols)

    if missing_cols:
        path_msg = f" at {dataset_path}" if dataset_path else ""
        _logger.warning(
            f"Dataset{path_msg} is missing index columns: {sorted(missing_cols)}. "
            "This appears to be a legacy dataset format."
        )

    return available_cols


def read_embedding_dataset(path: Path) -> Dataset:
    """Read the embedding dataset written by the EmbeddingWriter callback.

    Parameters
    ----------
    path : Path
        Path to the zarr store.

    Returns
    -------
    Dataset
        Xarray dataset with features and projections.
    """
    dataset = open_zarr(path)
    available_cols = get_available_index_columns(dataset, str(path))
    return dataset.set_index(sample=available_cols)


def _move_and_stack_embeddings(predictions: Sequence, key: str) -> NDArray:
    """Move embeddings to CPU and stack them into a numpy array."""
    return torch.cat([p[key].cpu() for p in predictions], dim=0).numpy()


def write_embedding_dataset(
    output_path: Path,
    features: np.ndarray,
    index_df: pd.DataFrame,
    projections: Optional[np.ndarray] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    phate_kwargs: Optional[Dict[str, Any]] = None,
    pca_kwargs: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """Write embeddings to an AnnData Zarr Store.

    Parameters
    ----------
    output_path : Path
        Path to the zarr store.
    features : np.ndarray
        Array of shape (n_samples, n_features) containing the embeddings.
    index_df : pd.DataFrame
        DataFrame containing the index information for each embedding.
    projections : np.ndarray, optional
        Array of shape (n_samples, n_projections) containing projections.
    umap_kwargs : dict, optional
        Keyword arguments passed to UMAP, by default None.
    phate_kwargs : dict, optional
        Keyword arguments passed to PHATE, by default None.
    pca_kwargs : dict, optional
        Keyword arguments passed to PCA, by default None.
    overwrite : bool, optional
        Whether to overwrite existing zarr store, by default False.
    """
    import anndata as ad

    if hasattr(ad, "settings") and hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path {output_path} already exists.")

    ultrack_indices = index_df.copy()
    ultrack_indices["fov_name"] = ultrack_indices["fov_name"].str.strip("/")
    n_samples = len(features)

    adata = ad.AnnData(X=features, obs=ultrack_indices)
    if projections is not None:
        adata.obsm["X_projections"] = projections

    if umap_kwargs:
        from viscy_utils.evaluation.dimensionality_reduction import (
            _fit_transform_umap,
        )

        if umap_kwargs["n_neighbors"] >= n_samples:
            _logger.warning(
                f"Reducing n_neighbors from {umap_kwargs['n_neighbors']} "
                f"to {min(15, n_samples // 2)} due to small dataset size"
            )
            umap_kwargs["n_neighbors"] = min(15, n_samples // 2)

        _logger.debug(f"Using UMAP kwargs: {umap_kwargs}")
        _, UMAP = _fit_transform_umap(features, **umap_kwargs)
        adata.obsm["X_umap"] = UMAP

    if phate_kwargs:
        from viscy_utils.evaluation.dimensionality_reduction import compute_phate

        _logger.debug(f"Using PHATE kwargs: {phate_kwargs}")
        if phate_kwargs["knn"] >= n_samples:
            _logger.warning(
                f"Reducing knn from {phate_kwargs['knn']} to {max(2, n_samples // 2)} due to small dataset size"
            )
            phate_kwargs["knn"] = max(2, n_samples // 2)

        try:
            _logger.debug("Computing PHATE")
            _, PHATE = compute_phate(features, **phate_kwargs)
            adata.obsm["X_phate"] = PHATE
        except Exception as e:
            _logger.warning(f"PHATE computation failed: {str(e)}")

    if pca_kwargs:
        from viscy_utils.evaluation.dimensionality_reduction import compute_pca

        _logger.debug(f"Using PCA kwargs: {pca_kwargs}")
        try:
            _logger.debug("Computing PCA")
            PCA_features, _ = compute_pca(features, **pca_kwargs)
            adata.obsm["X_pca"] = PCA_features
        except Exception as e:
            _logger.warning(f"PCA computation failed: {str(e)}")

    _logger.debug(f"Writing dataset to {output_path}")
    adata.write_zarr(output_path)


class EmbeddingWriter(BasePredictionWriter):
    """Callback to write embeddings to a zarr store.

    Parameters
    ----------
    output_path : Path
        Path to the zarr store.
    write_interval : str, optional
        When to write the embeddings, by default 'epoch'.
    umap_kwargs : dict, optional
        Keyword arguments passed to UMAP, by default None.
    phate_kwargs : dict, optional
        Keyword arguments passed to PHATE.
    pca_kwargs : dict, optional
        Keyword arguments passed to PCA.
    overwrite : bool, optional
        Whether to overwrite existing output, by default False.
    """

    def __init__(
        self,
        output_path: Path,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
        umap_kwargs: dict | None = None,
        phate_kwargs: dict | None = {
            "knn": 5,
            "decay": 40,
            "n_jobs": -1,
        },
        pca_kwargs: dict | None = {"n_components": 8},
        overwrite: bool = False,
    ):
        super().__init__(write_interval)
        self.output_path = Path(output_path)
        self.umap_kwargs = umap_kwargs
        self.phate_kwargs = phate_kwargs
        self.pca_kwargs = pca_kwargs
        self.overwrite = overwrite

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Check output path before prediction starts."""
        if self.output_path.exists():
            raise FileExistsError(f"Output path {self.output_path} already exists.")
        _logger.debug(f"Writing embeddings to {self.output_path}")

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence,
        batch_indices: Sequence[int],
    ) -> None:
        """Write predictions and dimensionality reductions to a zarr store."""
        features = _move_and_stack_embeddings(predictions, "features")
        projections = _move_and_stack_embeddings(predictions, "projections")
        ultrack_indices = pd.concat([pd.DataFrame(p["index"]) for p in predictions])

        write_embedding_dataset(
            output_path=self.output_path,
            features=features,
            index_df=ultrack_indices,
            projections=projections,
            umap_kwargs=self.umap_kwargs,
            phate_kwargs=self.phate_kwargs,
            pca_kwargs=self.pca_kwargs,
            overwrite=self.overwrite,
        )
