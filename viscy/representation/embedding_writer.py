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

from viscy.data.triplet import INDEX_COLUMNS
from viscy.representation.engine import ContrastivePrediction
from viscy.representation.evaluation.dimensionality_reduction import (
    _fit_transform_umap,
    compute_pca,
    compute_phate,
)

__all__ = ["read_embedding_dataset", "EmbeddingWriter", "write_embedding_dataset"]
_logger = logging.getLogger("lightning.pytorch")


def read_embedding_dataset(path: Path) -> Dataset:
    """
    Read the embedding dataset written by the EmbeddingWriter callback.
    Supports both legacy datasets (without x/y coordinates) and new datasets.

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
    # Check which index columns are present in the dataset
    available_cols = [col for col in INDEX_COLUMNS if col in dataset.coords]

    # Warn if any INDEX_COLUMNS are missing
    missing_cols = set(INDEX_COLUMNS) - set(available_cols)
    if missing_cols:
        _logger.warning(
            f"Dataset at {path} is missing index columns: {sorted(missing_cols)}. "
            "This appears to be a legacy dataset format."
        )

    return dataset.set_index(sample=available_cols)


def _move_and_stack_embeddings(
    predictions: Sequence[ContrastivePrediction], key: str
) -> NDArray:
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
    """
    Write embeddings to a zarr store in an Xarray-compatible format.

    Parameters
    ----------
    output_path : Path
        Path to the zarr store.
    features : np.ndarray
        Array of shape (n_samples, n_features) containing the embeddings.
    index_df : pd.DataFrame
        DataFrame containing the index information for each embedding.
    projections : np.ndarray, optional
        Array of shape (n_samples, n_projections) containing projection values, by default None.
    umap_kwargs : Dict[str, Any], optional
        Keyword arguments passed to UMAP, by default None (i.e. UMAP is not computed)
        Common parameters include:
        - n_components: int, dimensions for projection (default: 2)
        - n_neighbors: int, number of nearest neighbors (default: 15)
        - min_dist: float, minimum distance between points (default: 0.1)
        - metric: str, distance metric (default: 'euclidean')
    phate_kwargs : Dict[str, Any], optional
        Keyword arguments passed to PHATE, by default None (i.e. PHATE is not computed)
        Common parameters include:
        - knn: int, number of nearest neighbors (default: 5)
        - decay: int, decay rate for kernel (default: 40)
        - n_jobs: int, number of jobs for parallel processing
        - t: int, number of diffusion steps
        - potential_method: str, potential method to use
    pca_kwargs : Dict[str, Any], optional
        Keyword arguments passed to PCA, by default None (i.e. PCA is not computed)
        Common parameters include:
        - n_components: int, dimensions for projection (default: 2)
        - whiten: bool, whether to whiten (default: False)
    overwrite : bool, optional
        Whether to overwrite existing zarr store, by default False.

    Raises
    ------
    FileExistsError
        If output_path exists and overwrite is False.
    """
    output_path = Path(output_path)

    # Check if output_path exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path {output_path} already exists.")

    # Create a copy of the index DataFrame to avoid modifying the original
    ultrack_indices = index_df.copy()
    n_samples = len(features)

    # Set up default kwargs for each method
    if umap_kwargs:
        if umap_kwargs["n_neighbors"] >= n_samples:
            _logger.warning(
                f"Reducing n_neighbors from {umap_kwargs['n_neighbors']} to {min(15, n_samples // 2)} due to small dataset size"
            )
            umap_kwargs["n_neighbors"] = min(15, n_samples // 2)

        _logger.debug(f"Using UMAP kwargs: {umap_kwargs}")
        _, UMAP = _fit_transform_umap(features, **umap_kwargs)
        for i in range(UMAP.shape[1]):
            ultrack_indices[f"UMAP{i+1}"] = UMAP[:, i]

    if phate_kwargs:
        # Update with user-provided kwargs
        _logger.debug(f"Using PHATE kwargs: {phate_kwargs}")
        # Ensure knn is appropriate for dataset size for PHATE
        if phate_kwargs["knn"] >= n_samples:
            _logger.warning(
                f"Reducing knn from {phate_kwargs['knn']} to {max(2, n_samples // 2)} due to small dataset size"
            )
            phate_kwargs["knn"] = max(2, n_samples // 2)

        # Compute PHATE
        try:
            _logger.debug("Computing PHATE")
            _, PHATE = compute_phate(features, **phate_kwargs)
            for i in range(PHATE.shape[1]):
                ultrack_indices[f"PHATE{i+1}"] = PHATE[:, i]
        except Exception as e:
            _logger.warning(f"PHATE computation failed: {str(e)}")

    if pca_kwargs:
        # Update with user-provided kwargs
        _logger.debug(f"Using PCA kwargs: {pca_kwargs}")
        try:
            _logger.debug("Computing PCA")
            PCA_features, _ = compute_pca(features, **pca_kwargs)
            for i in range(PCA_features.shape[1]):
                ultrack_indices[f"PCA{i+1}"] = PCA_features[:, i]
        except Exception as e:
            _logger.warning(f"PCA computation failed: {str(e)}")

    # Create multi-index and dataset
    index = pd.MultiIndex.from_frame(ultrack_indices)

    # Create dataset dictionary with features
    dataset_dict = {"features": (("sample", "features"), features)}

    # Add projections if provided
    if projections is not None:
        dataset_dict["projections"] = (("sample", "projections"), projections)

    # Create the dataset
    dataset = Dataset(dataset_dict, coords={"sample": index}).reset_index("sample")

    _logger.debug(f"Writing dataset to {output_path}")
    with dataset.to_zarr(output_path, mode="w") as zarr_store:
        zarr_store.close()


class EmbeddingWriter(BasePredictionWriter):
    """
    Callback to write embeddings to a zarr store in an Xarray-compatible format.

    Parameters
    ----------
    output_path : Path
        Path to the zarr store.
    write_interval : Literal["batch", "epoch", "batch_and_epoch"], optional
        When to write the embeddings, by default 'epoch'.
    umap_kwargs : dict, optional
        Keyword arguments passed to UMAP, by default None (i.e. UMAP is not computed).
    phate_kwargs : dict, optional
        Keyword arguments passed to PHATE, by default PHATE is computed with default parameters.
    pca_kwargs : dict, optional
        Keyword arguments passed to PCA, by default PCA is computed with default parameters.
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
            "random_state": 42,
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
        if self.output_path.exists():
            raise FileExistsError(f"Output path {self.output_path} already exists.")
        _logger.debug(f"Writing embeddings to {self.output_path}")

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[ContrastivePrediction],
        batch_indices: Sequence[int],
    ) -> None:
        """Write predictions and dimensionality reductions to a zarr store.

        Parameters
        ----------
        trainer : Trainer
            Placeholder, ignored.
        pl_module : LightningModule
            Placeholder, ignored.
        predictions : Sequence[ContrastivePrediction]
            Sequence of output from the prediction steps.
        batch_indices : Sequence[int]
            Placeholder, ignored.
        """
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
