import logging
from pathlib import Path
from typing import Literal, Sequence

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
    compute_phate,
)

__all__ = ["read_embedding_dataset", "EmbeddingWriter"]
_logger = logging.getLogger("lightning.pytorch")


def read_embedding_dataset(path: Path) -> Dataset:
    """
    Read the embedding dataset written by the EmbeddingWriter callback.

    Parameters
    ----------
    path : Path
        Path to the zarr store.

    Returns
    -------
    Dataset
        Xarray dataset with features and projections.
    """
    return open_zarr(path).set_index(sample=INDEX_COLUMNS)


def _move_and_stack_embeddings(
    predictions: Sequence[ContrastivePrediction], key: str
) -> NDArray:
    """Move embeddings to CPU and stack them into a numpy array."""
    return torch.cat([p[key].cpu() for p in predictions], dim=0).numpy()


class EmbeddingWriter(BasePredictionWriter):
    """
    Callback to write embeddings to a zarr store in an Xarray-compatible format.

    Parameters
    ----------
    output_path : Path
        Path to the zarr store.
    write_interval : Literal["batch", "epoch", "batch_and_epoch"], optional
        When to write the embeddings, by default 'epoch'.
    phate_kwargs : dict, optional
        Keyword arguments passed to PHATE, by default None.
        Common parameters include:
        - knn: int, number of nearest neighbors (default: 5)
        - decay: int, decay rate for kernel (default: 40)
        - n_jobs: int, number of jobs for parallel processing
        - t: int, number of diffusion steps
        - potential_method: str, potential method to use
        See phate.PHATE for all available parameters.
    """

    def __init__(
        self,
        output_path: Path,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
        phate_kwargs: dict | None = None,
    ):
        super().__init__(write_interval)
        self.output_path = Path(output_path)

        # Set default PHATE parameters
        default_phate_kwargs = {
            "n_components": 2,
            "knn": 5,
            "decay": 40,
            "n_jobs": -1,
            "random_state": 42,
        }
        if phate_kwargs is not None:
            default_phate_kwargs.update(phate_kwargs)
        self.phate_kwargs = default_phate_kwargs

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

        _logger.info(
            f"Computing dimensionality reductions for {len(features)} samples."
        )
        _, umap = _fit_transform_umap(features, n_components=2, normalize=True)
        _, phate = compute_phate(
            features,
            **self.phate_kwargs,
        )

        # Add dimensionality reduction coordinates
        ultrack_indices["UMAP1"], ultrack_indices["UMAP2"] = umap[:, 0], umap[:, 1]
        ultrack_indices["PHATE1"], ultrack_indices["PHATE2"] = phate[:, 0], phate[:, 1]

        # Create multi-index and dataset
        index = pd.MultiIndex.from_frame(ultrack_indices)
        dataset = Dataset(
            {
                "features": (("sample", "features"), features),
                "projections": (("sample", "projections"), projections),
            },
            coords={"sample": index},
        ).reset_index("sample")

        _logger.debug(f"Writing predictions dataset:\n{dataset}")
        with dataset.to_zarr(self.output_path, mode="w") as zarr_store:
            zarr_store.close()
