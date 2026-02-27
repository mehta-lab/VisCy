"""Callback to snapshot embeddings during training for visualization."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from viscy_data._typing import ULTRACK_INDEX_COLUMNS, TripletSample
from viscy_utils.callbacks.embedding_writer import EmbeddingWriter, write_embedding_dataset

_logger = logging.getLogger("lightning.pytorch")


def _extract_mid_z_patches(images: torch.Tensor) -> np.ndarray:
    """Extract mid-Z slice patches from 5D tensors (B, C, Z, H, W).

    Parameters
    ----------
    images : torch.Tensor
        5D tensor of shape (B, C, Z, H, W).

    Returns
    -------
    np.ndarray
        4D array of shape (B, C, H, W) at the middle Z slice.
    """
    mid_z = images.shape[2] // 2
    return images[:, :, mid_z].detach().cpu().numpy()


class EmbeddingSnapshotCallback(Callback):
    """Snapshot validation embeddings and image patches every N epochs.

    Runs a single forward pass on the first validation batch at the
    specified epoch interval. Writes an AnnData zarr containing features,
    projections, tracking index, and optionally the mid-Z image patches.

    Only rank 0 writes to disk. No extra collective operations are introduced,
    so this is safe for DDP training.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write epoch snapshots. Each snapshot is saved as
        ``epoch_{N}.zarr`` inside this directory.
    every_n_epochs : int
        Frequency of snapshots in epochs.
    store_images : bool
        If True, store mid-Z image patches in ``obsm["X_images"]``.
    pca_kwargs : dict, optional
        Keyword arguments for PCA computation. Set to None to skip.
    """

    def __init__(
        self,
        output_dir: str | Path,
        every_n_epochs: int = 10,
        store_images: bool = True,
        pca_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.store_images = store_images
        self.pca_kwargs = pca_kwargs
        self._collecting = False
        self._features: torch.Tensor | None = None
        self._projections: torch.Tensor | None = None
        self._index: dict | None = None
        self._images: np.ndarray | None = None

    def _should_collect(self, trainer: Trainer) -> bool:
        return trainer.current_epoch % self.every_n_epochs == 0

    def _reset(self):
        self._collecting = False
        self._features = None
        self._projections = None
        self._index = None
        self._images = None

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Enable collection if current epoch matches snapshot frequency."""
        if self._should_collect(trainer):
            self._collecting = True

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: TripletSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect embeddings from the first validation batch when active."""
        if not self._collecting or self._features is not None:
            return
        with torch.no_grad():
            features, projections = pl_module(batch["anchor"])
        self._features = features.detach().cpu()
        self._projections = projections.detach().cpu()
        self._index = batch.get("index")
        if self.store_images:
            self._images = _extract_mid_z_patches(batch["anchor"])

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write snapshot to disk on rank 0 when collection is complete."""
        if not self._collecting or self._features is None:
            self._reset()
            return
        if trainer.global_rank != 0:
            self._reset()
            return

        epoch = trainer.current_epoch
        output_path = self.output_dir / f"epoch_{epoch}.zarr"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        features_np = self._features.numpy()
        projections_np = self._projections.numpy()

        if self._index is not None:
            available = {k: v for k, v in self._index.items() if k in ULTRACK_INDEX_COLUMNS}
            index_df = pd.DataFrame(available)
        else:
            index_df = pd.DataFrame({"fov_name": ["unknown"] * features_np.shape[0]})

        uns_metadata = EmbeddingWriter._collect_data_provenance(trainer)
        uns_metadata["epoch"] = epoch

        write_embedding_dataset(
            output_path=output_path,
            features=features_np,
            index_df=index_df,
            projections=projections_np,
            pca_kwargs=self.pca_kwargs,
            overwrite=True,
            uns_metadata=uns_metadata,
        )

        if self.store_images and self._images is not None:
            import anndata as ad

            adata = ad.read_zarr(output_path)
            b = self._images.shape[0]
            adata.obsm["X_images"] = self._images.reshape(b, -1)
            # Mid-Z extraction produces (C, Y, X) per sample
            adata.uns["image_shape_cyx"] = list(self._images.shape[1:])
            adata.write_zarr(output_path)

        _logger.info(f"Embedding snapshot saved: {output_path} ({features_np.shape[0]} samples)")
        self._reset()
