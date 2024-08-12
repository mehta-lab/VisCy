import logging
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from xarray import Dataset, open_zarr

from viscy.data.triplet import INDEX_COLUMNS

__all__ = ["read_embedding_dataset", "EmbeddingWriter"]
_logger = logging.getLogger("lightning.pytorch")


def read_embedding_dataset(path: Path) -> Dataset:
    return open_zarr(path).set_index(sample=INDEX_COLUMNS)


class EmbeddingWriter(BasePredictionWriter):
    def __init__(
        self,
        output_path: Path,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
    ):
        super().__init__(write_interval)
        self.output_path = Path(output_path)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.output_path.exists():
            raise FileExistsError(f"Output path {self.output_path} already exists.")
        _logger.debug(f"Writing embeddings to {self.output_path}")

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[dict],
        batch_indices: Sequence[int],
    ) -> None:
        features = torch.cat([p["features"] for p in predictions], dim=0)
        projections = torch.cat([p["projections"] for p in predictions], dim=0)
        index = pd.MultiIndex.from_frame(
            pd.concat([pd.DataFrame(p["index"]) for p in predictions])
        )
        dataset = Dataset(
            {
                "features": (("sample", "features"), features.cpu().numpy()),
                "projections": (
                    ("sample", "projections"),
                    projections.cpu().numpy(),
                ),
            },
            coords={"sample": index},
        ).reset_index("sample")
        _logger.debug(f"Wrtiting predictions dataset:\n{dataset}")
        zarr_store = dataset.to_zarr(self.output_path, mode="w")
        zarr_store.close()
