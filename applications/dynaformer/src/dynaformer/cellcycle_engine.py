"""LightningModule for cell-cycle prediction from temporally integrated foundation-model embeddings.

One module trains all three task variants of :class:`~dynaformer.models.TemporalEmbeddingModel`:

- regression: f(ê_t) predicts the nucleus's Geminin-mKO and PIP-mVenus medians;
- virtual staining: g(p_t, ê_t) predicts the two reporter images;
- multitask: both, with ``regression_weight * L_f + stain_weight * L_g``.

Both losses are an equal sum of absolute and squared error in normalized units (counts minus the frame's background,
over the reporter's scale), so the regression targets and the stain images share units, and the stain head is also
scored on the median of its prediction inside the true nucleus mask: the same quantity the regression head predicts.

Validation and testing run every track whole. Validation logs frame-, track- and phase-level scores of each readout
(:func:`~dynaformer.cellcycle_metrics.score_frames`); testing also writes the per-frame predictions and the scores to
``output_dir``. The reporters' names, their scales and the phase thresholds come from the data module (``reporters``,
``stats``), so the same module scores the PIP-FUCCI reporters of :mod:`dynaformer.cellcycle_data` and the FUCCI
reporters of :mod:`dynaformer.leger_data`.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from monai.optimizers import WarmupCosineSchedule
from torch import Tensor

from dynaformer.cellcycle_metrics import masked_median, pixel_pearson, score_frames
from dynaformer.models.temporal_embedding import TemporalEmbeddingModel
from viscy_utils.log_images import detach_sample, log_image_grid

__all__ = ["TemporalEmbeddingModule"]

_logger = logging.getLogger("lightning.pytorch")


def _l1_mse(pred: Tensor, target: Tensor) -> Tensor:
    residual = pred.float() - target.float()
    return residual.abs().mean() + residual.pow(2).mean()


class TemporalEmbeddingModule(LightningModule):
    """Train and evaluate a :class:`~dynaformer.models.TemporalEmbeddingModel` on cell-cycle reporters.

    Parameters
    ----------
    model : TemporalEmbeddingModel
        E (cached) -> T -> f and/or g.
    regression_weight : float
        Weight of the regression loss, by default 1.
    stain_weight : float
        Weight of the stain loss, by default 1.
    lr : float
        Peak learning rate, by default 3e-4.
    weight_decay : float
        AdamW weight decay on weights with more than one dimension, by default 0.05.
    warmup_steps : int
        Linear warm-up steps before the cosine decay, by default 500.
    log_samples : int
        Validation images logged per epoch, by default 6.
    output_dir : str | None
        Where testing writes ``predictions_<split>.csv`` and ``metrics_<split>.json``; ``None`` uses the trainer's
        log directory, by default ``None``.
    """

    def __init__(
        self,
        model: TemporalEmbeddingModel,
        regression_weight: float = 1.0,
        stain_weight: float = 1.0,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        log_samples: int = 6,
        output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.regression_weight = regression_weight
        self.stain_weight = stain_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.log_samples = log_samples
        self.output_dir = output_dir
        self._records: list[dict] = []
        self._samples: list[list[np.ndarray]] = []

    def forward(self, batch: dict) -> dict[str, Tensor]:
        """Run the model on a batch from :class:`~dynaformer.cellcycle_data.CellCycleDataModule`."""
        return self.model(batch.get("embeddings"), batch.get("source"), batch.get("image_index"), batch.get("frames"))

    def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> dict:
        """Move tensors only; the batch is a flat dict of tensors."""
        return {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def _losses(self, batch: dict, out: dict[str, Tensor], stage: str) -> Tensor:
        total = torch.zeros((), device=self.device)
        size = len(batch["track_id"])
        if "regression" in out:
            loss = _l1_mse(out["regression"], batch["target"])
            self.log(f"loss/regression/{stage}", loss, batch_size=size, sync_dist=stage != "train")
            total = total + self.regression_weight * loss
        if "stain" in out:
            loss = _l1_mse(out["stain"], batch["target_image"])
            self.log(f"loss/stain/{stage}", loss, batch_size=size, sync_dist=stage != "train")
            total = total + self.stain_weight * loss
        self.log(f"loss/{stage}", total, batch_size=size, prog_bar=True, sync_dist=stage != "train")
        return total

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """One optimization step."""
        return self._losses(batch, self(batch), "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Score one batch of whole validation tracks."""
        out = self(batch)
        self._losses(batch, out, "val")
        self._collect(batch, out)
        if "stain" in out and len(self._samples) < self.log_samples:
            n = min(self.log_samples - len(self._samples), len(out["stain"]))
            views = [batch["source"][:n], batch["target_image"][:n], out["stain"][:n].float()]
            self._samples.extend(detach_sample([v.unsqueeze(2) for v in views], n))

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Predict one batch of whole test tracks, every frame imaged."""
        self._collect(batch, self(batch))

    def _collect(self, batch: dict, out: dict[str, Tensor]) -> None:
        """Keep per-frame truths and predictions for the epoch-level scores."""
        record = {
            "row": batch["rows"].flatten().cpu().numpy(),
            "track_id": batch["track_id"][:, None].expand_as(batch["rows"]).flatten().cpu().numpy(),
            "t": batch["t"].flatten().cpu().numpy(),
            "true": batch["target"].flatten(0, 1).float().cpu().numpy(),
        }
        if "regression" in out:
            record["regression"] = out["regression"].flatten(0, 1).float().cpu().numpy()
        if "stain" in out:
            record["image_index"] = batch["image_index"].cpu().numpy()
            record["stain"] = masked_median(out["stain"], batch["mask"]).cpu().numpy()
            record["pixel_r"] = pixel_pearson(out["stain"], batch["target_image"]).cpu().numpy()
        self._records.append(record)

    def _frame_table(self) -> pd.DataFrame:
        """One row per evaluated frame: truth and every readout's prediction, in normalized units."""
        reporters = self.trainer.datamodule.reporters
        records = self._records
        if dist.is_available() and dist.is_initialized():
            gathered: list = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, records)
            records = [r for rank in gathered for r in rank]
        tables = []
        for record in records:
            table = pd.DataFrame({"row": record["row"], "track_id": record["track_id"], "t": record["t"]})
            for i, reporter in enumerate(reporters):
                table[f"true_{reporter}"] = record["true"][:, i]
                if "regression" in record:
                    table[f"regression_{reporter}"] = record["regression"][:, i]
                if "stain" in record:
                    table[f"stain_{reporter}"] = np.nan
                    table.loc[record["image_index"], f"stain_{reporter}"] = record["stain"][:, i]
                    table[f"pixel_r_{reporter}"] = np.nan
                    table.loc[record["image_index"], f"pixel_r_{reporter}"] = record["pixel_r"][:, i]
            tables.append(table)
        return pd.concat(tables, ignore_index=True).sort_values("row", ignore_index=True)

    def _score(self, table: pd.DataFrame) -> dict[str, dict]:
        """Scores of each readout present in the frame table."""
        stats, reporters = self.trainer.datamodule.stats, self.trainer.datamodule.reporters
        scale = np.asarray(stats["fucci"]["scale"])
        scores = {}
        for readout in ("regression", "stain"):
            if f"{readout}_{reporters[0]}" not in table:
                continue
            frames = table[["track_id"] + [f"true_{r}" for r in reporters]].copy()
            for reporter in reporters:
                frames[f"pred_{reporter}"] = table[f"{readout}_{reporter}"]
            scores[readout] = score_frames(frames, scale, stats["thresholds"], reporters=reporters)
            if readout == "stain":
                for reporter in reporters:
                    scores[readout][f"pixel_pearson_{reporter}"] = float(table[f"pixel_r_{reporter}"].mean())
                scores[readout]["pixel_pearson"] = float(
                    np.mean([scores[readout][f"pixel_pearson_{r}"] for r in reporters])
                )
        return scores

    def on_validation_epoch_end(self) -> None:
        """Log every readout's scores and a grid of example images."""
        if self._records and not self.trainer.sanity_checking:
            for readout, metrics in self._score(self._frame_table()).items():
                self.log_dict({f"val/{readout}/{k}": v for k, v in metrics.items()}, sync_dist=False)
        # The image grid helper writes to W&B or TensorBoard only.
        if self._samples and isinstance(self.logger, (TensorBoardLogger, WandbLogger)) and self.trainer.is_global_zero:
            log_image_grid(self.logger, "val_samples", self._samples, self.global_step)
        self._records.clear()
        self._samples.clear()

    def on_test_epoch_end(self) -> None:
        """Log the scores and write the per-frame predictions and the scores."""
        table = self._frame_table()
        scores = self._score(table)
        for readout, metrics in scores.items():
            self.log_dict({f"test/{readout}/{k}": v for k, v in metrics.items()}, sync_dist=False)
        if self.trainer.is_global_zero:
            out_dir = Path(self.output_dir or self.trainer.log_dir or ".")
            out_dir.mkdir(parents=True, exist_ok=True)
            split = self.trainer.datamodule.test_split
            scale = np.asarray(self.trainer.datamodule.stats["fucci"]["scale"])
            counts = table.copy()
            for column in counts.columns:
                for i, reporter in enumerate(self.trainer.datamodule.reporters):
                    if column.endswith(f"_{reporter}") and not column.startswith("pixel_r"):
                        counts[column] = counts[column] * scale[i]
            counts.to_csv(out_dir / f"predictions_{split}.csv", index=False, float_format="%.4g")
            with open(out_dir / f"metrics_{split}.json", "w") as f:
                json.dump(scores, f, indent=1)
            _logger.info(f"Wrote predictions and metrics of the {split} split to {out_dir}")
        self._records.clear()

    def configure_optimizers(self) -> tuple[list, list]:
        """AdamW (no decay on biases and norms) with linear warm-up and cosine decay, stepped per batch."""
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                (decay if parameter.ndim > 1 else no_decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr,
        )
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=self.warmup_steps, t_total=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Drop an attached frozen encoder from checkpoints; it is rebuilt from its weights. A trained one stays."""
        if self.model.encoder is not None and self.model.encoder.trainable:
            return
        state = checkpoint["state_dict"]
        for key in [k for k in state if k.startswith("model.encoder.")]:
            del state[key]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: D102
        if self.model.encoder is not None and not self.model.encoder.trainable:
            encoder_state = {f"model.encoder.{k}": v for k, v in self.model.encoder.state_dict().items()}
            state_dict = {**encoder_state, **state_dict}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
