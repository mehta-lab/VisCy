"""Dynacell LightningModule for supervised virtual staining benchmarks."""

import inspect
from typing import Literal, Sequence

import numpy as np
import torch
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import DivisiblePad
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR

from viscy_data import Sample
from viscy_models import Unet3d, UNeXt2
from viscy_models.celldiff import UNetViT3D
from viscy_utils.log_images import detach_sample, log_image_grid

_ARCHITECTURE: dict[str, type[nn.Module]] = {
    "UNetViT3D": UNetViT3D,
    "FNet3D": Unet3d,
    "UNeXt2": UNeXt2,
}


def _make_divisible_pad(model: nn.Module) -> DivisiblePad:
    """Build a DivisiblePad matching the model's spatial downsampling axes.

    Parameters
    ----------
    model : nn.Module
        A model with ``num_blocks`` and optionally ``downsamples_z``.

    Returns
    -------
    DivisiblePad
        Pads YX (and Z if ``downsamples_z``) to the nearest multiple of
        ``2**num_blocks``.
    """
    down_factor = 2**model.num_blocks
    if getattr(model, "downsamples_z", False):
        return DivisiblePad((0, down_factor, down_factor, down_factor))
    return DivisiblePad((0, 0, down_factor, down_factor))


def _center_crop_to_shape(tensor: Tensor, spatial_shape: tuple[int, ...]) -> Tensor:
    """Center-crop trailing spatial dimensions to the requested shape."""
    slices = [slice(None)] * tensor.ndim
    start_dim = tensor.ndim - len(spatial_shape)
    for dim, size in enumerate(spatial_shape, start=start_dim):
        current = tensor.shape[dim]
        if current < size:
            raise ValueError(f"Cannot crop dimension {dim} from {current} to {size}")
        start = (current - size) // 2
        slices[dim] = slice(start, start + size)
    return tensor[tuple(slices)]


class DynacellUNet(LightningModule):
    """Supervised regression U-Net for benchmark virtual staining.

    Parameters
    ----------
    architecture : {"UNetViT3D", "FNet3D", "UNeXt2"}
        Architecture key selecting the backbone.
    model_config : dict | None
        Keyword arguments forwarded to the backbone constructor.
    loss_function : nn.Module | None
        Loss function. Defaults to ``nn.MSELoss()``.
    lr : float
        Learning rate.
    schedule : {"WarmupCosine", "Constant"}
        LR scheduler type.
    log_batches_per_epoch : int
        Batches to log images per epoch.
    log_samples_per_batch : int
        Samples per batch to log.
    example_input_yx_shape : Sequence[int]
        YX shape for example input (used by FNet3D for graph logging).
        Ignored when the model provides ``input_spatial_size``.
    ckpt_path : str | None
        Checkpoint path to load model weights.
    """

    def __init__(
        self,
        architecture: Literal["UNetViT3D", "FNet3D", "UNeXt2"] = "UNetViT3D",
        model_config: dict | None = None,
        loss_function: nn.Module | None = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_yx_shape: Sequence[int] = (256, 256),
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_function", "ckpt_path"])
        if model_config is None:
            model_config = {}
        net_class = _ARCHITECTURE.get(architecture)
        if net_class is None:
            raise ValueError(f"Architecture {architecture!r} not in {set(_ARCHITECTURE)}")
        self.model = net_class(**model_config)
        self.loss_function = loss_function if loss_function is not None else nn.MSELoss()
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.training_step_outputs: list = []
        # Each entry is a list of (loss, batch_size) tuples for weighted aggregation.
        self.validation_losses: list[list[tuple[Tensor, int]]] = []
        self.validation_step_outputs: list = []

        # Cache fg_mask compatibility to avoid per-batch inspect.signature().
        sig = inspect.signature(self.loss_function.forward)
        self._loss_accepts_fg_mask = "fg_mask" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        # Build example_input_array for graph logging (TensorBoard/W&B).
        in_channels = model_config.get("in_channels") or 1
        if hasattr(self.model, "input_spatial_size"):
            # UNetViT3D: must use exact spatial dims.
            d, h, w = self.model.input_spatial_size
        else:
            # FNet3D: flexible spatial, use in_stack_depth + user YX.
            d = model_config.get("in_stack_depth") or 5
            h, w = example_input_yx_shape
        self.example_input_array = torch.rand(1, in_channels, d, h, w)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"])

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass through the model.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Model output.
        """
        return self.model(x)

    def _compute_loss(self, pred: Tensor, target: Tensor, batch: Sample) -> Tensor:
        """Compute loss, optionally passing fg_mask to the loss function."""
        if "fg_mask" in batch:
            if not self._loss_accepts_fg_mask:
                raise TypeError(
                    f"{type(self.loss_function).__name__} does not accept 'fg_mask'. "
                    f"Use SpotlightLoss or remove fg_mask_key from the data config."
                )
            return self.loss_function(pred, target, fg_mask=batch["fg_mask"])
        return self.loss_function(pred, target)

    def training_step(self, batch: Sample, batch_idx: int) -> Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Training loss.
        """
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)
        loss = self._compute_loss(pred, target, batch)
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        return loss

    def validation_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        """Execute a single validation step.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        pred = self.forward(source)
        loss = self._compute_loss(pred, target, batch)
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append((loss.detach(), source.shape[0]))
        self.log(
            f"loss/val/{dataloader_idx}",
            loss,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))

    def on_predict_start(self) -> None:
        """Build the divisible-pad transform for tiled inference."""
        self._predict_pad = _make_divisible_pad(self.model)

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Execute a single prediction step.

        Pads the input tile to the nearest multiple of the model's downsampling
        factor, runs the forward pass, then crops back to the original shape.

        Parameters
        ----------
        batch : Sample
            Input batch. Only ``"source"`` is used.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index, defaults to 0.

        Returns
        -------
        Tensor
            Model prediction, cropped to the input spatial shape.
        """
        source = batch["source"]
        original_shape = source.shape[2:]
        source = self._predict_pad(source)
        prediction = self.forward(source)
        return _center_crop_to_shape(prediction, original_shape)

    def on_train_epoch_end(self):
        """Log training image samples."""
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        """Log validation samples and aggregate loss weighted by batch size."""
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        if self.validation_losses:
            # Compute per-dataloader weighted mean, then weight dataloaders by sample count.
            dl_means, dl_totals = [], []
            for dl_batches in self.validation_losses:
                losses, sizes = zip(*dl_batches)
                # Create sizes on the same device as the losses to avoid device
                # mismatch on GPU/DDP where losses are on the model device.
                sizes_t = torch.tensor(sizes, dtype=torch.float, device=losses[0].device)
                dl_means.append((torch.stack(losses) * sizes_t).sum() / sizes_t.sum())
                dl_totals.append(sizes_t.sum())
            total_n = torch.stack(dl_totals).sum()
            weighted = sum(m * n for m, n in zip(dl_means, dl_totals))
            self.log("loss/validate", weighted / total_n, sync_dist=True)
        self.validation_step_outputs.clear()
        self.validation_losses.clear()

    def configure_optimizers(self):
        """Configure AdamW optimizer with LR scheduler."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if self.schedule == "WarmupCosine":
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=3,
                t_total=self.trainer.estimated_stepping_batches,
                warmup_multiplier=1e-3,
            )
            # t_total is a step count; must step the scheduler every optimizer
            # step, not once per epoch (Lightning's default).
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif self.schedule == "Constant":
            scheduler = ConstantLR(optimizer, factor=1, total_iters=self.trainer.max_epochs)
        else:
            raise ValueError(f"Unknown schedule {self.schedule!r}, expected 'WarmupCosine' or 'Constant'")
        return [optimizer], [scheduler]

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        """Log image grid to the active logger."""
        if not imgs or not self.trainer.is_global_zero or self.logger is None:
            return
        log_image_grid(self.logger, key, imgs, self.current_epoch)
