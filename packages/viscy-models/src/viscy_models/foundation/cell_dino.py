"""CELL-DINO foundation model wrapper for frozen feature extraction.

CELL-DINO is a DINOv2-architecture ViT pretrained on fluorescence microscopy
(Human Protein Atlas).  The ``channel_adaptive_dino_vitl16`` checkpoint
processes one channel at a time through a single-channel ViT-L/16 stem; the
wrapper reshapes ``(B, C, H, W) -> (B*C, 1, H, W)``, runs the backbone, and
mean-pools the cls token across channels to produce a fixed-dimension
embedding regardless of the input channel count.

Weights are loaded from a local ``.pth`` state_dict; nothing is fetched
from the network.  See
``/hpc/projects/organelle_phenotyping/models/CELL-DINO/model_weights/weights/``
for the published checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from viscy_models.foundation._dinov2_vit import vit_large


class CellDinoModel(nn.Module):
    """Wrap CELL-DINO (channel-adaptive ViT-L/16) for microscopy embeddings.

    The model accepts raw dataloader tensors ``(B, C, D, H, W)`` directly in
    :meth:`forward` — preprocessing is applied inline.  Z-slice selection is
    **not** handled here — configure ``z_range`` on the dataloader so it
    delivers the correct focal plane.

    Parameters
    ----------
    weights_path : str
        Path to the local ``.pth`` state_dict.  The default
        ``channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth`` is a
        single-channel ViT-L/16 trained at 224 px with ``init_values=1.0``
        and ``block_chunks=4``.
    img_size : int
        Spatial size after :meth:`preprocess_2d`, by default ``224``.
    patch_size : int
        Patch size for the ViT, by default ``16``.
    freeze : bool
        If ``True`` (default), all backbone parameters are frozen and the
        model is kept in eval mode.
    projection : nn.Module or None
        Optional trainable projection head applied to backbone features.
        When provided, :meth:`forward` returns ``(features, projection(features))``.
        When ``None`` (default), returns ``(features, features)``.
    """

    def __init__(
        self,
        weights_path: str,
        img_size: int = 224,
        patch_size: int = 16,
        freeze: bool = True,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.projection = projection
        self.target_size = (img_size, img_size)

        self.model = vit_large(
            patch_size=patch_size,
            in_chans=1,
            channel_adaptive=True,
            img_size=img_size,
            init_values=1.0,
            block_chunks=4,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"CELL-DINO state_dict mismatch — missing={missing}, unexpected={unexpected}")

        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "CellDinoModel":
        """Override train to keep backbone in eval when frozen."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def preprocess_2d(self, x: Tensor) -> Tensor:
        """Convert a raw dataloader tensor to CELL-DINO input.

        Squeezes singleton Z (or takes the middle slice if Z>1), resizes to
        ``self.target_size``, and per-image min/max scales each
        ``(B*C)`` map to ``[0, 1]``.  No ImageNet mean/std is applied —
        CELL-DINO uses simple ``[0,1]`` normalization in training.

        Parameters
        ----------
        x : Tensor
            ``(B, C, D, H, W)`` or ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            ``(B, C, H_target, W_target)`` ready for :meth:`forward`.  The
            wrapper reshapes ``(B, C, ...) -> (B*C, 1, ...)`` inside
            :meth:`forward`, so this method preserves the channel axis.
        """
        if x.ndim == 5:
            if x.shape[2] == 1:
                x = x[:, :, 0]
            else:
                x = x[:, :, x.shape[2] // 2]

        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min).clamp(min=1e-8)
        return x.view(b, c, h, w)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run CELL-DINO on an image batch and mean-pool over channels.

        Preprocessing is applied inline, so raw dataloader tensors
        ``(B, C, D, H, W)`` or ``(B, C, H, W)`` can be passed directly.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(features, projections)`` where features are the
            channel-mean-pooled cls token of shape ``(B, 1024)``.  If
            ``projection`` was provided at init, projections are
            ``self.projection(features)``; otherwise both elements are the
            same features tensor.
        """
        x = self.preprocess_2d(x)
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        cls = self.model(x)
        cls = cls.view(b, c, -1).mean(dim=1)
        if self.projection is not None:
            return (cls, self.projection(cls))
        return (cls, cls)
