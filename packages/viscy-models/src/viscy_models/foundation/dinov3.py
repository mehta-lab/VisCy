"""DINOv3 foundation model wrapper for frozen feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DINOv3Model(nn.Module):
    """Wrap a HuggingFace DINOv3 vision model for microscopy images.

    The model expects preprocessed ``(B, 3, H, W)`` input in its
    :meth:`forward`.  Use :meth:`preprocess_2d` to convert raw dataloader
    output (e.g. ``(B, C, D, H, W)`` from ``TripletDataModule``) into the
    expected format (channel repeat, resize, ImageNet normalisation).

    Z-slice selection is **not** handled here — configure ``z_range`` on the
    dataloader so it delivers the correct focal plane (see
    ``get_z_range()`` in the evaluation utilities).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g.
        ``"facebook/dinov3-small-imagenet1k-1-layer"``.
    freeze : bool
        If ``True`` (default), all backbone parameters are frozen and the
        model is kept in eval mode.
    """

    def __init__(self, model_name: str, freeze: bool = True) -> None:
        super().__init__()

        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)

        image_mean = torch.tensor(processor.image_mean, dtype=torch.float32)
        image_std = torch.tensor(processor.image_std, dtype=torch.float32)
        self.register_buffer("image_mean", image_mean.view(1, 3, 1, 1))
        self.register_buffer("image_std", image_std.view(1, 3, 1, 1))

        size_cfg = processor.size
        self.target_size = (
            (size_cfg["height"], size_cfg["width"])
            if "height" in size_cfg
            else (size_cfg["shortest_edge"], size_cfg["shortest_edge"])
        )

        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "DINOv3Model":
        """Override train to keep backbone in eval when frozen."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def preprocess_2d(self, x: Tensor) -> Tensor:
        """Convert a raw dataloader tensor to a normalised RGB image.

        Handles squeezing a singleton Z dim, repeating/trimming channels
        to 3, resizing to the model's expected spatial size, rescaling to
        [0, 1], and applying ImageNet normalisation.

        Z-slice selection should happen upstream (e.g. via ``z_range`` in
        ``TripletDataModule``).  If ``D > 1`` is passed, the middle slice
        is taken as a fallback.

        Parameters
        ----------
        x : Tensor
            ``(B, C, D, H, W)`` or ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            ``(B, 3, H_target, W_target)`` ready for :meth:`forward`.
        """
        if x.ndim == 5:
            if x.shape[2] == 1:
                x = x[:, :, 0]
            else:
                x = x[:, :, x.shape[2] // 2]

        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, :1]], dim=1)
        elif x.shape[1] > 3:
            x = x[:, :3]

        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        x_min = x.flatten(1).min(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
        x_max = x.flatten(1).max(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
        scale = (x_max - x_min).clamp(min=1e-8)
        x = (x - x_min) / scale

        x = (x - self.image_mean) / self.image_std
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run the DINOv3 backbone on a preprocessed image batch.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, 3, H, W)`` — already preprocessed
            (resized, normalised).  Call :meth:`preprocess` first when
            working with raw 3-D volumes.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(features, features)`` — both are the pooler output of shape
            ``(B, hidden_dim)``.  No separate projection head is used.
        """
        features = self.model(pixel_values=x).pooler_output
        return (features, features)
