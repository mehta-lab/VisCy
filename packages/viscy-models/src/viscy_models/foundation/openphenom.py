"""OpenPhenom foundation model wrapper for frozen feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OpenPhenomModel(nn.Module):
    """Wrap Recursion's OpenPhenom CA-MAE ViT-S/16 for microscopy images.

    OpenPhenom accepts 1–11 channel uint8 input at 256×256 and normalises
    internally.  :meth:`preprocess_2d` handles Z-squeeze, resize, and
    float→uint8 conversion.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"recursionpharma/OpenPhenom"``.
    freeze : bool
        If ``True`` (default), all backbone parameters are frozen and the
        model is kept in eval mode.
    """

    def __init__(self, model_name: str, freeze: bool = True) -> None:
        super().__init__()

        from huggingface_hub import PyTorchModelHubMixin  # noqa: F401
        from open_phenom import MAEModel

        self.model = MAEModel.from_pretrained(model_name)
        self.model.return_channelwise_embeddings = False
        self.target_size = (256, 256)

        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "OpenPhenomModel":
        """Override train to keep backbone in eval when frozen."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def preprocess_2d(self, x: Tensor) -> Tensor:
        """Convert a raw dataloader tensor to uint8 input for OpenPhenom.

        Handles squeezing a singleton Z dim, resizing to 256×256, and
        rescaling float values to [0, 255] uint8 (OpenPhenom normalises
        internally).

        Unlike DINOv3, no channel manipulation is needed — OpenPhenom
        accepts 1–11 channels natively.

        Parameters
        ----------
        x : Tensor
            ``(B, C, D, H, W)`` or ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            ``(B, C, 256, 256)`` uint8 tensor ready for :meth:`forward`.
        """
        if x.ndim == 5:
            if x.shape[2] == 1:
                x = x[:, :, 0]
            else:
                x = x[:, :, x.shape[2] // 2]

        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        x_min = x.flatten(1).min(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
        x_max = x.flatten(1).max(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
        scale = (x_max - x_min).clamp(min=1e-8)
        x = (x - x_min) / scale * 255.0

        return x.to(torch.uint8)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run the OpenPhenom backbone on a preprocessed image batch.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, 256, 256)`` uint8.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(features, features)`` — both are the embedding of shape
            ``(B, 384)``.  No separate projection head is used.
        """
        features = self.model.predict(x)
        return (features, features)
