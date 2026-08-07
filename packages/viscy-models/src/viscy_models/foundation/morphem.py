"""MorphEm foundation model wrapper for frozen feature extraction.

MorphEm is a ViT-S/16 pretrained with DINO on microscopy images
(``CaicedoLab/MorphEm`` on HuggingFace).  Like CELL-DINO it processes one
channel at a time through a single-channel ViT; the wrapper reshapes
``(B, C, H, W) -> (B*C, 1, H, W)``, runs the backbone, and mean-pools the
cls token across channels to produce a fixed-dimension embedding regardless
of the input channel count.

Weights are fetched from HuggingFace via ``AutoModel.from_pretrained`` with
``trust_remote_code=True`` (the repo ships custom preprocessing classes).
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MorphEmModel(nn.Module):
    """Wrap MorphEm (DINO ViT-S) for microscopy embeddings.

    The model accepts raw dataloader tensors ``(B, C, D, H, W)`` directly in
    :meth:`forward` — preprocessing is applied inline.  Z-slice selection is
    **not** handled here — configure ``z_range`` on the dataloader so it
    delivers the correct focal plane.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, by default ``"CaicedoLab/MorphEm"``.
    img_size : int
        Spatial size after :meth:`preprocess_2d`, by default ``224``.
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
        model_name: str = "CaicedoLab/MorphEm",
        img_size: int = 224,
        freeze: bool = True,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()

        import transformers.modeling_utils as _mu
        from transformers import AutoModel

        # MorphEm's `trust_remote_code` VisionTransformer targets an older
        # transformers API and never sets `all_tied_weights_keys`, which the
        # transformers>=5 `from_pretrained` finalizer reads. Provide a
        # class-level empty default so the (untied) remote model loads.
        if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
            _mu.PreTrainedModel.all_tied_weights_keys = {}

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.target_size = (img_size, img_size)
        self.projection = projection

        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "MorphEmModel":
        """Override train to keep backbone in eval when frozen."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def preprocess_2d(self, x: Tensor) -> Tensor:
        """Convert a raw dataloader tensor to MorphEm input.

        Squeezes singleton Z (or takes the middle slice if Z>1), resizes
        to ``self.target_size``, then applies per-image per-channel
        spatial z-score (``PerImageNormalize``): for each ``(B, C)`` map,
        subtract the spatial mean over ``(H, W)`` and divide by the spatial
        std.  This matches the MorphEm inference recipe and makes the input
        statistics independent of upstream normalization.

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

        m = x.mean(dim=(-2, -1), keepdim=True)
        s = x.std(dim=(-2, -1), unbiased=False, keepdim=True)
        return (x - m) / (s + 1e-7)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run MorphEm on an image batch and mean-pool over channels.

        Preprocessing is applied inline, so raw dataloader tensors
        ``(B, C, D, H, W)`` or ``(B, C, H, W)`` can be passed directly.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(features, projections)`` where features are the
            channel-mean-pooled cls token of shape ``(B, 384)``.  If
            ``projection`` was provided at init, projections are
            ``self.projection(features)``; otherwise both elements are the
            same features tensor.
        """
        x = self.preprocess_2d(x)
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        # forward_features returns the DINOv2 dict; the cls token is the
        # normalized class embedding. (model(x) would instead return a
        # BaseModelOutput of all tokens.)
        cls = self.model.forward_features(x)["x_norm_clstoken"]
        cls = cls.view(b, c, -1).mean(dim=1)
        if self.projection is not None:
            return (cls, self.projection(cls))
        return (cls, cls)
