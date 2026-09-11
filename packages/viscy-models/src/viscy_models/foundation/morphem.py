"""MorphEm foundation model wrapper for frozen feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MorphEmModel(nn.Module):
    """Wrap CaicedoLab/MorphEm (DINO-pretrained ViT-S/16 on microscopy).

    MorphEm is a single-channel ViT: each microscopy channel is encoded
    independently and the CLS tokens are reduced to a fixed embedding.
    :meth:`preprocess_2d` handles Z-squeeze, resize to 224, and per-image
    per-channel spatial z-score (the published ``PerImageNormalize`` recipe).

    Parameters
    ----------
    model_name : str
        HuggingFace id (``"CaicedoLab/MorphEm"``) or a local snapshot dir.
    revision : str | None
        Hub commit SHA (or tag) to load. MorphEm ships its ViT as
        ``trust_remote_code`` Python, so an unpinned id would execute
        whatever is at the repository head on a cache miss. Ignored for a
        local snapshot dir.
    img_size : int
        Spatial size inputs are resized to, by default 224.
    freeze : bool
        If ``True`` (default), backbone params are frozen and kept in eval mode.
    projection : nn.Module | None
        Optional projection head applied to the pooled embedding.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        img_size: int = 224,
        freeze: bool = True,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()

        import transformers
        from transformers import AutoModel

        # transformers 5.x compatibility shim. MorphEm's
        # trust_remote_code VisionTransformer (authored against transformers 4.x)
        # never sets `all_tied_weights_keys`, which 5.x's meta-device loader
        # (_move_missing_keys_from_meta_to_device) requires -> AttributeError on load.
        # Provide a no-tie class-level default; models whose tie_weights() runs set
        # their own instance attr, which shadows this, so other loads are unaffected.
        # Idempotent: only set when absent. Verified to load CaicedoLab/MorphEm on 5.9.0.
        if not isinstance(getattr(transformers.PreTrainedModel, "all_tied_weights_keys", None), dict):
            transformers.PreTrainedModel.all_tied_weights_keys = {}

        self.model = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        self.target_size = (img_size, img_size)
        self.projection = projection
        # Expose embed dim for foundation-wrapper parity (dynaclr foundation_engine
        # may read it pre-forward). Mirror the katamari reference's 3-tier
        # `_infer_embed_dim` (embed_dim -> num_features -> patch_embed.proj.out_channels);
        # raise loud if none resolve rather than silently defaulting.
        # Verified at load: CaicedoLab/MorphEm self-reports embed_dim == num_features == 384.
        self.embed_dim = self._infer_embed_dim(self.model)

        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        # Single-channel contract: MorphEm is a 1-channel ViT (CHAMMI feeds one
        # channel at a time). Assert at load so a 3-channel patch_embed surfaces
        # immediately instead of erroring deep in forward().
        proj = getattr(getattr(self.model, "patch_embed", None), "proj", None)
        in_ch = getattr(proj, "in_channels", None)
        if in_ch is not None and in_ch != 1:
            raise ValueError(
                f"MorphEm expected a single-channel patch_embed (in_channels=1), got {in_ch}. "
                "The single-channel feed in forward() would error; revisit channel handling."
            )

    @staticmethod
    def _infer_embed_dim(model: nn.Module) -> int:
        """Resolve the backbone embedding dimension from common ViT attributes."""
        for attr in ("embed_dim", "num_features"):
            value = getattr(model, attr, None)
            if value is not None:
                return int(value)
        proj = getattr(getattr(model, "patch_embed", None), "proj", None)
        out_channels = getattr(proj, "out_channels", None)
        if out_channels is not None:
            return int(out_channels)
        raise ValueError("Unable to infer embed_dim from the MorphEm backbone.")

    def train(self, mode: bool = True) -> "MorphEmModel":
        """Keep the backbone in eval when frozen."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def preprocess_2d(self, x: Tensor) -> Tensor:
        """Convert a raw dataloader tensor to MorphEm input.

        Squeezes a singleton Z (middle slice if Z>1), applies per-image
        per-channel spatial z-score (zero mean, unit std), THEN resizes to
        ``target_size`` — matching the published recipe ordering
        ``Compose([PerImageNormalize(), Resize(224)])`` (normalize before
        resize; reversing the order shifts the post-interp mean/scale).

        Parameters
        ----------
        x : Tensor
            ``(B, C, D, H, W)`` or ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            ``(B, C, target, target)`` float ready for :meth:`forward`.
        """
        if x.ndim == 5:
            x = x[:, :, 0] if x.shape[2] == 1 else x[:, :, x.shape[2] // 2]
        x = x.to(torch.float32)
        # PerImageNormalize: per-image per-channel spatial z-score. Uses the
        # (x-m)/(std+eps) form matching CellDinoModel.preprocess_2d (the VisCy
        # self_normalize convention) — numerically equivalent to the reference
        # InstanceNorm2d's (x-m)/sqrt(var+eps) for any non-degenerate crop.
        m = x.mean(dim=(-2, -1), keepdim=True)
        s = x.std(dim=(-2, -1), unbiased=False, keepdim=True)
        x = (x - m) / (s + 1e-7)
        # Resize AFTER normalize (published Compose order). antialias=True matches
        # the reference v2.Resize(antialias=True); inert on upscale, correct on downscale.
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False, antialias=True)
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a (preprocessed) image batch to a channel-mean CLS embedding.

        Each channel is encoded independently (MorphEm is single-channel),
        and the CLS tokens are mean-pooled over channels to a fixed ``(B, D)``.

        Parameters
        ----------
        x : Tensor
            ``(B, C, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(features, projection(features) or features)``, each ``(B, D)``.
        """
        b, c = x.shape[0], x.shape[1]
        x = x.reshape(b * c, 1, x.shape[-2], x.shape[-1])
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]  # (B*C, D)
        cls = cls.reshape(b, c, -1).mean(dim=1)  # (B, D)
        if self.projection is not None:
            return cls, self.projection(cls)
        return cls, cls
