"""Temporal integration of per-frame foundation-model embeddings, read out by a regression head and/or a UNet.

    phase movie [p_t] --E--> [e_t] --T--> [ê_t] --f--> per-frame reporter levels      (regression)
                                              └--g(p_t, ê_t)--> per-frame reporter image (virtual staining)

E is a frozen foundation model (:class:`~dynaformer.models.foundation.FoundationFrameEncoder`). Its embeddings are
normally read from a cache written by :mod:`dynaformer.embed`, which runs the same module, so training never pays
for E; an ``encoder`` can still be attached to run the model on raw phase movies. An encoder trained with the model
(``trainable``, e.g. a ViT from scratch) has no cache and always runs on the frames. T is the bidirectional RoPE
transformer (:class:`~dynaformer.models.temporal.RoPETransformerEncoder`), so a model trained on clips runs whole
tracks. ``n_layers=0`` leaves T as a LayerNorm: every ê_t then depends on its own frame only, the single-frame
ablation.
"""

from torch import Tensor, nn

from dynaformer.models.conditioned_unet import ConditionedUNet
from dynaformer.models.foundation import FoundationFrameEncoder
from dynaformer.models.temporal import RoPETransformerEncoder

__all__ = ["MLPRegressionHead", "TemporalEmbeddingModel"]


class MLPRegressionHead(nn.Module):
    """Small MLP f mapping a temporal embedding ê_t to per-frame values.

    Parameters
    ----------
    in_dims : int
        Width of ê_t.
    out_dims : int
        Number of predicted values, by default 2 (Geminin-mKO and PIP-mVenus medians).
    hidden_dims : int
        Hidden width, by default 256.
    dropout : float
        Dropout after the hidden layer, by default 0.1.
    """

    def __init__(self, in_dims: int, out_dims: int = 2, hidden_dims: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dims, out_dims)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(..., in_dims)`` to ``(..., out_dims)``."""
        return self.net(x)


class TemporalEmbeddingModel(nn.Module):
    """Temporal transformer over per-frame embeddings with a regression head f and/or a stain UNet g.

    Parameters
    ----------
    embed_dim : int
        Width of the per-frame embeddings e_t.
    temporal : RoPETransformerEncoder | None
        T. ``None`` only for a plain per-frame UNet that reads no embeddings (the unconditioned baseline).
    regression_head : nn.Module | None
        f, applied to every ê_t.
    stain_head : ConditionedUNet | None
        g, applied to the frames that have images. Conditioned on ê_t when built with ``cond_dim > 0``.
    input_dropout : float
        Dropout on the projected embeddings, by default 0.1.
    encoder : FoundationFrameEncoder | None
        E, to run on raw phase movies, frozen or trained with the model. ``None`` when the embeddings come from a
        cache, by default ``None``.
    """

    def __init__(
        self,
        embed_dim: int,
        temporal: RoPETransformerEncoder | None = None,
        regression_head: nn.Module | None = None,
        stain_head: ConditionedUNet | None = None,
        input_dropout: float = 0.1,
        encoder: FoundationFrameEncoder | None = None,
    ) -> None:
        super().__init__()
        if regression_head is None and stain_head is None:
            raise ValueError("TemporalEmbeddingModel needs a regression head, a stain head, or both.")
        conditioned = stain_head is not None and stain_head.cond_dim > 0
        if temporal is None and (regression_head is not None or conditioned):
            raise ValueError("The regression head and a conditioned stain head read ê_t, which needs a temporal model.")
        if temporal is not None and not (regression_head is not None or conditioned):
            raise ValueError("Nothing reads the temporal model's output; drop it or condition the stain head.")
        if conditioned and stain_head.cond_dim != temporal.d_model:
            raise ValueError(f"Stain head cond_dim {stain_head.cond_dim} != temporal d_model {temporal.d_model}.")
        if encoder is not None and encoder.embed_dim != embed_dim:
            raise ValueError(f"Encoder embed_dim {encoder.embed_dim} != model embed_dim {embed_dim}.")
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.temporal = temporal
        self.input_proj = (
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, temporal.d_model), nn.Dropout(input_dropout))
            if temporal is not None
            else None
        )
        self.regression_head = regression_head
        self.stain_head = stain_head
        self.conditioned = conditioned

    def embed(self, frames: Tensor) -> Tensor:
        """Run E on a ``(B, T, 1, Y, X)`` z-scored phase movie, returning ``(B, T, embed_dim)``."""
        if self.encoder is None:
            raise ValueError("No encoder attached; pass cached embeddings instead.")
        b, t = frames.shape[:2]
        return self.encoder(frames.flatten(0, 1)).view(b, t, -1)

    def forward(
        self,
        embeddings: Tensor | None = None,
        images: Tensor | None = None,
        image_index: Tensor | None = None,
        frames: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Integrate a clip of embeddings over time and read it out.

        Parameters
        ----------
        embeddings : Tensor | None
            ``(B, T, embed_dim)`` cached embeddings e_t. Computed from ``frames`` with the attached encoder when
            ``None``.
        images : Tensor | None
            ``(M, 1, Y, X)`` phase images of the frames the stain head runs on.
        image_index : Tensor | None
            ``(M,)`` flat indices of those frames into ``B * T``, pairing each image with its ê_t.
        frames : Tensor | None
            ``(B, T, 1, Y, X)`` phase movie, only used without cached embeddings.

        Returns
        -------
        dict[str, Tensor]
            ``tokens`` ê_t ``(B, T, d_model)`` and ``regression`` ``(B, T, out_dims)`` when there is a temporal model
            and a regression head; ``stain`` ``(M, out_channels, Y, X)`` when there is a stain head and images.
        """
        out: dict[str, Tensor] = {}
        if self.temporal is not None:
            if embeddings is None:
                embeddings = self.embed(frames)
            if embeddings.shape[-1] != self.embed_dim:
                raise ValueError(f"Embeddings are {embeddings.shape[-1]} wide, the model expects {self.embed_dim}.")
            out["tokens"] = self.temporal(self.input_proj(embeddings))
            if self.regression_head is not None:
                out["regression"] = self.regression_head(out["tokens"])
        if self.stain_head is not None and images is not None:
            cond = out["tokens"].flatten(0, 1)[image_index] if self.conditioned else None
            out["stain"] = self.stain_head(images, cond)
        return out
