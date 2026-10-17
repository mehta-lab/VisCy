"""Composition of a per-frame encoder, a temporal module, a decoder and heads."""

import torch
from torch import Tensor, nn

from dynaformer.models.encoders import FrameFeatures

__all__ = ["SpatioTemporalModel"]


class SpatioTemporalModel(nn.Module):
    """Decoupled spatio-temporal model over a movie of tracked frames.

    Time is folded into the batch for the encoder and the decoder, so both stay
    per-frame; the temporal module is the only component that mixes across frames.
    The temporally-mixed token is what reaches the decoder as conditioning and what
    the auxiliary heads read, so temporal context flows into every output.

    Parameters
    ----------
    encoder : nn.Module
        Per-frame encoder returning :class:`~dynaformer.models.encoders.FrameFeatures`.
        Must expose ``token_dim``.
    temporal : nn.Module
        Sequence model over ``(B, T, d_model)``. Must expose ``d_model``.
    decoder : nn.Module | None
        Image decoder taking ``(spatial, cond)``. ``None`` trains the heads only,
        which is a much cheaper model but loses the dense per-pixel supervision
        that makes the shared representation strong.
    heads : dict[str, nn.Module] | None
        Auxiliary per-frame heads keyed by name, applied to the temporal tokens.
        Each head maps ``(N, d_model)`` to ``(N, out_dim)``.
    condition_decoder : bool
        Pass the temporal token to the decoder as conditioning, by default ``True``.
        Requires the decoder to have been built with a matching ``cond_dim``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module | None = None,
        heads: dict[str, nn.Module] | None = None,
        condition_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal
        self.decoder = decoder
        self.heads = nn.ModuleDict(heads or {})
        self.condition_decoder = condition_decoder
        if decoder is None and not self.heads:
            raise ValueError("SpatioTemporalModel needs a decoder, at least one head, or both.")
        if condition_decoder and decoder is not None and getattr(decoder, "cond_proj", None) is None:
            raise ValueError(
                "condition_decoder=True but the decoder was built without conditioning; "
                "set the decoder's cond_dim to the temporal module's d_model."
            )
        self.token_proj = (
            nn.Identity() if encoder.token_dim == temporal.d_model else nn.Linear(encoder.token_dim, temporal.d_model)
        )

    def encode(self, x: Tensor) -> tuple[FrameFeatures, Tensor, int]:
        """Encode every frame and mix the frame tokens along time.

        Parameters
        ----------
        x : Tensor
            ``(B, T, C, Z, Y, X)``.

        Returns
        -------
        tuple[FrameFeatures, Tensor, int]
            Per-frame features with time folded into the batch, temporal tokens of
            shape ``(B, T, d_model)``, and the number of frames ``T``.
        """
        if x.ndim != 6:
            raise ValueError(f"Expected a (B, T, C, Z, Y, X) movie, got shape {tuple(x.shape)}.")
        b, t = x.shape[:2]
        features: FrameFeatures = self.encoder(x.reshape(b * t, *x.shape[2:]))
        tokens = self.temporal(self.token_proj(features.token).view(b, t, -1))
        return features, tokens, t

    def forward(self, x: Tensor) -> tuple[Tensor | None, dict[str, Tensor]]:
        """Run the full model on a movie.

        Parameters
        ----------
        x : Tensor
            ``(B, T, C, Z, Y, X)``.

        Returns
        -------
        tuple[Tensor | None, dict[str, Tensor]]
            The decoded movie of shape ``(B, T, C_out, Z_out, Y, X)`` (``None``
            when no decoder is configured), and per-head outputs of shape
            ``(B, T, out_dim)``.
        """
        b = x.shape[0]
        features, tokens, t = self.encode(x)
        flat_tokens = tokens.reshape(b * t, -1)

        head_outputs = {name: head(flat_tokens).view(b, t, -1) for name, head in self.heads.items()}

        image = None
        if self.decoder is not None:
            cond = flat_tokens if self.condition_decoder else None
            frames = self.decoder(features.spatial, cond=cond)
            image = frames.view(b, t, *frames.shape[1:])
        return image, head_outputs

    @torch.no_grad()
    def predict_track(self, x: Tensor) -> tuple[Tensor | None, dict[str, Tensor]]:
        """Run a whole track in one forward pass.

        Identical to :meth:`forward`; it exists to make the intent explicit. The
        temporal module carries no length-dependent parameters, so feeding a full
        track rather than tiled windows is valid and removes window-boundary
        discontinuities from the predicted trajectory.

        Parameters
        ----------
        x : Tensor
            ``(B, T, C, Z, Y, X)`` with ``T`` the full track length.

        Returns
        -------
        tuple[Tensor | None, dict[str, Tensor]]
            Same as :meth:`forward`.
        """
        return self.forward(x)
