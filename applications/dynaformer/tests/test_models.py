"""Model composition, conditioning and length-agnosticism."""

import pytest
import torch

from dynaformer.models import (
    MAEImageDecoder,
    RoPETransformerEncoder,
    SpatioTemporalModel,
    UNeXt2FrameEncoder,
    UNeXt2ImageDecoder,
)
from viscy_models.components.heads import RegressionHead

YX = 64
D_MODEL = 64


def build_unext2_model(condition_decoder: bool = True, with_head: bool = True) -> SpatioTemporalModel:
    """Build a small UNeXt2 encoder/decoder model on CPU."""
    encoder = UNeXt2FrameEncoder(
        in_channels=1,
        in_stack_depth=1,
        stem_kernel_size=(1, 4, 4),
        yx_patch_size=(YX, YX),
        token_dim=D_MODEL,
        token_channels=8,
    )
    decoder = UNeXt2ImageDecoder(
        encoder.feature_channels,
        out_channels=2,
        out_stack_depth=1,
        stem_stride=encoder.stem_stride,
        cond_dim=D_MODEL if condition_decoder else 0,
    )
    heads = {"scalar": RegressionHead("scalar", "scalar_target", in_dims=D_MODEL, out_dims=2)} if with_head else None
    return SpatioTemporalModel(
        encoder=encoder,
        temporal=RoPETransformerEncoder(d_model=D_MODEL, n_heads=4, n_layers=2),
        decoder=decoder,
        heads=heads,
        condition_decoder=condition_decoder,
    )


def test_unext2_model_shapes():
    """Image and head outputs keep the batch and time axes."""
    model = build_unext2_model()
    image, heads = model(torch.rand(2, 3, 1, 1, YX, YX))
    assert image.shape == (2, 3, 2, 1, YX, YX)
    assert heads["scalar"].shape == (2, 3, 2)


@pytest.mark.parametrize("frames", [1, 2, 9])
def test_rope_accepts_any_clip_length(frames):
    """The same weights run clips of any length, which whole-track inference needs."""
    model = build_unext2_model()
    image, heads = model(torch.rand(1, frames, 1, 1, YX, YX))
    assert image.shape == (1, frames, 2, 1, YX, YX)
    assert heads["scalar"].shape == (1, frames, 2)


def test_temporal_conditioning_is_identity_at_initialization():
    """A freshly built conditioned decoder matches the unconditioned one exactly.

    The conditioning projection is zero-initialized so training starts from a
    working per-frame model and only adds temporal corrections. If this drifts,
    the conditioned model no longer starts where the per-frame model does.
    """
    model = build_unext2_model(condition_decoder=True).eval()
    movie = torch.rand(2, 4, 1, 1, YX, YX)
    with torch.no_grad():
        conditioned, _ = model(movie)
        model.condition_decoder = False
        unconditioned, _ = model(movie)
    assert torch.equal(conditioned, unconditioned)


def test_temporal_conditioning_changes_output_once_trained():
    """After the conditioning weights move, the decoder actually uses them."""
    model = build_unext2_model(condition_decoder=True).eval()
    with torch.no_grad():
        model.decoder.cond_proj.proj.weight.normal_(std=0.1)
        model.decoder.cond_proj.proj.bias.normal_(std=0.1)
        movie = torch.rand(2, 4, 1, 1, YX, YX)
        conditioned, _ = model(movie)
        model.condition_decoder = False
        unconditioned, _ = model(movie)
    assert not torch.allclose(conditioned, unconditioned)


def test_conditioned_model_rejects_unconditioned_decoder():
    """Asking for conditioning without a conditioning projection is an error."""
    encoder = UNeXt2FrameEncoder(
        in_channels=1, in_stack_depth=1, stem_kernel_size=(1, 4, 4), yx_patch_size=(YX, YX), token_dim=D_MODEL
    )
    decoder = UNeXt2ImageDecoder(encoder.feature_channels, out_channels=2, stem_stride=encoder.stem_stride)
    with pytest.raises(ValueError, match="without conditioning"):
        SpatioTemporalModel(
            encoder=encoder,
            temporal=RoPETransformerEncoder(d_model=D_MODEL, n_heads=4, n_layers=1),
            decoder=decoder,
            condition_decoder=True,
        )


def test_mae_decoder_shapes_and_zero_init():
    """The transformer decoder unpatchifies to the requested size and starts unconditioned."""
    decoder = MAEImageDecoder(
        embed_dim=32, out_channels=2, grid_size=4, patch_size=8, out_size=(YX, YX), cond_dim=16
    ).eval()
    tokens = torch.rand(3, 16, 32)
    with torch.no_grad():
        conditioned = decoder(tokens, cond=torch.rand(3, 16))
        unconditioned = decoder(tokens, cond=None)
    assert conditioned.shape == (3, 2, 1, YX, YX)
    assert torch.equal(conditioned, unconditioned)


def test_mae_decoder_native_grid_size():
    """Without a target size the reconstruction stays on the patch grid."""
    decoder = MAEImageDecoder(embed_dim=32, out_channels=1, grid_size=4, patch_size=8)
    assert decoder(torch.rand(2, 16, 32)).shape == (2, 1, 1, 32, 32)


def test_encoder_rejects_mismatched_patch_size():
    """A frame size other than the configured one is reported, not silently reshaped."""
    encoder = UNeXt2FrameEncoder(
        in_channels=1, in_stack_depth=1, stem_kernel_size=(1, 4, 4), yx_patch_size=(YX, YX), token_dim=D_MODEL
    )
    with pytest.raises(ValueError, match="yx_patch_size"):
        encoder(torch.rand(1, 1, 1, YX * 2, YX * 2))


def test_model_requires_a_decoder_or_a_head():
    """A model with neither output is rejected rather than silently doing nothing."""
    encoder = UNeXt2FrameEncoder(
        in_channels=1, in_stack_depth=1, stem_kernel_size=(1, 4, 4), yx_patch_size=(YX, YX), token_dim=D_MODEL
    )
    with pytest.raises(ValueError, match="decoder, at least one head"):
        SpatioTemporalModel(encoder, RoPETransformerEncoder(d_model=D_MODEL, n_heads=4, n_layers=1))


@pytest.mark.slow
def test_morphem_encoder_downloads_and_returns_tokens():
    """The pretrained encoder yields a class token and a patch grid."""
    from dynaformer.models import MorphEmFrameEncoder

    encoder = MorphEmFrameEncoder(freeze=True)
    features = encoder(torch.rand(2, 1, 1, 256, 256))
    assert features.token.shape == (2, encoder.token_dim)
    assert features.spatial.shape == (2, encoder.grid_size**2, encoder.spatial_dim)
