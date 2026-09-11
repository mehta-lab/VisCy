"""Tests for the MorphEm foundation wrapper.

``transformers.AutoModel`` is stubbed so the test never touches the network or
the HF cache; it exercises the real :class:`MorphEmModel` preprocessing and
channel-mean forward contract against a fake ViT-S backbone.
"""

import pytest
import torch

from viscy_models.foundation import MorphEmModel

_EMBED_DIM = 384
_N_PATCHES = 196  # (224 / 16) ** 2


class _StubBackbone(torch.nn.Module):
    """Minimal MorphEm stand-in: single-channel patch embed + forward_features."""

    def __init__(self, embed_dim: int = _EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Conv2d(1, embed_dim, kernel_size=16, stride=16)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = x.shape[0]
        # Make the CLS token depend on the input so embeddings are non-constant.
        cls = x.flatten(1).mean(dim=1, keepdim=True).expand(n, self.embed_dim).clone()
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": torch.zeros(n, _N_PATCHES, self.embed_dim),
        }


@pytest.fixture
def captured_load(monkeypatch):
    """Stub ``AutoModel.from_pretrained`` and capture its call args."""
    captured: dict = {}

    def _fake_from_pretrained(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _StubBackbone()

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", _fake_from_pretrained)
    return captured


def test_load_passes_trust_remote_code(captured_load):
    """The wrapper loads via trust_remote_code and infers the embed dim."""
    model = MorphEmModel(model_name="CaicedoLab/MorphEm")
    assert captured_load["name"] == "CaicedoLab/MorphEm"
    assert captured_load["kwargs"]["trust_remote_code"] is True
    assert captured_load["kwargs"]["revision"] is None
    assert model.embed_dim == _EMBED_DIM


def test_load_forwards_pinned_revision(captured_load):
    """A pinned Hub commit reaches ``from_pretrained`` so the remote code is frozen."""
    MorphEmModel(model_name="CaicedoLab/MorphEm", revision="0e8d58787421f83f975634d72420d85c5dfc9c2c")
    assert captured_load["kwargs"]["revision"] == "0e8d58787421f83f975634d72420d85c5dfc9c2c"
    assert captured_load["kwargs"]["trust_remote_code"] is True


def test_preprocess_2d_normalizes_and_resizes(captured_load):
    """5-D input is Z-squeezed, per-image z-scored, then resized to 224."""
    model = MorphEmModel(model_name="CaicedoLab/MorphEm")
    x = torch.randn(2, 1, 1, 40, 40)
    out = model.preprocess_2d(x)
    assert out.shape == (2, 1, 224, 224)
    # Per-image z-score happens before resize; upscale bilinear keeps mean ~0.
    per_image_mean = out.mean(dim=(-2, -1))
    assert torch.all(per_image_mean.abs() < 0.05)
    assert torch.all(out.std(dim=(-2, -1)) > 0.1)


def test_forward_returns_features_projection_tuple(captured_load):
    """Forward on a single-channel batch returns a (features, proj) (B, D) tuple."""
    model = MorphEmModel(model_name="CaicedoLab/MorphEm")
    x = torch.randn(2, 1, 224, 224)
    features, projection = model(x)
    assert features.shape == (2, _EMBED_DIM)
    assert projection.shape == (2, _EMBED_DIM)
    assert torch.equal(features, projection)  # no projection head -> identity


def test_forward_channel_mean(captured_load):
    """Multi-channel input is encoded per-channel and mean-pooled to (B, D)."""
    model = MorphEmModel(model_name="CaicedoLab/MorphEm")
    x = torch.randn(2, 3, 224, 224)
    features, _ = model(x)
    assert features.shape == (2, _EMBED_DIM)


def test_single_channel_assert(monkeypatch):
    """A 3-channel patch_embed is rejected at load time."""

    class _ThreeChannel(_StubBackbone):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed.proj = torch.nn.Conv2d(3, _EMBED_DIM, kernel_size=16, stride=16)

    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda name, **kwargs: _ThreeChannel(),
    )
    with pytest.raises(ValueError, match="single-channel"):
        MorphEmModel(model_name="CaicedoLab/MorphEm")
