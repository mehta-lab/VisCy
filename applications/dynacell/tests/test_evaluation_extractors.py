"""Unit tests for the eval-pipeline feature extractors in ``dynacell.evaluation.utils``."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from dynacell.evaluation import utils as eval_utils  # noqa: E402


def _stub_processor_and_model(monkeypatch: pytest.MonkeyPatch):
    """Replace ``AutoImageProcessor`` / ``AutoModel`` with call-recording mocks.

    Returns the (processor_mock, model_mock) pair so tests can inspect
    invocations without loading any real HuggingFace weights.
    """
    processor_mock = MagicMock(name="processor")
    processor_return = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    processor_return_obj = MagicMock(name="processor_call_return")
    processor_return_obj.to.return_value = processor_return
    processor_mock.return_value = processor_return_obj

    model_mock = MagicMock(name="model")
    model_mock.device = torch.device("cpu")
    model_mock.return_value = MagicMock(pooler_output=torch.zeros(1, 768))

    auto_processor = MagicMock()
    auto_processor.from_pretrained.return_value = processor_mock
    auto_model = MagicMock()
    auto_model.from_pretrained.return_value = model_mock

    monkeypatch.setattr(eval_utils, "AutoImageProcessor", auto_processor)
    monkeypatch.setattr(eval_utils, "AutoModel", auto_model)
    return processor_mock, model_mock


def test_dinov3_extract_features_passes_do_rescale_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extract_features`` must opt out of the processor's default 1/255 rescale.

    The DINOv3 ``AutoImageProcessor`` ships with ``do_rescale=True`` and
    ``rescale_factor=1/255`` — appropriate for uint8 [0, 255] PIL input.
    Our crops arrive as float [0, 1] (robust percentile normalization is
    applied upstream by :func:`build_crops`), so leaving rescale on divides
    by 255 a second time and the model sees essentially-black inputs whose
    pooled features are cosine-uncorrelated with the intended
    representation.
    """
    processor_mock, _ = _stub_processor_and_model(monkeypatch)
    extractor = eval_utils.DinoV3FeatureExtractor("facebook/test-dinov3")

    extractor.extract_features(np.zeros((64, 64), dtype=np.float32))

    processor_mock.assert_called_once()
    assert processor_mock.call_args.kwargs.get("do_rescale") is False, (
        f"processor call should pass do_rescale=False; got kwargs={processor_mock.call_args.kwargs!r}"
    )


def test_dinov3_extract_features_batch_passes_do_rescale_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extract_features_batch`` must opt out of the rescale on every chunk."""
    processor_mock, _ = _stub_processor_and_model(monkeypatch)
    extractor = eval_utils.DinoV3FeatureExtractor("facebook/test-dinov3")

    images = [np.zeros((64, 64), dtype=np.float32) for _ in range(3)]
    extractor.extract_features_batch(images, batch_size=2)

    # batch_size=2 over 3 images means 2 processor calls (chunk_size=2 then chunk_size=1).
    assert processor_mock.call_count == 2
    for call in processor_mock.call_args_list:
        assert call.kwargs.get("do_rescale") is False, (
            f"every chunk should pass do_rescale=False; got kwargs={call.kwargs!r}"
        )


def test_dinov3_extractor_pins_processor_do_rescale_false_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the extractor must flip ``processor.do_rescale`` to ``False``.

    The per-call ``do_rescale=False`` override on every ``self.processor(...)``
    invocation is the primary guard; pinning the instance attribute here is
    defense-in-depth so a future helper that forgets the kwarg cannot
    silently re-enable the buggy double-rescale path.
    """
    processor_mock, _ = _stub_processor_and_model(monkeypatch)
    processor_mock.do_rescale = True  # simulate the HF default before init touches it
    eval_utils.DinoV3FeatureExtractor("facebook/test-dinov3")
    assert processor_mock.do_rescale is False


def test_dinov3_preprocess_version_is_v3() -> None:
    """The recipe-version tag must read ``imagenet_normalize_v3``.

    The v3 bump invalidates every v2 cache entry, which was extracted from
    raw min-max crops (outlier-dominated, GT-vs-pred asymmetric) before
    ``build_crops`` switched to robust percentile normalization. Soft-
    invalidate (see ``_auto_invalidate_on_preprocess_version_mismatch`` in
    ``pipeline_cache.py``) keys on this string.
    """
    assert eval_utils.DinoV3FeatureExtractor.PREPROCESS_VERSION == "imagenet_normalize_v3"


class _StubMorphEm(torch.nn.Module):
    """Stand-in for ``MorphEmModel`` that skips the HF load entirely."""

    def __init__(self, model_name: str, revision: str | None = None, img_size: int = 224, freeze: bool = True) -> None:
        super().__init__()
        self.model_name = model_name
        self.revision = revision

    def preprocess_2d(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = torch.zeros(x.shape[0], 384)
        return feats, feats


def test_morphem_extract_features_returns_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extract_features`` returns a single ``(1, 384)`` CLS embedding."""
    monkeypatch.setattr(eval_utils, "MorphEmModel", _StubMorphEm)
    extractor = eval_utils.MorphEmFeatureExtractor("CaicedoLab/MorphEm")
    out = extractor.extract_features(np.zeros((64, 64), dtype=np.float32))
    assert tuple(out.shape) == (1, 384)


def test_morphem_extract_features_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extract_features_batch`` stacks crops and returns ``(N, 384)``."""
    monkeypatch.setattr(eval_utils, "MorphEmModel", _StubMorphEm)
    extractor = eval_utils.MorphEmFeatureExtractor("CaicedoLab/MorphEm")
    images = [np.zeros((64, 64), dtype=np.float32) for _ in range(3)]
    out = extractor.extract_features_batch(images, batch_size=2)
    assert tuple(out.shape) == (3, 384)


def test_morphem_extractor_forwards_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub commit pin reaches ``MorphEmModel`` so trust_remote_code loads audited code."""
    monkeypatch.setattr(eval_utils, "MorphEmModel", _StubMorphEm)
    extractor = eval_utils.MorphEmFeatureExtractor("CaicedoLab/MorphEm", revision="0e8d5878")
    assert extractor.model.revision == "0e8d5878"


def test_morphem_preprocess_version() -> None:
    """The recipe-version tag must read ``per_image_norm_v2``.

    The v2 bump invalidates v1 caches after ``build_crops`` switched to
    robust percentile normalization — the non-affine percentile clip
    changes the z-scored input even though the min-max prescale alone
    would cancel under MorphEm's per-image z-score.
    """
    assert eval_utils.MorphEmFeatureExtractor.PREPROCESS_VERSION == "per_image_norm_v2"


class _CaptureContrastive:
    """Stand-in ``ContrastiveModule`` that records the tensor it is called with."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.last_input: torch.Tensor | None = None

    def to(self, *args, **kwargs) -> "_CaptureContrastive":
        return self

    def eval(self) -> "_CaptureContrastive":
        return self

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.last_input = x
        return torch.zeros(x.shape[0], 8), None


def _make_dynaclr_extractor(monkeypatch: pytest.MonkeyPatch) -> tuple[object, _CaptureContrastive]:
    """Build a DynaCLR extractor whose model is a capture stub (no checkpoint)."""
    capture = _CaptureContrastive()

    class _StubModule:
        @staticmethod
        def load_from_checkpoint(checkpoint, map_location, encoder):
            return capture

    monkeypatch.setattr(eval_utils, "ContrastiveModule", _StubModule)
    monkeypatch.setattr(eval_utils, "ContrastiveEncoder", lambda **kw: object())
    extractor = eval_utils.DynaCLRFeatureExtractor(checkpoint="unused.ckpt", encoder_config={})
    return extractor, capture


def test_dynaclr_zscores_crop_before_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The crop reaching the encoder must be per-image z-scored (mean~0, std~1).

    DynaCLR trains on ``NormalizeSampled`` z-scored inputs, so feeding it the
    bounded [0, 1] crop from ``build_crops`` was a train/test mismatch. The
    extractor now per-crop z-scores; an arbitrary-range crop must arrive at
    the encoder with zero spatial mean and unit spatial std.
    """
    extractor, capture = _make_dynaclr_extractor(monkeypatch)
    crop = np.arange(256, dtype=np.float32).reshape(16, 16) * 3.0 + 50.0  # non-constant, offset range

    extractor.extract_features(crop)

    x = capture.last_input
    assert x is not None
    assert float(x.mean().abs()) < 1e-4
    assert abs(float(x.std(unbiased=False)) - 1.0) < 1e-3


def test_dynaclr_batch_zscores_each_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batched crops are z-scored independently per crop."""
    extractor, capture = _make_dynaclr_extractor(monkeypatch)
    crops = [
        np.full((8, 8), 5.0, dtype=np.float32) + np.arange(64, dtype=np.float32).reshape(8, 8),
        np.arange(64, dtype=np.float32).reshape(8, 8) * 10.0,  # very different range
    ]

    extractor.extract_features_batch(crops, batch_size=8)

    x = capture.last_input  # (2, 1, 1, 8, 8)
    per_crop_mean = x.mean(dim=(-2, -1))
    per_crop_std = x.std(dim=(-2, -1), unbiased=False)
    assert torch.all(per_crop_mean.abs() < 1e-4)
    assert torch.all((per_crop_std - 1.0).abs() < 1e-3)


def test_morphem_null_name_soft_skips_in_load_eval_models() -> None:
    """A null ``pretrained_model_name`` disables MorphEm without crashing.

    Mirrors the celldino ``weights_path: null`` disable path: with the
    flag on but the hub id unset, ``load_eval_models`` returns
    ``models.morphem is None`` and no identity tags.
    """
    from omegaconf import OmegaConf

    from dynacell.evaluation.model_loader import LoadFlags, load_eval_models

    config = OmegaConf.create(
        {"target_name": "nucleus", "feature_extractor": {"morphem": {"pretrained_model_name": None}}}
    )
    models = load_eval_models(config, flags=LoadFlags(masks=False, morphem=True))
    assert models.morphem is None
    assert models.morphem_model_name is None
    assert models.morphem_preprocess_version is None


def test_load_eval_models_pins_morphem_to_the_configured_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """``feature_extractor.morphem.revision`` is what the extractor loads, not the hub head."""
    from omegaconf import OmegaConf

    from dynacell.evaluation.model_loader import LoadFlags, load_eval_models

    monkeypatch.setattr(eval_utils, "MorphEmModel", _StubMorphEm)
    config = OmegaConf.create(
        {
            "target_name": "nucleus",
            "feature_extractor": {"morphem": {"pretrained_model_name": "CaicedoLab/MorphEm", "revision": "0e8d5878"}},
        }
    )
    models = load_eval_models(config, flags=LoadFlags(masks=False, morphem=True))
    assert models.morphem_model_name == "CaicedoLab/MorphEm"
    assert models.morphem.model.revision == "0e8d5878"
