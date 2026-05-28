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
    Our crops arrive as float [0, 1] (``_minmax_norm`` is applied
    upstream by :func:`build_crops`), so leaving rescale on divides by
    255 a second time and the model sees essentially-black inputs whose
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


def test_dinov3_preprocess_version_is_v2() -> None:
    """The recipe-version tag must read ``imagenet_normalize_v2``.

    The v2 bump invalidates every v1 cache entry, which was extracted
    with ``do_rescale=True`` and is incompatible with the corrected
    feature distribution. Soft-invalidate (see
    ``_auto_invalidate_on_preprocess_version_mismatch`` in
    ``pipeline_cache.py``) keys on this string.
    """
    assert eval_utils.DinoV3FeatureExtractor.PREPROCESS_VERSION == "imagenet_normalize_v2"
