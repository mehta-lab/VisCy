"""Unit tests for Katamari's single-channel tensor contract."""

import builtins
import sys

import pytest
import torch

from dynaclr.evaluation.katamari import (
    KatamariFovNormalize,
    mask_incompatible_flash_attn,
    prepare_single_channel_input,
)


def test_masks_only_incompatible_optional_flash_attn(monkeypatch):
    original_import = builtins.__import__

    def fail_flash_attn(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("incompatible glibc")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_flash_attn)

    with pytest.warns(RuntimeWarning, match="PyTorch attention fallback"):
        assert mask_incompatible_flash_attn() is True
    assert sys.modules["flash_attn"] is None


def test_fov_normalize_uses_p1_and_p99_span() -> None:
    transform = KatamariFovNormalize(keys=["GFP"])
    sample = {
        "GFP": torch.tensor([[[[10.0, 55.0, 100.0]]]]),
        "norm_meta": {
            "GFP": {
                "fov_statistics": {
                    "p1": torch.tensor(10.0),
                    "p99_p1": torch.tensor(90.0),
                }
            }
        },
    }

    result = transform(sample)["GFP"]

    torch.testing.assert_close(result, torch.tensor([[[[0.0, 0.5, 1.0]]]]))


def test_prepare_single_channel_input_adds_time_axis_and_resizes():
    images = torch.rand(2, 1, 1, 160, 160)
    prepared = prepare_single_channel_input(images, (224, 224))
    assert prepared.shape == (2, 1, 1, 1, 224, 224)
    assert prepared.dtype == torch.float32


def test_prepare_single_channel_input_preserves_native_pixels():
    images = torch.rand(2, 1, 1, 224, 224)
    prepared = prepare_single_channel_input(images, (224, 224))
    torch.testing.assert_close(prepared[:, 0, :, 0], images[:, :, 0])


def test_prepare_single_channel_input_rejects_multichannel():
    with pytest.raises(ValueError, match="single-channel"):
        prepare_single_channel_input(torch.rand(2, 2, 1, 160, 160), (224, 224))


def test_prepare_single_channel_input_requires_reduced_z():
    with pytest.raises(ValueError, match="Z reduction"):
        prepare_single_channel_input(torch.rand(2, 1, 5, 160, 160), (224, 224))


def test_prepare_single_channel_input_rejects_invalid_size():
    with pytest.raises(ValueError, match="two positive integers"):
        prepare_single_channel_input(torch.rand(2, 1, 1, 160, 160), (224, 0))
