"""TDD tests for ChannelDropout augmentation module."""

import pytest
import torch

from viscy_data.channel_dropout import ChannelDropout


def _make_input(batch: int = 4, channels: int = 2, z: int = 8, y: int = 64, x: int = 64) -> torch.Tensor:
    """Create a non-zero (B, C, Z, Y, X) test tensor."""
    return torch.randn(batch, channels, z, y, x) + 1.0  # shift so no accidental zeros


class TestChannelDropoutZeros:
    """Test that ChannelDropout zeros specified channels."""

    def test_channel_dropout_zeros_specified_channel(self):
        """p=1.0, channels=[1]: channel 1 must be all zeros."""
        module = ChannelDropout(channels=[1], p=1.0)
        module.train()
        inp = _make_input()
        out = module(inp)
        assert (out[:, 1] == 0).all(), "Channel 1 should be all zeros with p=1.0"

    def test_channel_dropout_preserves_other_channels(self):
        """p=1.0, channels=[1]: channel 0 must be unchanged."""
        module = ChannelDropout(channels=[1], p=1.0)
        module.train()
        inp = _make_input()
        out = module(inp)
        assert torch.equal(out[:, 0], inp[:, 0]), "Channel 0 should be unchanged"


class TestChannelDropoutProbability:
    """Test probability boundary conditions."""

    def test_channel_dropout_p_zero_identity(self):
        """p=0.0: output equals input exactly."""
        module = ChannelDropout(channels=[1], p=0.0)
        module.train()
        inp = _make_input()
        out = module(inp)
        assert torch.equal(out, inp), "p=0.0 should be identity"

    def test_channel_dropout_p_one_always_drops(self):
        """p=1.0: always drops, run multiple times."""
        module = ChannelDropout(channels=[1], p=1.0)
        module.train()
        for _ in range(10):
            inp = _make_input()
            out = module(inp)
            assert (out[:, 1] == 0).all(), "p=1.0 should always drop"

    def test_channel_dropout_probabilistic(self):
        """p=0.5: run 100 times, expect ~50% dropout rate (within 20-80%)."""
        module = ChannelDropout(channels=[1], p=0.5)
        module.train()
        dropped_count = 0
        total_samples = 0
        for _ in range(100):
            inp = _make_input(batch=1)
            out = module(inp)
            total_samples += 1
            if (out[0, 1] == 0).all():
                dropped_count += 1
        drop_rate = dropped_count / total_samples
        assert 0.20 <= drop_rate <= 0.80, f"Drop rate {drop_rate:.2f} outside expected range [0.20, 0.80]"


class TestChannelDropoutEval:
    """Test eval mode behavior."""

    def test_channel_dropout_eval_mode_identity(self):
        """eval mode: output equals input regardless of p."""
        module = ChannelDropout(channels=[1], p=1.0)
        module.eval()
        inp = _make_input()
        out = module(inp)
        assert torch.equal(out, inp), "eval mode should be identity"


class TestChannelDropoutPerSample:
    """Test per-sample independent dropout."""

    def test_channel_dropout_per_sample_independent(self):
        """batch of 16, p=0.5: not all samples should have the same dropout pattern."""
        module = ChannelDropout(channels=[1], p=0.5)
        module.train()
        # Run enough times to observe variation
        found_variation = False
        for _ in range(50):
            inp = _make_input(batch=16)
            out = module(inp)
            # Check if some samples dropped and some didn't
            sample_dropped = [(out[b, 1] == 0).all().item() for b in range(16)]
            if not all(sample_dropped) and any(sample_dropped):
                found_variation = True
                break
        assert found_variation, "Per-sample dropout should show variation across batch"


class TestChannelDropoutProperties:
    """Test tensor property preservation."""

    def test_channel_dropout_preserves_dtype_device(self):
        """float32 in -> float32 out, same device."""
        module = ChannelDropout(channels=[1], p=1.0)
        module.train()
        inp = _make_input().float()
        out = module(inp)
        assert out.dtype == inp.dtype, f"Expected {inp.dtype}, got {out.dtype}"
        assert out.device == inp.device, f"Expected {inp.device}, got {out.device}"

    def test_channel_dropout_does_not_modify_input(self):
        """Input tensor must not be modified after forward pass."""
        module = ChannelDropout(channels=[1], p=1.0)
        module.train()
        inp = _make_input()
        inp_clone = inp.clone()
        _ = module(inp)
        assert torch.equal(inp, inp_clone), "Input tensor should not be modified"

    def test_channel_dropout_multiple_channels(self):
        """channels=[0,1], p=1.0: both channels zeroed for all samples."""
        module = ChannelDropout(channels=[0, 1], p=1.0)
        module.train()
        inp = _make_input()
        out = module(inp)
        assert (out[:, 0] == 0).all(), "Channel 0 should be zeroed"
        assert (out[:, 1] == 0).all(), "Channel 1 should be zeroed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestChannelDropoutCUDA:
    """Test CUDA compatibility."""

    def test_channel_dropout_cuda(self):
        """Works on GPU tensors."""
        module = ChannelDropout(channels=[1], p=1.0).cuda()
        module.train()
        inp = _make_input().cuda()
        out = module(inp)
        assert out.device.type == "cuda", "Output should be on CUDA"
        assert (out[:, 1] == 0).all(), "Channel 1 should be zeroed on CUDA"
