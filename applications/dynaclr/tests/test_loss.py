"""TDD tests for NTXentHCL: NT-Xent with hard-negative concentration."""

import pytest
import torch
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn

from dynaclr.loss import NTXentHCL


def _make_embeddings_and_labels(
    batch_size: int = 16,
    embed_dim: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create (2N, D) embeddings and (2N,) labels for contrastive loss.

    First N are anchors, next N are positives.
    labels[i] == labels[i + N] for positive pairs.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    embeddings = torch.randn(2 * batch_size, embed_dim, generator=gen, device=device)
    indices = torch.arange(batch_size, device=device)
    labels = torch.cat([indices, indices])
    return embeddings, labels


class TestNTXentHCLSubclass:
    """Verify NTXentHCL is a proper subclass of NTXentLoss and nn.Module."""

    def test_ntxent_hcl_is_ntxent_subclass(self):
        loss = NTXentHCL()
        assert isinstance(loss, NTXentLoss)

    def test_ntxent_hcl_is_nn_module(self):
        loss = NTXentHCL()
        assert isinstance(loss, nn.Module)


class TestNTXentHCLBetaZero:
    """Verify beta=0.0 produces identical results to standard NTXentLoss."""

    def test_ntxent_hcl_beta_zero_matches_standard(self):
        temperature = 0.1
        standard = NTXentLoss(temperature=temperature)
        hcl = NTXentHCL(temperature=temperature, beta=0.0)

        embeddings, labels = _make_embeddings_and_labels(batch_size=16, embed_dim=128)

        loss_standard = standard(embeddings, labels)
        loss_hcl = hcl(embeddings, labels)

        assert torch.allclose(loss_hcl, loss_standard, atol=1e-6), (
            f"beta=0.0 HCL loss ({loss_hcl.item():.8f}) != "
            f"standard NTXent loss ({loss_standard.item():.8f})"
        )


class TestNTXentHCLBetaPositive:
    """Verify beta>0 produces different results (hard-negative concentration)."""

    def test_ntxent_hcl_beta_positive_differs(self):
        embeddings, labels = _make_embeddings_and_labels(batch_size=16, embed_dim=128)

        hcl_zero = NTXentHCL(temperature=0.1, beta=0.0)
        hcl_pos = NTXentHCL(temperature=0.1, beta=0.5)

        loss_zero = hcl_zero(embeddings, labels)
        loss_pos = hcl_pos(embeddings, labels)

        assert not torch.allclose(loss_zero, loss_pos, atol=1e-6), (
            f"beta=0.5 loss ({loss_pos.item():.8f}) should differ from "
            f"beta=0.0 loss ({loss_zero.item():.8f})"
        )

    def test_ntxent_hcl_hard_negatives_increase_loss(self):
        """Construct embeddings with a hard negative close to anchor.

        With beta>0, this hard negative gets upweighted, increasing loss.
        """
        torch.manual_seed(42)
        batch_size = 8
        embed_dim = 64

        # Create random embeddings for most pairs
        embeddings = torch.randn(2 * batch_size, embed_dim)
        # Make first negative (index 1) very similar to anchor (index 0)
        embeddings[1] = embeddings[0] + 0.01 * torch.randn(embed_dim)

        indices = torch.arange(batch_size)
        labels = torch.cat([indices, indices])

        hcl_zero = NTXentHCL(temperature=0.1, beta=0.0)
        hcl_pos = NTXentHCL(temperature=0.1, beta=1.0)

        loss_zero = hcl_zero(embeddings, labels)
        loss_pos = hcl_pos(embeddings, labels)

        assert loss_pos.item() > loss_zero.item(), (
            f"beta=1.0 loss ({loss_pos.item():.6f}) should be > "
            f"beta=0.0 loss ({loss_zero.item():.6f}) with hard negatives"
        )


class TestNTXentHCLGradients:
    """Verify gradient computation works correctly."""

    def test_ntxent_hcl_returns_scalar_with_grad(self):
        hcl = NTXentHCL(temperature=0.1, beta=0.5)
        embeddings, labels = _make_embeddings_and_labels(batch_size=8, embed_dim=64)
        embeddings.requires_grad_(True)

        loss = hcl(embeddings, labels)

        assert loss.shape == (), f"Loss shape should be (), got {loss.shape}"
        assert loss.requires_grad, "Loss should require grad"

    def test_ntxent_hcl_backward_passes(self):
        """Verify backward pass completes and gradients exist."""
        encoder = nn.Linear(64, 32)
        hcl = NTXentHCL(temperature=0.1, beta=0.5)

        torch.manual_seed(42)
        x = torch.randn(16, 64)
        embeddings = encoder(x)
        # Create pairs: first 8 are anchors, last 8 are positives
        indices = torch.arange(8)
        labels = torch.cat([indices, indices])

        loss = hcl(embeddings, labels)
        loss.backward()

        assert encoder.weight.grad is not None, "Encoder weight should have gradients"
        assert encoder.weight.grad.abs().sum() > 0, "Gradients should be non-zero"


class TestNTXentHCLTemperature:
    """Verify temperature parameter effect."""

    def test_ntxent_hcl_temperature_effect(self):
        embeddings, labels = _make_embeddings_and_labels(batch_size=16, embed_dim=128)

        hcl_low_temp = NTXentHCL(temperature=0.05, beta=0.5)
        hcl_high_temp = NTXentHCL(temperature=0.5, beta=0.5)

        loss_low = hcl_low_temp(embeddings, labels)
        loss_high = hcl_high_temp(embeddings, labels)

        assert not torch.allclose(loss_low, loss_high, atol=1e-4), (
            f"Different temperatures should produce different losses: "
            f"low={loss_low.item():.6f}, high={loss_high.item():.6f}"
        )


class TestNTXentHCLEdgeCases:
    """Edge cases: batch size 1, large batch, default parameters."""

    def test_ntxent_hcl_batch_size_one(self):
        """Single pair should not crash (loss may be degenerate)."""
        hcl = NTXentHCL(temperature=0.1, beta=0.5)
        embeddings, labels = _make_embeddings_and_labels(batch_size=1, embed_dim=64)
        loss = hcl(embeddings, labels)

        assert not torch.isnan(loss), "Loss should not be NaN for batch_size=1"
        assert not torch.isinf(loss), "Loss should not be Inf for batch_size=1"

    def test_ntxent_hcl_large_batch(self):
        """128 pairs should complete without numerical issues."""
        hcl = NTXentHCL(temperature=0.07, beta=0.5)
        embeddings, labels = _make_embeddings_and_labels(
            batch_size=128, embed_dim=128
        )
        loss = hcl(embeddings, labels)

        assert not torch.isnan(loss), "Loss should not be NaN for large batch"
        assert not torch.isinf(loss), "Loss should not be Inf for large batch"
        assert loss.item() > 0, "Loss should be positive"

    def test_ntxent_hcl_default_parameters(self):
        hcl = NTXentHCL()
        assert hcl.temperature == 0.07, f"Default temperature should be 0.07, got {hcl.temperature}"
        assert hcl.beta == 0.5, f"Default beta should be 0.5, got {hcl.beta}"


class TestNTXentHCLCUDA:
    """CUDA tests (skipped if no GPU available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ntxent_hcl_cuda(self):
        temperature = 0.1
        standard = NTXentLoss(temperature=temperature).cuda()
        hcl = NTXentHCL(temperature=temperature, beta=0.0).cuda()

        embeddings, labels = _make_embeddings_and_labels(
            batch_size=16, embed_dim=128, device="cuda"
        )

        loss_standard = standard(embeddings, labels)
        loss_hcl = hcl(embeddings, labels)

        assert torch.allclose(loss_hcl, loss_standard, atol=1e-6), (
            f"CUDA: beta=0.0 HCL ({loss_hcl.item():.8f}) != "
            f"standard ({loss_standard.item():.8f})"
        )
