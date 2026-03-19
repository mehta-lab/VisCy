"""Tests for NTXentHCL loss."""

import torch
from pytorch_metric_learning.losses import NTXentLoss

from viscy_models.contrastive.loss import NTXentHCL


def _make_embeddings_and_labels(n: int, dim: int = 64, seed: int = 0) -> tuple:
    """Return L2-normalized embeddings and labels for n anchor+positive pairs."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    embeddings = torch.randn(n * 2, dim, generator=rng)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    # Labels: 0,0,1,1,... — each pair shares a label
    labels = torch.arange(n).repeat_interleave(2)
    return embeddings, labels


def test_beta_zero_matches_ntxent():
    """NTXentHCL(beta=0) must produce identical loss to standard NTXentLoss."""
    embeddings, labels = _make_embeddings_and_labels(8)

    standard = NTXentLoss(temperature=0.1)
    hcl = NTXentHCL(temperature=0.1, beta=0.0)

    loss_standard = standard(embeddings, labels)
    loss_hcl = hcl(embeddings, labels)

    torch.testing.assert_close(loss_hcl, loss_standard)


def test_hard_negatives_increase_loss():
    """beta>0 should produce higher loss when hard negatives dominate.

    HCL only has effect with multiple negatives — with a single negative the
    re-weighting normalizes to 1.0 and reduces to standard NT-Xent.

    We build two batches of 8 anchor+positive pairs (16 embeddings total):
    - easy_batch: all negatives are random (low similarity to any anchor)
    - hard_batch: same positives, but negatives are near-copies of anchors

    With beta>0, HCL up-weights the hard negatives, producing higher loss
    on the hard batch relative to standard NT-Xent.
    """
    dim = 64
    n_pairs = 8
    temperature = 0.2
    beta = 0.5
    torch.manual_seed(0)

    anchors = torch.nn.functional.normalize(torch.randn(n_pairs, dim), dim=1)
    positives = torch.nn.functional.normalize(anchors + 0.01 * torch.randn(n_pairs, dim), dim=1)

    # Easy negatives: random directions
    easy_negs = torch.nn.functional.normalize(torch.randn(n_pairs, dim), dim=1)
    # Hard negatives: near-copies of anchors (high cosine similarity)
    hard_negs = torch.nn.functional.normalize(anchors + 0.05 * torch.randn(n_pairs, dim), dim=1)

    # Interleave anchor, positive per pair: [a0, p0, a1, p1, ...]
    anchor_positive = torch.stack([anchors, positives], dim=1).reshape(n_pairs * 2, dim)
    labels = torch.arange(n_pairs).repeat_interleave(2)

    hcl = NTXentHCL(temperature=temperature, beta=beta)
    standard = NTXentLoss(temperature=temperature)

    easy_batch = torch.cat([anchor_positive, easy_negs])
    easy_labels = torch.cat([labels, torch.arange(n_pairs, n_pairs * 2)])

    hard_batch = torch.cat([anchor_positive, hard_negs])
    hard_labels = easy_labels.clone()

    loss_easy_standard = standard(easy_batch, easy_labels)
    loss_hard_standard = standard(hard_batch, hard_labels)
    loss_easy_hcl = hcl(easy_batch, easy_labels)
    loss_hard_hcl = hcl(hard_batch, hard_labels)

    gap_standard = loss_hard_standard - loss_easy_standard
    gap_hcl = loss_hard_hcl - loss_easy_hcl

    assert gap_hcl > gap_standard, (
        f"HCL (beta={beta}) should widen the easy/hard gap vs standard NT-Xent. "
        f"gap_standard={gap_standard:.4f}, gap_hcl={gap_hcl:.4f}"
    )


def test_hard_negatives_get_higher_gradient():
    """In a batch with mixed easy/hard negatives, hard ones get larger gradients.

    HCL requires multiple negatives to re-weight — we build a batch with
    n_pairs anchor+positive pairs plus one easy and one hard negative.
    The hard negative should receive a larger gradient than the easy one.
    """
    dim = 64
    n_pairs = 8
    torch.manual_seed(0)

    anchors = torch.nn.functional.normalize(torch.randn(n_pairs, dim), dim=1)
    positives = torch.nn.functional.normalize(anchors + 0.01 * torch.randn(n_pairs, dim), dim=1)
    anchor_positive = torch.stack([anchors, positives], dim=1).reshape(n_pairs * 2, dim)
    ap_labels = torch.arange(n_pairs).repeat_interleave(2)

    easy_neg = torch.nn.functional.normalize(torch.randn(1, dim), dim=1).requires_grad_(True)
    hard_neg = (
        torch.nn.functional.normalize(anchors[0:1] + 0.05 * torch.randn(1, dim), dim=1).detach().requires_grad_(True)
    )

    hcl = NTXentHCL(temperature=0.2, beta=0.5)

    # Batch with easy negative
    easy_batch = torch.cat([anchor_positive, easy_neg])
    easy_labels = torch.cat([ap_labels, torch.tensor([n_pairs])])
    hcl(easy_batch, easy_labels).backward()
    grad_easy = easy_neg.grad.norm().item()

    # Batch with hard negative (same structure, different negative)
    hard_batch = torch.cat([anchor_positive, hard_neg])
    hard_labels = easy_labels.clone()
    hcl(hard_batch, hard_labels).backward()
    grad_hard = hard_neg.grad.norm().item()

    assert grad_hard > grad_easy, (
        f"Hard negative should receive larger gradient. grad_easy={grad_easy:.4f}, grad_hard={grad_hard:.4f}"
    )
