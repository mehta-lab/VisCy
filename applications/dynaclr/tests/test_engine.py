"""Smoke tests for DynaCLR engine modules."""

import torch
from conftest import SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W, SimpleEncoder
from torch import nn

from dynaclr.engine import ContrastiveModule


def test_contrastive_module_init():
    """Test ContrastiveModule initializes without error."""
    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        loss_function=nn.TripletMarginLoss(margin=0.5),
        lr=1e-3,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )
    assert module.lr == 1e-3
    assert module.model is encoder


def test_contrastive_module_forward():
    """Test ContrastiveModule forward pass."""
    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )

    x = torch.randn(2, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W)
    features, projections = module(x)
    assert features.shape == (2, 64)
    assert projections.shape == (2, 32)
