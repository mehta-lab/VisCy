"""Smoke tests for DynaCLR engine modules."""

import torch
from dynaclr.engine import ContrastiveModule
from torch import nn


def test_contrastive_module_init():
    """Test ContrastiveModule initializes without error."""

    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 64)
            self.proj = nn.Linear(64, 32)

        def forward(self, x):
            x = x.flatten(1)
            features = self.fc(x)
            projections = self.proj(features)
            return features, projections

    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        loss_function=nn.TripletMarginLoss(margin=0.5),
        lr=1e-3,
        example_input_array_shape=(1, 1, 1, 1, 10),
    )
    assert module.lr == 1e-3
    assert module.model is encoder


def test_contrastive_module_forward():
    """Test ContrastiveModule forward pass."""

    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 64)
            self.proj = nn.Linear(64, 32)

        def forward(self, x):
            x = x.flatten(1)
            features = self.fc(x)
            projections = self.proj(features)
            return features, projections

    encoder = SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        example_input_array_shape=(1, 1, 1, 1, 10),
    )

    x = torch.randn(2, 1, 1, 1, 10)
    features, projections = module(x)
    assert features.shape == (2, 64)
    assert projections.shape == (2, 32)
