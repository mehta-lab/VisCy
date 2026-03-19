"""Smoke tests for DynaCLR engine modules."""

import torch
from conftest import SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W, SimpleEncoder, SyntheticTripletDataModule
from lightning.pytorch import Trainer, seed_everything
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


def test_embedding_pca_logged_every_n_epochs():
    """PCA logging is triggered at epochs 0, n, 2n, ... and not in between."""
    seed_everything(0)
    n = 2
    module = ContrastiveModule(
        encoder=SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        log_embeddings_every_n_epochs=n,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )

    logged_epochs = []
    module.log_embedding_pca = lambda embeddings, meta, tag: logged_epochs.append(module.current_epoch)

    trainer = Trainer(
        max_epochs=5,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(module, datamodule=SyntheticTripletDataModule())

    assert logged_epochs == [0, 2, 4]


def test_embedding_pca_skipped_when_none():
    """No PCA logging when log_embeddings_every_n_epochs=None."""
    seed_everything(0)
    module = ContrastiveModule(
        encoder=SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        log_embeddings_every_n_epochs=None,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )

    logged = []
    module.log_embedding_pca = lambda *a, **kw: logged.append(True)

    trainer = Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(module, datamodule=SyntheticTripletDataModule())

    assert logged == []


def test_embedding_accumulator_cleared_after_epoch():
    """_embedding_outputs is empty after on_validation_epoch_end."""
    seed_everything(0)
    module = ContrastiveModule(
        encoder=SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        log_embeddings_every_n_epochs=1,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )
    module.log_embedding_pca = lambda *a, **kw: None

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(module, datamodule=SyntheticTripletDataModule())

    assert module._embedding_outputs == []
