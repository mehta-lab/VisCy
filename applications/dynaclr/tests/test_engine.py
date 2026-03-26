"""Smoke tests for DynaCLR engine modules."""

import torch
from lightning.pytorch import Trainer, seed_everything
from torch import nn

from dynaclr.engine import ContrastiveModule
from viscy_models.components.heads import ClassificationHead


def test_contrastive_module_init(_SimpleEncoder, synth_dims):
    """Test ContrastiveModule initializes without error."""
    encoder = _SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        loss_function=nn.TripletMarginLoss(margin=0.5),
        lr=1e-3,
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
    )
    assert module.lr == 1e-3
    assert module.model is encoder


def test_contrastive_module_forward(_SimpleEncoder, synth_dims):
    """Test ContrastiveModule forward pass."""
    encoder = _SimpleEncoder()
    module = ContrastiveModule(
        encoder=encoder,
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
    )

    x = torch.randn(2, synth_dims["c"], synth_dims["d"], synth_dims["h"], synth_dims["w"])
    features, projections = module(x)
    assert features.shape == (2, 64)
    assert projections.shape == (2, 32)


def test_embedding_pca_logged_every_n_epochs(_SimpleEncoder, _SyntheticTripletDataModule, synth_dims):
    """PCA logging is triggered at epochs 0, n, 2n, ... and not in between."""
    seed_everything(0)
    n = 2
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        log_embeddings_every_n_epochs=n,
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
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
    trainer.fit(module, datamodule=_SyntheticTripletDataModule())

    assert logged_epochs == [0, 2, 4]


def test_embedding_pca_skipped_when_none(_SimpleEncoder, _SyntheticTripletDataModule, synth_dims):
    """No PCA logging when log_embeddings_every_n_epochs=None."""
    seed_everything(0)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        log_embeddings_every_n_epochs=None,
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
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
    trainer.fit(module, datamodule=_SyntheticTripletDataModule())

    assert logged == []


def test_auxiliary_heads_registered_as_module_dict(_SimpleEncoder, synth_dims, synth_n_classes):
    """auxiliary_heads are stored in an nn.ModuleDict so parameters are tracked."""
    head = ClassificationHead("gene_ko", "gene_ko", in_dims=64, hidden_dims=32, num_classes=synth_n_classes)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        auxiliary_heads={"gene_ko": head},
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
    )
    assert isinstance(module.auxiliary_heads, nn.ModuleDict)
    assert "gene_ko" in module.auxiliary_heads
    assert any("auxiliary_heads" in name for name, _ in module.named_parameters())


def test_auxiliary_heads_training_step_runs(
    _SimpleEncoder, _SyntheticLabeledTripletDataModule, synth_dims, synth_n_classes
):
    """ContrastiveModule with a ClassificationHead trains without error."""
    seed_everything(0)
    head = ClassificationHead("gene_ko", "gene_ko", in_dims=64, hidden_dims=32, num_classes=synth_n_classes)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        auxiliary_heads={"gene_ko": head},
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(module, datamodule=_SyntheticLabeledTripletDataModule())


def test_auxiliary_heads_labels_from_anchor_meta(_SimpleEncoder, synth_dims, synth_n_classes):
    """Labels are correctly extracted from anchor_meta["labels"] and aux loss is nonzero."""
    seed_everything(0)
    head = ClassificationHead("gene_ko", "gene_ko", in_dims=64, hidden_dims=32, num_classes=synth_n_classes)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        auxiliary_heads={"gene_ko": head},
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
    )
    anchor_meta = [{"labels": {"gene_ko": torch.tensor([0, 1, 2, 3])}}]
    features = torch.randn(4, 64)

    y = module._get_labels({"anchor_meta": anchor_meta}, "gene_ko")
    assert y is not None
    assert y.shape == (4,)
    assert y.dtype == torch.long

    aux_loss = module._run_auxiliary_heads(features, {"anchor_meta": anchor_meta}, "train")
    assert aux_loss.item() > 0


def test_auxiliary_heads_none_no_extra_loss(_SimpleEncoder, _SyntheticTripletDataModule, synth_dims):
    """ContrastiveModule with no auxiliary heads trains identically to before."""
    seed_everything(0)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        auxiliary_heads=None,
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
    )
    assert len(module.auxiliary_heads) == 0
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(module, datamodule=_SyntheticTripletDataModule())


def test_embedding_accumulator_cleared_after_epoch(_SimpleEncoder, _SyntheticTripletDataModule, synth_dims):
    """_embedding_outputs is empty after on_validation_epoch_end."""
    seed_everything(0)
    module = ContrastiveModule(
        encoder=_SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        log_embeddings_every_n_epochs=1,
        example_input_array_shape=(
            1,
            synth_dims["c"],
            synth_dims["d"],
            synth_dims["h"],
            synth_dims["w"],
        ),
    )
    module.log_embedding_pca = lambda *a, **kw: None

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(module, datamodule=_SyntheticTripletDataModule())

    assert module._embedding_outputs == []
