"""Tests for the EmbeddingSnapshotCallback."""

import anndata as ad
from conftest import SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W, SimpleEncoder, SyntheticTripletDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from dynaclr.engine import ContrastiveModule
from viscy_utils.callbacks import EmbeddingSnapshotCallback


def _make_module():
    return ContrastiveModule(
        encoder=SimpleEncoder(),
        loss_function=nn.TripletMarginLoss(margin=0.5),
        lr=1e-3,
        example_input_array_shape=(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
    )


def test_snapshot_written_at_correct_epochs(tmp_path):
    """Snapshots are written at epoch 0 and epoch 2 with every_n_epochs=2."""
    seed_everything(42)
    snapshot_dir = tmp_path / "snapshots"
    callback = EmbeddingSnapshotCallback(
        output_dir=snapshot_dir,
        every_n_epochs=2,
        store_images=False,
    )
    trainer = Trainer(
        max_epochs=3,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path / "logs"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(_make_module(), datamodule=SyntheticTripletDataModule())

    assert (snapshot_dir / "epoch_0.zarr").exists()
    assert (snapshot_dir / "epoch_2.zarr").exists()
    assert not (snapshot_dir / "epoch_1.zarr").exists()


def test_snapshot_contains_features_and_projections(tmp_path):
    """Snapshot AnnData has correct shapes for features and projections."""
    seed_everything(42)
    snapshot_dir = tmp_path / "snapshots"
    callback = EmbeddingSnapshotCallback(
        output_dir=snapshot_dir,
        every_n_epochs=1,
        store_images=False,
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path / "logs"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(_make_module(), datamodule=SyntheticTripletDataModule(batch_size=4))

    adata = ad.read_zarr(snapshot_dir / "epoch_0.zarr")
    assert adata.X.shape == (4, 64)
    assert adata.obsm["X_projections"].shape == (4, 32)
    assert "fov_name" in adata.obs.columns


def test_snapshot_stores_images(tmp_path):
    """When store_images=True, mid-Z patches are saved in obsm."""
    seed_everything(42)
    snapshot_dir = tmp_path / "snapshots"
    callback = EmbeddingSnapshotCallback(
        output_dir=snapshot_dir,
        every_n_epochs=1,
        store_images=True,
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path / "logs"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(_make_module(), datamodule=SyntheticTripletDataModule(batch_size=4))

    adata = ad.read_zarr(snapshot_dir / "epoch_0.zarr")
    assert "X_images" in adata.obsm
    image_shape = list(adata.uns["image_shape_cyx"])
    assert image_shape == [SYNTH_C, SYNTH_H, SYNTH_W]
    images = adata.obsm["X_images"].reshape(-1, *image_shape)
    assert images.shape == (4, SYNTH_C, SYNTH_H, SYNTH_W)


def test_snapshot_with_pca(tmp_path):
    """PCA is computed when pca_kwargs is provided."""
    seed_everything(42)
    snapshot_dir = tmp_path / "snapshots"
    callback = EmbeddingSnapshotCallback(
        output_dir=snapshot_dir,
        every_n_epochs=1,
        store_images=False,
        pca_kwargs={"n_components": 3},
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path / "logs"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(_make_module(), datamodule=SyntheticTripletDataModule(batch_size=4))

    adata = ad.read_zarr(snapshot_dir / "epoch_0.zarr")
    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape == (4, 3)


def test_snapshot_only_captures_first_batch(tmp_path):
    """Only the first validation batch is captured, not all batches."""
    seed_everything(42)
    snapshot_dir = tmp_path / "snapshots"
    callback = EmbeddingSnapshotCallback(
        output_dir=snapshot_dir,
        every_n_epochs=1,
        store_images=False,
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=tmp_path / "logs"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    # 8 samples, batch_size=2 => 4 val batches, but only first is captured
    trainer.fit(
        _make_module(),
        datamodule=SyntheticTripletDataModule(batch_size=2, num_samples=8),
    )

    adata = ad.read_zarr(snapshot_dir / "epoch_0.zarr")
    assert adata.X.shape[0] == 2
