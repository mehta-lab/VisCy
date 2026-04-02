"""Test fixtures for dynacell application tests."""

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# UNetViT3D test spatial sizes.
# With dims=[32,64,128], num_res_block=[2,2], stride (1,2,2):
#   latent = [8, 8, 8], patch_size=4 → 8 tokens.
SYNTH_B = 2
SYNTH_C = 1
SYNTH_D_VIT = 8
SYNTH_H_VIT = 32
SYNTH_W_VIT = 32

# FNet3D test spatial sizes (depth=1 → divisor=2).
SYNTH_D_FNET = 4
SYNTH_H_FNET = 16
SYNTH_W_FNET = 16

# UNeXt2 test spatial sizes (num_blocks=6 → divisor=64; YX must be ≥64 for training).
SYNTH_D_UNEXT2 = 5
SYNTH_H_UNEXT2 = 64
SYNTH_W_UNEXT2 = 64


class SyntheticDataset(Dataset):
    """Synthetic dataset returning Sample dicts."""

    def __init__(self, size=8, n_channels=1, depth=8, height=32, width=32):
        self.size = size
        self.n_channels = n_channels
        self.depth = depth
        self.height = height
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "source": torch.randn(self.n_channels, self.depth, self.height, self.width),
            "target": torch.randn(self.n_channels, self.depth, self.height, self.width),
            "index": (f"row/col/pos/{idx}", torch.tensor(0), torch.tensor(0)),
        }


class SyntheticDataModule(LightningDataModule):
    """DataModule wrapping SyntheticDataset for DynacellUNet train/val."""

    def __init__(self, batch_size=2, num_samples=8, **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dataset_kwargs = dataset_kwargs

    def train_dataloader(self):
        return DataLoader(
            SyntheticDataset(self.num_samples, **self.dataset_kwargs),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            SyntheticDataset(self.num_samples, **self.dataset_kwargs),
            batch_size=self.batch_size,
        )


@pytest.fixture
def synth_vit_batch():
    """Synthetic batch matching UNetViT3D spatial requirements."""
    return {
        "source": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D_VIT, SYNTH_H_VIT, SYNTH_W_VIT),
        "target": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D_VIT, SYNTH_H_VIT, SYNTH_W_VIT),
        "index": (
            ["row/col/pos/0"] * SYNTH_B,
            [torch.tensor(0)] * SYNTH_B,
            [torch.tensor(0)] * SYNTH_B,
        ),
    }


@pytest.fixture
def synth_fnet_batch():
    """Synthetic batch for FNet3D."""
    return {
        "source": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D_FNET, SYNTH_H_FNET, SYNTH_W_FNET),
        "target": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D_FNET, SYNTH_H_FNET, SYNTH_W_FNET),
        "index": (
            ["row/col/pos/0"] * SYNTH_B,
            [torch.tensor(0)] * SYNTH_B,
            [torch.tensor(0)] * SYNTH_B,
        ),
    }


@pytest.fixture
def synth_unext2_batch():
    """Synthetic batch for UNeXt2 (YX=64 required by num_blocks=6)."""
    return {
        "source": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D_UNEXT2, SYNTH_H_UNEXT2, SYNTH_W_UNEXT2),
        "target": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D_UNEXT2, SYNTH_H_UNEXT2, SYNTH_W_UNEXT2),
        "index": (
            ["row/col/pos/0"] * SYNTH_B,
            [torch.tensor(0)] * SYNTH_B,
            [torch.tensor(0)] * SYNTH_B,
        ),
    }


@pytest.fixture
def _SyntheticDataModule():
    """Return the SyntheticDataModule class (not an instance)."""
    return SyntheticDataModule


@pytest.fixture
def tiny_hcs_zarr(tmp_path):
    """Create a minimal HCS OME-Zarr with 4 positions."""
    zarr_path = tmp_path / "tiny.zarr"
    channel_names = ["Phase3D", "Fluorescence"]
    rng = np.random.default_rng(42)
    with open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=channel_names) as dataset:
        for col in ("1", "2"):
            for fov in ("0", "1"):
                pos = dataset.create_position("A", col, fov)
                pos.create_image(
                    "0",
                    rng.random((1, len(channel_names), SYNTH_D_VIT, SYNTH_H_VIT, SYNTH_W_VIT)).astype(np.float32),
                    chunks=(1, 1, SYNTH_D_VIT, SYNTH_H_VIT, SYNTH_W_VIT),
                )
    norm_meta = {ch: {"fov_statistics": {"mean": 0.5, "std": 0.29, "otsu_threshold": 0.5}} for ch in channel_names}
    with open_ome_zarr(zarr_path, mode="r+") as ds:
        for _, fov in ds.positions():
            fov.zattrs["normalization"] = norm_meta
    return zarr_path
