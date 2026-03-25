"""Test fixtures for cytoland application tests."""

from pathlib import Path

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms.compose import Compose
from torch.utils.data import DataLoader, Dataset

from viscy_data.combined import CombinedDataModule
from viscy_data.gpu_aug import GPUTransformDataModule
from viscy_transforms import BatchedStackChannelsd

# Synthetic data dimensions
SYNTH_B = 2  # batch size
SYNTH_C = 1  # input channels (phase)
SYNTH_D = 5  # depth (z-stack)
SYNTH_H = 64  # height
SYNTH_W = 64  # width

# FCMAE needs 128x128 (64x64 creates degenerate 2x2 bottleneck with 7x7 depthwise conv).
FCMAE_H = 128
FCMAE_W = 128

# MixedLoss 5-scale MS-SSIM needs spatial/16 >= 11 (no padding in MONAI SSIM kernel).
MIXED_LOSS_H = 192
MIXED_LOSS_W = 192

# HPC path constants for inference reproducibility tests.
CHECKPOINT_PATH = Path(
    "/hpc/projects/comp.micro/virtual_staining/models/fcmae-cyto3d-sensor/"
    "vscyto3d-logs/hek-a549-ipsc-finetune/checkpoints/"
    "epoch=83-step=14532-loss=0.492.ckpt"
)

DATA_ZARR_PATH = Path(
    "/hpc/projects/virtual_staining/datasets/mehta-lab/VS_datasets/VSCyto3D/test/vscyto3d_test_fixture.zarr"
)

REFERENCE_ZARR_PATH = Path(
    "/hpc/projects/virtual_staining/datasets/mehta-lab/VS_datasets/VSCyto3D/test/vscyto3d_test_reference.zarr"
)

HPC_PATHS_AVAILABLE = all(p.exists() for p in [CHECKPOINT_PATH, DATA_ZARR_PATH, REFERENCE_ZARR_PATH])

GPU_AVAILABLE = torch.cuda.is_available()

requires_hpc_and_gpu = pytest.mark.skipif(
    not (HPC_PATHS_AVAILABLE and GPU_AVAILABLE),
    reason="Requires HPC data paths and CUDA GPU",
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "hpc_integration: requires HPC paths and GPU")


@pytest.fixture
def checkpoint_path():
    """Return path to vscyto3d checkpoint."""
    return CHECKPOINT_PATH


@pytest.fixture
def data_zarr_path():
    """Return path to input HCS OME-Zarr data."""
    return DATA_ZARR_PATH


@pytest.fixture
def reference_zarr_path():
    """Return path to reference prediction OME-Zarr."""
    return REFERENCE_ZARR_PATH


@pytest.fixture
def synth_dims():
    """Synthetic data dimensions shared across tests."""
    return {
        "b": SYNTH_B,
        "c": SYNTH_C,
        "d": SYNTH_D,
        "h": SYNTH_H,
        "w": SYNTH_W,
        "fcmae_h": FCMAE_H,
        "fcmae_w": FCMAE_W,
        "mixed_loss_h": MIXED_LOSS_H,
        "mixed_loss_w": MIXED_LOSS_W,
    }


@pytest.fixture
def synthetic_batch():
    """Create a synthetic batch dict matching the Sample type."""
    return {
        "source": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
        "target": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
        "index": (
            ["row/col/pos/0" for _ in range(SYNTH_B)],
            [torch.tensor(0) for _ in range(SYNTH_B)],
            [torch.tensor(0) for _ in range(SYNTH_B)],
        ),
    }


# ---------------------------------------------------------------------------
# Synthetic datasets and data modules for training integration tests
# ---------------------------------------------------------------------------


class SyntheticHCSDataset(Dataset):
    """Synthetic dataset returning Sample dicts with source, target, index."""

    def __init__(self, size=8, n_channels=1, depth=SYNTH_D, height=SYNTH_H, width=SYNTH_W):
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


class SyntheticHCSDataModule(LightningDataModule):
    """DataModule wrapping SyntheticHCSDataset for VSUNet train/val."""

    def __init__(self, batch_size=2, num_samples=8, **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dataset_kwargs = dataset_kwargs

    def train_dataloader(self):
        return DataLoader(
            SyntheticHCSDataset(self.num_samples, **self.dataset_kwargs),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            SyntheticHCSDataset(self.num_samples, **self.dataset_kwargs),
            batch_size=self.batch_size,
        )


class SyntheticGPUTransformDataset(Dataset):
    """Synthetic dataset returning [dict] matching CachedOmeZarrDataset format.

    Each item is a list containing one dict with per-channel-name tensors,
    compatible with ``list_data_collate``.
    """

    def __init__(self, size=8, depth=SYNTH_D, height=FCMAE_H, width=FCMAE_W):
        self.size = size
        self.depth = depth
        self.height = height
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return [
            {
                "Phase3D": torch.randn(1, self.depth, self.height, self.width),
                "Fluorescence": torch.randn(1, self.depth, self.height, self.width),
            }
        ]


class SyntheticGPUTransformDataModule(GPUTransformDataModule):
    """Synthetic GPUTransformDataModule with BatchedStackChannelsd for FCMAE tests.

    GPU transforms use BatchedStackChannelsd to map channel-name keys
    to source/target on batched ``(B, 1, Z, Y, X)`` tensors, matching the
    production CachedOmeZarrDataModule pattern with batched GPU transforms.
    """

    def __init__(self, batch_size=2, num_samples=8, depth=SYNTH_D, height=FCMAE_H, width=FCMAE_W):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 0
        self.pin_memory = False
        self.prefetch_factor = None
        self._depth = depth
        self._height = height
        self._width = width
        self._num_samples = num_samples
        stack = BatchedStackChannelsd({"source": ["Phase3D"], "target": ["Fluorescence"]})
        self._train_gpu = Compose([stack])
        self._val_gpu = Compose([stack])

    def setup(self, stage):
        self.train_dataset = SyntheticGPUTransformDataset(self._num_samples, self._depth, self._height, self._width)
        self.val_dataset = SyntheticGPUTransformDataset(self._num_samples, self._depth, self._height, self._width)

    @property
    def train_cpu_transforms(self):
        return Compose([])

    @property
    def train_gpu_transforms(self):
        return self._train_gpu

    @property
    def val_cpu_transforms(self):
        return Compose([])

    @property
    def val_gpu_transforms(self):
        return self._val_gpu


def make_synthetic_combined_datamodule(**kwargs):
    """Create a CombinedDataModule wrapping one SyntheticGPUTransformDataModule."""
    return CombinedDataModule([SyntheticGPUTransformDataModule(**kwargs)])


@pytest.fixture
def _SyntheticHCSDataModule():
    """Return the SyntheticHCSDataModule class."""
    return SyntheticHCSDataModule


@pytest.fixture
def _make_synthetic_combined_datamodule():
    """Return the make_synthetic_combined_datamodule factory function."""
    return make_synthetic_combined_datamodule


@pytest.fixture
def tiny_hcs_zarr(tmp_path):
    """Create a minimal HCS OME-Zarr with 4 positions for integration tests.

    Uses FCMAE_H/W spatial dims so both VSUNet (with yx_patch_size crop)
    and FCMAE tests can use the same fixture.
    """
    zarr_path = tmp_path / "tiny.zarr"
    channel_names = ["Phase3D", "Fluorescence"]
    dataset = open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=channel_names)
    rng = np.random.default_rng(42)
    for row in ("A",):
        for col in ("1", "2"):
            for fov in ("0", "1"):
                pos = dataset.create_position(row, col, fov)
                pos.create_image(
                    "0",
                    rng.random((1, len(channel_names), SYNTH_D, FCMAE_H, FCMAE_W)).astype(np.float32),
                    chunks=(1, 1, SYNTH_D, FCMAE_H, FCMAE_W),
                )
    dataset.close()
    # Write per-FOV normalization metadata.
    norm_meta = {ch: {"fov_statistics": {"mean": 0.5, "std": 0.29, "otsu_threshold": 0.5}} for ch in channel_names}
    with open_ome_zarr(zarr_path, mode="r+") as ds:
        for _, fov in ds.positions():
            fov.zattrs["normalization"] = norm_meta
    return zarr_path
