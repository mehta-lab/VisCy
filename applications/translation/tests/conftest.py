"""Test fixtures for translation application tests."""

from pathlib import Path

import pytest
import torch

# Synthetic data dimensions
SYNTH_B = 2  # batch size
SYNTH_C = 1  # input channels (phase)
SYNTH_D = 5  # depth (z-stack)
SYNTH_H = 64  # height
SYNTH_W = 64  # width

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
