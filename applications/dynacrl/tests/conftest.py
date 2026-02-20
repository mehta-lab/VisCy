"""Shared fixtures and skip markers for DynaCLR integration tests."""

from pathlib import Path

import pytest
import torch

CHECKPOINT_PATH = Path(
    "/hpc/projects/organelle_phenotyping/models/"
    "SEC61_TOMM20_G3BP1_Sensor/time_interval/"
    "dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/"
    "saved_checkpoints/epoch=104-step=53760.ckpt"
)

REFERENCE_ZARR_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/"
    "4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/"
    "v3/timeaware_phase_160patch_104ckpt.zarr"
)

DATA_ZARR_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/"
    "4-phenotyping/train-test/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
)

TRACKS_ZARR_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/"
    "1-preprocess/label-free/3-track/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr"
)

HPC_PATHS_AVAILABLE = all(p.exists() for p in [CHECKPOINT_PATH, REFERENCE_ZARR_PATH, DATA_ZARR_PATH, TRACKS_ZARR_PATH])

GPU_AVAILABLE = torch.cuda.is_available()

requires_hpc_and_gpu = pytest.mark.skipif(
    not (HPC_PATHS_AVAILABLE and GPU_AVAILABLE),
    reason="Requires HPC data paths and CUDA GPU",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "hpc_integration: requires HPC paths and GPU")


@pytest.fixture
def checkpoint_path():
    return CHECKPOINT_PATH


@pytest.fixture
def reference_zarr_path():
    return REFERENCE_ZARR_PATH


@pytest.fixture
def data_zarr_path():
    return DATA_ZARR_PATH


@pytest.fixture
def tracks_zarr_path():
    return TRACKS_ZARR_PATH
