"""Shared fixtures, constants, and helpers for DynaCLR integration tests."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from viscy_data._typing import TripletSample
from viscy_data.collection import Collection, ExperimentEntry, save_collection

# ---------------------------------------------------------------------------
# HPC / GPU skip markers
# ---------------------------------------------------------------------------

_CHECKPOINT_PATH = Path(
    "/hpc/projects/organelle_phenotyping/models/"
    "SEC61_TOMM20_G3BP1_Sensor/time_interval/"
    "dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/"
    "saved_checkpoints/epoch=104-step=53760.ckpt"
)
_REFERENCE_ZARR_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/"
    "4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/"
    "v3/timeaware_phase_160patch_104ckpt.zarr"
)
_DATA_ZARR_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/"
    "4-phenotyping/train-test/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
)
_TRACKS_ZARR_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/"
    "1-preprocess/label-free/3-track/"
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr"
)

_HPC_PATHS_AVAILABLE = all(
    p.exists() for p in [_CHECKPOINT_PATH, _REFERENCE_ZARR_PATH, _DATA_ZARR_PATH, _TRACKS_ZARR_PATH]
)
_GPU_AVAILABLE = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Synthetic HCS data dimensions
# ---------------------------------------------------------------------------

IMG_H = 64
IMG_W = 64
N_T = 10
N_Z = 1
N_TRACKS = 5

# Synthetic tensor dimensions shared across unit tests.
SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W = 1, 1, 4, 4
SYNTH_FLAT_DIM = SYNTH_C * SYNTH_D * SYNTH_H * SYNTH_W
SYNTH_N_CLASSES = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_tracks_csv(
    path: Path,
    n_tracks: int = N_TRACKS,
    n_t: int = N_T,
    *,
    start_t: int = 0,
    parent_map: dict[int, int] | None = None,
    border_cell_track: int | None = None,
    outside_cell_track: int | None = None,
) -> None:
    """Write a tracking CSV with standard columns."""
    rows = []
    for tid in range(n_tracks):
        for t in range(start_t, start_t + n_t):
            y, x = 32.0, 32.0
            if border_cell_track is not None and tid == border_cell_track:
                y, x = 10.0, 10.0
            if outside_cell_track is not None and tid == outside_cell_track:
                y, x = -1.0, -1.0
            ptid = float("nan")
            if parent_map and tid in parent_map:
                ptid = parent_map[tid]
            rows.append(
                {
                    "track_id": tid,
                    "t": t,
                    "id": tid * n_t + t,
                    "parent_track_id": ptid,
                    "parent_id": float("nan"),
                    "z": 0,
                    "y": y,
                    "x": x,
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def create_experiment(
    tmp_path: Path,
    name: str,
    channel_names: list[str],
    wells: list[tuple[str, str]],
    perturbation_wells: dict[str, list[str]],
    fovs_per_well: int = 1,
    n_tracks: int = N_TRACKS,
    n_t: int = N_T,
    interval_minutes: float = 30.0,
    start_hpi: float = 0.0,
    channels: list | None = None,
    pixel_size_xy_um: float | None = None,
    pixel_size_z_um: float | None = None,
    parent_map: dict[int, int] | None = None,
    border_cell_track: int | None = None,
    outside_cell_track: int | None = None,
) -> ExperimentEntry:
    """Create a mini HCS OME-Zarr store, tracking CSVs, and return an ExperimentEntry."""
    from iohub.ngff import open_ome_zarr

    zarr_path = tmp_path / f"{name}.zarr"
    tracks_root = tmp_path / f"tracks_{name}"
    n_ch = len(channel_names)
    rng = np.random.default_rng(42)

    with open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=channel_names) as plate:
        for row, col in wells:
            for fov_idx in range(fovs_per_well):
                pos = plate.create_position(row, col, str(fov_idx))
                arr = pos.create_zeros(
                    "0",
                    shape=(n_t, n_ch, N_Z, IMG_H, IMG_W),
                    dtype=np.float32,
                )
                arr[:] = rng.standard_normal(arr.shape).astype(np.float32)
                fov_name = f"{row}/{col}/{fov_idx}"
                csv_path = tracks_root / fov_name / "tracks.csv"
                make_tracks_csv(
                    csv_path,
                    n_tracks=n_tracks,
                    n_t=n_t,
                    parent_map=parent_map,
                    border_cell_track=border_cell_track,
                    outside_cell_track=outside_cell_track,
                )

    return ExperimentEntry(
        name=name,
        data_path=str(zarr_path),
        tracks_path=str(tracks_root),
        channel_names=channel_names,
        channels=channels or [],
        perturbation_wells=perturbation_wells,
        interval_minutes=interval_minutes,
        start_hpi=start_hpi,
        pixel_size_xy_um=pixel_size_xy_um,
        pixel_size_z_um=pixel_size_z_um,
    )


def write_collection_yaml(
    tmp_path: Path,
    entries: list[ExperimentEntry],
) -> Path:
    """Write a collection YAML from ExperimentEntry objects."""
    collection = Collection(
        name="test_collection",
        experiments=entries,
    )
    yaml_path = tmp_path / "collection.yml"
    save_collection(collection, yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# Synthetic encoder and data classes
# ---------------------------------------------------------------------------


class SimpleEncoder(nn.Module):
    """Lightweight encoder that mimics ContrastiveEncoder's (features, projections) API."""

    def __init__(self, in_dim: int = SYNTH_FLAT_DIM, feature_dim: int = 64, projection_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(in_dim, feature_dim)
        self.proj = nn.Linear(feature_dim, projection_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.flatten(1)
        features = self.fc(x)
        projections = self.proj(features)
        return features, projections


class SyntheticTripletDataset(Dataset):
    """Generate random triplets with tracking index metadata."""

    def __init__(self, size: int = 8):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TripletSample:
        return {
            "anchor": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "positive": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "negative": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "index": {
                "fov_name": f"fov_{idx}",
                "id": idx,
                "track_id": idx % 3,
                "t": idx,
            },
            "anchor_meta": [
                {
                    "experiment": "exp_a",
                    "perturbation": "control" if idx % 2 == 0 else "treated",
                    "hours_post_perturbation": float(idx),
                    "t": idx,
                }
            ],
        }


class SyntheticTripletDataModule(LightningDataModule):
    """DataModule wrapping SyntheticTripletDataset for train and val."""

    def __init__(self, batch_size: int = 4, num_samples: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            SyntheticTripletDataset(self.num_samples),
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            SyntheticTripletDataset(self.num_samples),
            batch_size=self.batch_size,
        )


class SyntheticLabeledTripletDataset(Dataset):
    """Triplet dataset with integer class labels for auxiliary head testing.

    Uses ``__getitems__`` to return a pre-batched dict matching the real
    ``MultiExperimentTripletDataset`` contract (no default collation).
    """

    def __init__(self, size: int = 8, n_classes: int = SYNTH_N_CLASSES):
        self.size = size
        self.n_classes = n_classes

    def __len__(self) -> int:
        return self.size

    def __getitems__(self, indices: list[int]) -> TripletSample:
        b = len(indices)
        return {
            "anchor": torch.randn(b, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "positive": torch.randn(b, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "negative": torch.randn(b, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "index": {
                "fov_name": [f"fov_{i}" for i in indices],
                "id": list(indices),
                "track_id": [i % 3 for i in indices],
                "t": list(indices),
            },
            "anchor_meta": [
                {
                    "experiment": "exp_a",
                    "perturbation": "control",
                    "t": i,
                    "labels": {"gene_ko": i % self.n_classes},
                }
                for i in indices
            ],
        }


class SyntheticLabeledTripletDataModule(LightningDataModule):
    """DataModule wrapping SyntheticLabeledTripletDataset for auxiliary head tests."""

    def __init__(self, batch_size: int = 4, num_samples: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            SyntheticLabeledTripletDataset(self.num_samples),
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            SyntheticLabeledTripletDataset(self.num_samples),
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
        )


# ---------------------------------------------------------------------------
# Pytest configuration and fixtures
# ---------------------------------------------------------------------------


def pytest_configure(config):
    # anndata zarr writer does not support pandas ArrowStringArray (default in
    # pandas 2.x when PyArrow is installed). Verified still broken in
    # anndata==0.12.6 + zarr==3.x: IORegistryError on write_zarr with an
    # ArrowStringArray index. Remove when anndata fixes zarr 3 support.
    # See: https://github.com/scverse/anndata/issues/1510
    pd.options.future.infer_string = False
    config.addinivalue_line("markers", "hpc_integration: requires HPC paths and GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``hpc_integration`` when HPC paths or GPU are unavailable."""
    if _HPC_PATHS_AVAILABLE and _GPU_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="Requires HPC data paths and CUDA GPU")
    for item in items:
        if "hpc_integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def checkpoint_path():
    return _CHECKPOINT_PATH


@pytest.fixture
def data_zarr_path():
    return _DATA_ZARR_PATH


@pytest.fixture
def tracks_zarr_path():
    return _TRACKS_ZARR_PATH


@pytest.fixture
def annotated_adata() -> ad.AnnData:
    """Synthetic AnnData with cell_death_state labels for classifier tests."""
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 16
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    fov_names = [f"A/{(i % 4) + 1}/0" for i in range(n_samples)]
    labels = (["alive"] * 20) + (["dead"] * 20) + (["apoptotic"] * 20)
    obs = pd.DataFrame(
        {
            "fov_name": fov_names,
            "id": np.arange(n_samples),
            "cell_death_state": labels,
        }
    )
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def annotated_adata_zarr(annotated_adata, tmp_path) -> dict:
    """Write annotated_adata to zarr + CSV and return dataset dict."""
    zarr_path = tmp_path / "emb.zarr"
    annotated_adata.write_zarr(zarr_path)

    csv_path = tmp_path / "ann.csv"
    annotated_adata.obs[["fov_name", "id", "cell_death_state"]].to_csv(csv_path, index=False)

    return {"embeddings": str(zarr_path), "annotations": str(csv_path)}


@pytest.fixture
def synth_dims():
    """Synthetic tensor dimensions shared across unit tests."""
    return {"c": SYNTH_C, "d": SYNTH_D, "h": SYNTH_H, "w": SYNTH_W, "flat": SYNTH_FLAT_DIM}


@pytest.fixture
def hcs_dims():
    """Synthetic HCS data dimensions."""
    return {"img_h": IMG_H, "img_w": IMG_W, "n_t": N_T, "n_z": N_Z, "n_tracks": N_TRACKS}


# ---------------------------------------------------------------------------
# Factory fixtures — expose helper functions/classes to test files without
# requiring ``from .conftest import …`` (which breaks ``pytest --co`` when
# conftest is collected as a regular module).
# ---------------------------------------------------------------------------


@pytest.fixture
def _create_experiment():
    return create_experiment


@pytest.fixture
def _write_collection_yaml():
    return write_collection_yaml


@pytest.fixture
def _make_tracks_csv():
    return make_tracks_csv


@pytest.fixture
def _SimpleEncoder():
    return SimpleEncoder


@pytest.fixture
def _SyntheticTripletDataModule():
    return SyntheticTripletDataModule


@pytest.fixture
def _SyntheticLabeledTripletDataModule():
    return SyntheticLabeledTripletDataModule


@pytest.fixture
def synth_n_classes():
    return SYNTH_N_CLASSES


# ---------------------------------------------------------------------------
# YAML config validation helpers
# ---------------------------------------------------------------------------


def extract_class_paths(obj):
    """Recursively extract all class_path values from a parsed YAML dict."""

    paths = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "class_path" and isinstance(value, str):
                paths.append(value)
            else:
                paths.extend(extract_class_paths(value))
    elif isinstance(obj, list):
        for item in obj:
            paths.extend(extract_class_paths(item))
    return paths


def resolve_class_path(class_path: str):
    """Resolve a dotted class_path to the actual class object."""
    import importlib

    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@pytest.fixture
def _extract_class_paths():
    return extract_class_paths


@pytest.fixture
def _resolve_class_path():
    return resolve_class_path
