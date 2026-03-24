"""Shared test helpers and constants for DynaCLR tests.

Importable as a regular module under pytest's importlib import mode.
Fixtures live in conftest.py; this module holds constants, helper functions,
and synthetic data classes that test files import directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from viscy_data._typing import TripletSample
from viscy_data.collection import ChannelEntry, Collection, ExperimentEntry, save_collection

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

requires_hpc_and_gpu = pytest.mark.skipif(
    not (_HPC_PATHS_AVAILABLE and _GPU_AVAILABLE),
    reason="Requires HPC data paths and CUDA GPU",
)

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


def make_tracks_csv(
    path: Path,
    n_tracks: int = N_TRACKS,
    n_t: int = N_T,
    *,
    start_t: int = 0,
    parent_map: dict[int, int] | None = None,
) -> None:
    """Write a tracking CSV with standard columns."""
    rows = []
    for tid in range(n_tracks):
        for t in range(start_t, start_t + n_t):
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
                    "y": 32.0,
                    "x": 32.0,
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
    condition_wells: dict[str, list[str]],
    fovs_per_well: int = 1,
    n_tracks: int = N_TRACKS,
    n_t: int = N_T,
    interval_minutes: float = 30.0,
    start_hpi: float = 0.0,
    pixel_size_xy_um: float | None = None,
    pixel_size_z_um: float | None = None,
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
                make_tracks_csv(csv_path, n_tracks=n_tracks, n_t=n_t)

    return ExperimentEntry(
        name=name,
        data_path=str(zarr_path),
        tracks_path=str(tracks_root),
        channels=[ChannelEntry(name=ch, marker=ch) for ch in channel_names],
        channel_names=channel_names,
        condition_wells=condition_wells,
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


def build_flat_cell_index(
    output_path: Path,
    entries: list[ExperimentEntry],
    channel_markers: dict[str, list[tuple[str, str]]],
) -> pd.DataFrame:
    """Build a flat cell index parquet (one row per cell x timepoint x channel).

    Parameters
    ----------
    output_path : Path
        Where to write the parquet.
    entries : list[ExperimentEntry]
        Experiment entries (must have data_path pointing to a valid zarr).
    channel_markers : dict[str, list[tuple[str, str]]]
        ``{experiment_name: [(zarr_channel_name, marker), ...]}``
        defining which channels to expand and their marker labels.

    Returns
    -------
    pd.DataFrame
        The flat cell index.
    """
    from viscy_data.cell_index import write_cell_index

    rows: list[dict] = []
    for entry in entries:
        channels = channel_markers[entry.name]
        zarr_path = Path(entry.data_path)
        tracks_root = Path(entry.tracks_path)

        from iohub.ngff import open_ome_zarr

        with open_ome_zarr(str(zarr_path), mode="r") as plate:
            for _pos_path, position in plate.positions():
                fov_path = position.zgroup.name.strip("/")
                parts = fov_path.split("/")
                well_name = "/".join(parts[:2])
                fov_name = parts[2]

                if well_name not in {w for ws in entry.perturbation_wells.values() for w in ws}:
                    continue

                condition = next(
                    (c for c, ws in entry.perturbation_wells.items() if well_name in ws),
                    "unknown",
                )

                csv_dir = tracks_root / fov_path
                csv_files = list(csv_dir.glob("*.csv"))
                if not csv_files:
                    continue
                tracks_df = pd.read_csv(csv_files[0])

                for _, track_row in tracks_df.iterrows():
                    cell_id = f"{entry.name}_{fov_path}_{int(track_row['track_id'])}_{int(track_row['t'])}"
                    base = {
                        "cell_id": cell_id,
                        "experiment": entry.name,
                        "store_path": str(zarr_path),
                        "tracks_path": str(tracks_root),
                        "fov": fov_name,
                        "well": well_name,
                        "y": float(track_row["y"]),
                        "x": float(track_row["x"]),
                        "z": int(track_row.get("z", 0)),
                        "perturbation": condition,
                        "t": int(track_row["t"]),
                        "track_id": int(track_row["track_id"]),
                        "global_track_id": f"{entry.name}_{fov_path}_{int(track_row['track_id'])}",
                        "lineage_id": f"{entry.name}_{fov_path}_{int(track_row['track_id'])}",
                        "parent_track_id": int(track_row["parent_track_id"])
                        if pd.notna(track_row.get("parent_track_id"))
                        else -1,
                        "hours_post_perturbation": entry.start_hpi + track_row["t"] * entry.interval_minutes / 60.0,
                        "interval_minutes": entry.interval_minutes,
                        "microscope": entry.microscope,
                        "organelle": entry.organelle,
                        "pixel_size_xy_um": entry.pixel_size_xy_um,
                        "pixel_size_z_um": entry.pixel_size_z_um,
                    }
                    for ch_name, marker in channels:
                        row = {**base, "channel_name": ch_name, "marker": marker}
                        rows.append(row)

    df = pd.DataFrame(rows)
    write_cell_index(df, output_path)
    return df


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


SYNTH_N_CLASSES = 10


class SyntheticLabeledTripletDataset(Dataset):
    """Triplet dataset with integer class labels for auxiliary head testing."""

    def __init__(self, size: int = 8, n_classes: int = SYNTH_N_CLASSES):
        self.size = size
        self.n_classes = n_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TripletSample:
        sample = {
            "anchor": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "positive": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "negative": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "index": {"fov_name": f"fov_{idx}", "id": idx, "track_id": idx % 3, "t": idx},
            "anchor_meta": [
                {
                    "experiment": "exp_a",
                    "perturbation": "control",
                    "t": idx,
                    "labels": {"gene_ko": idx % self.n_classes},
                }
            ],
        }
        return sample


class SyntheticLabeledTripletDataModule(LightningDataModule):
    """DataModule wrapping SyntheticLabeledTripletDataset for auxiliary head tests."""

    def __init__(self, batch_size: int = 4, num_samples: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(SyntheticLabeledTripletDataset(self.num_samples), batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(SyntheticLabeledTripletDataset(self.num_samples), batch_size=self.batch_size)
