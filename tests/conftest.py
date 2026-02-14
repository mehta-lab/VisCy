from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from pytest import FixtureRequest, TempPathFactory, fixture

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

channel_names = ["Phase", "Retardance", "GFP", "DAPI"]


def _build_hcs(
    path: Path,
    channel_names: list[str],
    zyx_shape: tuple[int, int, int],
    dtype: DTypeLike,
    max_value: int | float,
    sharded: bool = False,
    multiscales: bool = False,
    num_timepoints: int = 2,
):
    dataset = open_ome_zarr(
        path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
        version="0.4" if not sharded else "0.5",
    )
    for row in ("A", "B"):
        for col in ("1", "2"):
            for fov in ("0", "1", "2", "3"):
                pos = dataset.create_position(row, col, fov)
                pos.create_image(
                    "0",
                    (
                        np.random.rand(num_timepoints, len(channel_names), *zyx_shape)
                        * max_value
                    ).astype(dtype),
                    chunks=(1, 1, 1, *zyx_shape[1:]),
                    shards_ratio=(
                        num_timepoints,
                        len(channel_names),
                        zyx_shape[0],
                        1,
                        1,
                    )
                    if sharded
                    else None,
                )
                if multiscales:
                    pos["1"] = pos["0"][::2, :, ::2, ::2, ::2]


@fixture(scope="session")
def preprocessed_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("preprocessed.zarr")
    _build_hcs(
        dataset_path, channel_names, (12, 256, 256), np.float32, 1.0, multiscales=True
    )
    # U[0, 1)
    expected = {"mean": 0.5, "std": 1 / np.sqrt(12), "median": 0.5, "iqr": 0.5}
    norm_meta = {channel: {"dataset_statistics": expected} for channel in channel_names}
    with open_ome_zarr(dataset_path, mode="r+") as dataset:
        dataset.zattrs["normalization"] = norm_meta
        for _, fov in dataset.positions():
            fov.zattrs["normalization"] = norm_meta
    return dataset_path


@fixture(scope="function", params=[False, True])
def small_hcs_dataset(
    tmp_path_factory: TempPathFactory, request: FixtureRequest
) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("small.zarr")
    _build_hcs(
        dataset_path, channel_names, (12, 64, 64), np.uint16, 1, sharded=request.param
    )
    return dataset_path


@fixture(scope="function")
def small_hcs_labels(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset with labels."""
    dataset_path = tmp_path_factory.mktemp("small_with_labels.zarr")
    _build_hcs(
        dataset_path, ["nuclei_labels", "membrane_labels"], (12, 64, 64), np.uint16, 50
    )
    return dataset_path


@fixture(scope="function")
def labels_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("labels.zarr")
    _build_hcs(dataset_path, ["DAPI", "GFP"], (2, 16, 16), np.uint16, 3)
    return dataset_path


@fixture(scope="function")
def tracks_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a HCS OME-Zarr dataset with tracking CSV results."""
    dataset_path = tmp_path_factory.mktemp("tracks.zarr")
    _build_hcs(dataset_path, ["nuclei_labels"], (1, 256, 256), np.uint16, 3)
    for fov_name, _ in open_ome_zarr(dataset_path).positions():
        fake_tracks = pd.DataFrame(
            {
                "track_id": [0, 1],
                "t": [0, 1],
                "y": [100, 200],
                "x": [96, 160],
                "id": [0, 1],
                "parent_track_id": [-1, -1],
                "parent_id": [-1, -1],
            }
        )
        fake_tracks.to_csv(dataset_path / fov_name / "tracks.csv", index=False)
    return dataset_path


@fixture(scope="function")
def temporal_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a temporal HCS OME-Zarr dataset with multiple timepoints."""
    dataset_path = tmp_path_factory.mktemp("temporal.zarr")
    _build_hcs(
        dataset_path,
        channel_names[:2],  # Use first 2 channels
        (10, 50, 50),
        np.uint16,
        65535,
        num_timepoints=5,
    )
    return dataset_path


@fixture(scope="function")
def tracks_with_gaps_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a HCS OME-Zarr dataset with tracking results with gaps in time."""
    dataset_path = tmp_path_factory.mktemp("tracks_gaps.zarr")
    _build_hcs(dataset_path, ["nuclei_labels"], (1, 256, 256), np.uint16, 3)

    # Define different track patterns for different FOVs
    track_patterns = {
        "A/1/0": [
            # Track 0: complete sequence t=[0,1,2,3]
            {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            {"track_id": 0, "t": 1, "y": 128, "x": 128, "id": 1},
            {"track_id": 0, "t": 2, "y": 128, "x": 128, "id": 2},
            {"track_id": 0, "t": 3, "y": 128, "x": 128, "id": 3},
            # Track 1: ends early t=[0,1]
            {"track_id": 1, "t": 0, "y": 100, "x": 100, "id": 4},
            {"track_id": 1, "t": 1, "y": 100, "x": 100, "id": 5},
        ],
        "A/1/1": [
            # Track 0: gap at t=2, has t=[0,1,3]
            {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            {"track_id": 0, "t": 1, "y": 128, "x": 128, "id": 1},
            {"track_id": 0, "t": 3, "y": 128, "x": 128, "id": 2},
            # Track 1: even timepoints only t=[0,2,4]
            {"track_id": 1, "t": 0, "y": 100, "x": 100, "id": 3},
            {"track_id": 1, "t": 2, "y": 100, "x": 100, "id": 4},
            {"track_id": 1, "t": 4, "y": 100, "x": 100, "id": 5},
        ],
        "A/2/0": [
            # Track 0: single timepoint t=[0]
            {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            # Track 1: complete short sequence t=[0,1,2]
            {"track_id": 1, "t": 0, "y": 100, "x": 100, "id": 1},
            {"track_id": 1, "t": 1, "y": 100, "x": 100, "id": 2},
            {"track_id": 1, "t": 2, "y": 100, "x": 100, "id": 3},
        ],
    }

    for fov_name, _ in open_ome_zarr(dataset_path).positions():
        if fov_name in track_patterns:
            tracks_data = track_patterns[fov_name]
        else:
            # Default tracks for other FOVs
            tracks_data = [
                {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            ]

        tracks_df = pd.DataFrame(tracks_data)
        tracks_df["parent_track_id"] = -1
        tracks_df["parent_id"] = -1
        tracks_df.to_csv(dataset_path / fov_name / "tracks.csv", index=False)

    return dataset_path


def _make_synthetic_embeddings(
    tmp_path: Path,
    name: str,
    n_samples: int,
    n_features: int,
    channels: list[str],
    rng: np.random.Generator,
) -> Path:
    """Create a fake embeddings directory with per-channel zarr files."""
    version_dir = tmp_path / name
    version_dir.mkdir(parents=True, exist_ok=True)

    for channel in channels:
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "fov_name": [f"A/{(i % 3) + 1}/0" for i in range(n_samples)],
                "id": np.arange(n_samples),
                "t": np.zeros(n_samples, dtype=int),
                "track_id": np.arange(n_samples),
            }
        )
        adata = ad.AnnData(X=X, obs=obs)
        adata.write_zarr(version_dir / f"timeaware_{channel}_160patch_99ckpt.zarr")

    return version_dir


def _make_annotation_csv(
    tmp_path: Path,
    name: str,
    n_samples: int,
    tasks: dict[str, list[str]],
    rng: np.random.Generator,
) -> Path:
    """Create a fake annotation CSV with specified task columns."""
    csv_path = tmp_path / f"{name}_annotations.csv"
    data = {
        "fov_name": [f"A/{(i % 3) + 1}/0" for i in range(n_samples)],
        "id": np.arange(n_samples),
    }
    for task, labels in tasks.items():
        data[task] = rng.choice(labels, size=n_samples)

    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@fixture(scope="function")
def synthetic_train_data(tmp_path_factory: TempPathFactory):
    """Two synthetic training datasets for eval pipeline tests.

    Each: 40 samples x 16 features, channels=[phase, organelle].
    Tasks: infection_state, cell_division_state.
    """
    from applications.DynaCLR.evaluation.linear_classifiers.evaluate_dataset import (
        TrainDataset,
    )

    base = tmp_path_factory.mktemp("train_data")
    rng = np.random.default_rng(42)
    channels = ["phase", "organelle"]
    tasks = {
        "infection_state": ["infected", "uninfected"],
        "cell_division_state": ["interphase", "mitosis"],
    }

    datasets = []
    for i in range(2):
        emb_dir = _make_synthetic_embeddings(
            base, f"train_ds_{i}/predictions/v1", 40, 16, channels, rng
        )
        csv_path = _make_annotation_csv(base, f"train_ds_{i}", 40, tasks, rng)
        datasets.append(TrainDataset(embeddings_dir=emb_dir, annotations=csv_path))

    return datasets


@fixture(scope="function")
def synthetic_test_data(tmp_path_factory: TempPathFactory):
    """Held-out test dataset: 60 samples x 16 features, 2 channels + CSV."""
    base = tmp_path_factory.mktemp("test_data")
    rng = np.random.default_rng(99)
    channels = ["phase", "organelle"]
    tasks = {
        "infection_state": ["infected", "uninfected"],
        "cell_division_state": ["interphase", "mitosis"],
    }

    emb_dir = _make_synthetic_embeddings(
        base, "test_ds/predictions/v1", 60, 16, channels, rng
    )
    csv_path = _make_annotation_csv(base, "test_ds", 60, tasks, rng)
    return emb_dir, csv_path


@fixture(scope="function")
def eval_config(synthetic_train_data, synthetic_test_data, tmp_path_factory):
    """Full DatasetEvalConfig with 1 model, 2 channels, 2 tasks."""
    from applications.DynaCLR.evaluation.linear_classifiers.evaluate_dataset import (
        DatasetEvalConfig,
        ModelSpec,
    )

    test_emb_dir, test_csv = synthetic_test_data
    return DatasetEvalConfig(
        dataset_name="test_dataset",
        test_annotations_csv=test_csv,
        models={
            "test_model": ModelSpec(
                name="TestModel",
                train_datasets=synthetic_train_data,
                test_embeddings_dir=test_emb_dir,
                version="v1",
                wandb_project="test-project",
            ),
        },
        output_dir=tmp_path_factory.mktemp("eval_output"),
        channels=["phase", "organelle"],
        tasks=["infection_state", "cell_division_state"],
    )


@fixture(scope="function")
def annotated_adata() -> ad.AnnData:
    """Provides an in-memory AnnData with 60 samples, 16 features, and annotations."""
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
            "t": rng.integers(0, 10, size=n_samples),
            "track_id": rng.integers(0, 5, size=n_samples),
            "parent_track_id": np.full(n_samples, -1),
            "parent_id": np.full(n_samples, -1),
            "x": rng.integers(0, 256, size=n_samples),
            "y": rng.integers(0, 256, size=n_samples),
            "cell_death_state": labels,
        }
    )

    return ad.AnnData(X=X, obs=obs)


@fixture(scope="function")
def annotated_adata_zarr(tmp_path_factory: TempPathFactory, annotated_adata) -> dict:
    """Writes annotated_adata to zarr and a matching annotations CSV.

    Returns a dict with 'embeddings' and 'annotations' paths.
    """
    base_dir = tmp_path_factory.mktemp("annotated_data")

    zarr_path = base_dir / "embeddings.zarr"
    adata_disk = annotated_adata.copy()
    cols_for_csv = ["fov_name", "id", "cell_death_state"]
    annotations_df = adata_disk.obs[cols_for_csv].copy()
    adata_disk.obs = adata_disk.obs.drop(columns=["cell_death_state"])
    adata_disk.write_zarr(zarr_path)

    csv_path = base_dir / "annotations.csv"
    annotations_df.to_csv(csv_path, index=False)

    return {"embeddings": zarr_path, "annotations": csv_path}
