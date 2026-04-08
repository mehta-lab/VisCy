"""Tests for the orchestrated linear classifiers evaluation."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.evaluate_config import AnnotationSource, LinearClassifiersStepConfig, TaskSpec
from dynaclr.evaluation.linear_classifiers.orchestrated import run_linear_classifiers


def _make_embeddings(tmp_path: Path, n_cells: int = 200, n_features: int = 16) -> Path:
    """Create a synthetic embeddings zarr with experiment/marker/perturbation in obs."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_cells, n_features)).astype(np.float32)

    half = n_cells // 2
    obs = pd.DataFrame(
        {
            "fov_name": pd.array([f"A/1/FOV{i % 5}" for i in range(n_cells)], dtype=object),
            "id": list(range(n_cells)),
            "t": [i % 10 for i in range(n_cells)],
            "track_id": list(range(n_cells)),
            "experiment": pd.array(["exp_A"] * half + ["exp_B"] * half, dtype=object),
            "marker": pd.array(
                ["Phase3D"] * (half // 2)
                + ["TOMM20"] * (half // 2)
                + ["Phase3D"] * (half // 2)
                + ["TOMM20"] * (half // 2),
                dtype=object,
            ),
            "perturbation": pd.array(
                ["uninfected"] * (n_cells // 4)
                + ["ZIKV"] * (n_cells // 4)
                + ["uninfected"] * (n_cells // 4)
                + ["ZIKV"] * (n_cells // 4),
                dtype=object,
            ),
        }
    )

    obs.index = pd.RangeIndex(n_cells)
    adata = ad.AnnData(X=X, obs=obs)
    zarr_path = tmp_path / "embeddings.zarr"
    adata.write_zarr(zarr_path)
    return zarr_path


def _make_annotations(tmp_path: Path, experiment: str, fov_names: list[str], ids: list[int]) -> Path:
    """Create a synthetic annotation CSV with infection_state labels."""
    labels = ["uninfected" if i % 3 != 0 else "infected" for i in ids]
    df = pd.DataFrame(
        {
            "fov_name": fov_names,
            "id": ids,
            "infection_state": labels,
            "organelle_state": ["normal" if i % 4 != 0 else "abnormal" for i in ids],
        }
    )
    csv_path = tmp_path / f"{experiment}_annotations.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_run_linear_classifiers_single_task(tmp_path):
    """End-to-end: one task, one marker filter, two experiments."""
    zarr_path = _make_embeddings(tmp_path)
    adata = ad.read_zarr(zarr_path)

    # Build annotation CSVs per experiment
    for exp in ["exp_A", "exp_B"]:
        exp_mask = adata.obs["experiment"] == exp
        fovs = adata.obs.loc[exp_mask, "fov_name"].tolist()
        ids = adata.obs.loc[exp_mask, "id"].tolist()
        _make_annotations(tmp_path, exp, fovs, ids)

    config = LinearClassifiersStepConfig(
        annotations=[
            AnnotationSource(experiment="exp_A", path=str(tmp_path / "exp_A_annotations.csv")),
            AnnotationSource(experiment="exp_B", path=str(tmp_path / "exp_B_annotations.csv")),
        ],
        tasks=[TaskSpec(task="infection_state", marker_filter="Phase3D")],
        use_scaling=True,
        split_train_data=0.8,
    )

    output_dir = tmp_path / "linear_classifiers"
    results = run_linear_classifiers(zarr_path, config, output_dir)

    assert not results.empty
    assert "task" in results.columns
    assert "val_accuracy" in results.columns
    assert results.iloc[0]["task"] == "infection_state"
    assert results.iloc[0]["marker_filter"] == "Phase3D"
    assert (output_dir / "metrics_summary.csv").exists()


def test_run_linear_classifiers_multiple_tasks(tmp_path):
    """Multiple tasks and marker filters produce one row each in results."""
    zarr_path = _make_embeddings(tmp_path)
    adata = ad.read_zarr(zarr_path)

    for exp in ["exp_A", "exp_B"]:
        exp_mask = adata.obs["experiment"] == exp
        fovs = adata.obs.loc[exp_mask, "fov_name"].tolist()
        ids = adata.obs.loc[exp_mask, "id"].tolist()
        _make_annotations(tmp_path, exp, fovs, ids)

    config = LinearClassifiersStepConfig(
        annotations=[
            AnnotationSource(experiment="exp_A", path=str(tmp_path / "exp_A_annotations.csv")),
            AnnotationSource(experiment="exp_B", path=str(tmp_path / "exp_B_annotations.csv")),
        ],
        tasks=[
            TaskSpec(task="infection_state", marker_filter="Phase3D"),
            TaskSpec(task="organelle_state", marker_filter="TOMM20"),
        ],
        use_scaling=True,
        split_train_data=0.8,
    )

    output_dir = tmp_path / "linear_classifiers"
    results = run_linear_classifiers(zarr_path, config, output_dir)

    assert len(results) == 2
    tasks = set(results["task"].tolist())
    assert "infection_state" in tasks
    assert "organelle_state" in tasks


def test_run_linear_classifiers_no_marker_filter(tmp_path):
    """Running without marker_filter uses all embeddings."""
    zarr_path = _make_embeddings(tmp_path)
    adata = ad.read_zarr(zarr_path)

    for exp in ["exp_A", "exp_B"]:
        exp_mask = adata.obs["experiment"] == exp
        fovs = adata.obs.loc[exp_mask, "fov_name"].tolist()
        ids = adata.obs.loc[exp_mask, "id"].tolist()
        _make_annotations(tmp_path, exp, fovs, ids)

    config = LinearClassifiersStepConfig(
        annotations=[
            AnnotationSource(experiment="exp_A", path=str(tmp_path / "exp_A_annotations.csv")),
            AnnotationSource(experiment="exp_B", path=str(tmp_path / "exp_B_annotations.csv")),
        ],
        tasks=[TaskSpec(task="infection_state", marker_filter=None)],
        use_scaling=True,
        split_train_data=0.8,
    )

    output_dir = tmp_path / "linear_classifiers"
    results = run_linear_classifiers(zarr_path, config, output_dir)

    assert not results.empty
    # Without marker filter, n_samples is larger than with Phase3D filter
    assert results.iloc[0]["n_samples"] == adata.n_obs


def test_run_linear_classifiers_missing_metadata_raises(tmp_path):
    """Raises ValueError when embeddings.zarr lacks experiment/marker columns."""
    X = np.random.standard_normal((50, 8)).astype(np.float32)
    obs = pd.DataFrame({"fov_name": pd.array([f"A/1/FOV{i}" for i in range(50)], dtype=object), "id": list(range(50))})
    obs.index = pd.RangeIndex(50)
    adata = ad.AnnData(X=X, obs=obs)
    zarr_path = tmp_path / "embeddings.zarr"
    adata.write_zarr(zarr_path)

    config = LinearClassifiersStepConfig(
        annotations=[AnnotationSource(experiment="exp_A", path=str(tmp_path / "ann.csv"))],
        tasks=[TaskSpec(task="infection_state")],
    )

    with pytest.raises(ValueError, match="missing columns"):
        run_linear_classifiers(zarr_path, config, tmp_path / "out")


def test_run_linear_classifiers_unknown_marker_skipped(tmp_path):
    """If marker_filter matches no rows, task is skipped gracefully."""
    zarr_path = _make_embeddings(tmp_path)
    adata = ad.read_zarr(zarr_path)

    for exp in ["exp_A", "exp_B"]:
        exp_mask = adata.obs["experiment"] == exp
        fovs = adata.obs.loc[exp_mask, "fov_name"].tolist()
        ids = adata.obs.loc[exp_mask, "id"].tolist()
        _make_annotations(tmp_path, exp, fovs, ids)

    config = LinearClassifiersStepConfig(
        annotations=[
            AnnotationSource(experiment="exp_A", path=str(tmp_path / "exp_A_annotations.csv")),
        ],
        tasks=[TaskSpec(task="infection_state", marker_filter="NonExistentMarker")],
    )

    output_dir = tmp_path / "linear_classifiers"
    results = run_linear_classifiers(zarr_path, config, output_dir)

    assert results.empty
