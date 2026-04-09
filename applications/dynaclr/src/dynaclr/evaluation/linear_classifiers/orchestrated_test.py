"""Tests for the orchestrated linear classifiers evaluation."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.evaluate_config import AnnotationSource, LinearClassifiersStepConfig, TaskSpec
from dynaclr.evaluation.linear_classifiers.orchestrated import run_linear_classifiers


def _make_embeddings_zarr(
    path: Path,
    n_cells: int = 200,
    n_features: int = 16,
    experiment: str = "exp_A",
    use_id_col: bool = True,
) -> ad.AnnData:
    """Write a synthetic embeddings zarr and return the AnnData."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_cells, n_features)).astype(np.float32)

    half = n_cells // 2
    obs: dict = {
        "fov_name": [f"A/1/FOV{i % 5}" for i in range(n_cells)],
        "t": [i % 10 for i in range(n_cells)],
        "track_id": list(range(n_cells)),
        "experiment": [experiment] * n_cells,
        "marker": ["Phase3D"] * half + ["TOMM20"] * half,
        "perturbation": ["uninfected"] * (n_cells // 2) + ["ZIKV"] * (n_cells // 2),
    }
    if use_id_col:
        obs["id"] = list(range(n_cells))

    df = pd.DataFrame(obs)
    # Convert string columns to object dtype — pandas 3 defaults to ArrowStringArray
    # which anndata's zarr writer does not support.
    for col in df.select_dtypes("string").columns:
        df[col] = df[col].astype(object)
    df.index = pd.Index([str(i) for i in range(n_cells)], dtype=object)
    var = pd.DataFrame(index=pd.Index([str(i) for i in range(n_features)], dtype=object))
    adata = ad.AnnData(X=X, obs=df, var=var)
    adata.write_zarr(path)
    return adata


def _make_embeddings_dir(tmp_path: Path, n_cells: int = 200, n_features: int = 16) -> Path:
    """Write two per-experiment zarrs to a directory; return the directory path."""
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    _make_embeddings_zarr(emb_dir / "exp_A.zarr", n_cells=n_cells, n_features=n_features, experiment="exp_A")
    _make_embeddings_zarr(emb_dir / "exp_B.zarr", n_cells=n_cells, n_features=n_features, experiment="exp_B")
    return emb_dir


def _make_annotations(tmp_path: Path, experiment: str, fov_names: list, ts: list, track_ids: list) -> Path:
    """Create a synthetic annotation CSV with infection_state and organelle_state labels."""
    labels = ["uninfected" if i % 3 != 0 else "infected" for i in range(len(fov_names))]
    df = pd.DataFrame(
        {
            "fov_name": fov_names,
            "t": ts,
            "track_id": track_ids,
            "infection_state": labels,
            "organelle_state": ["normal" if i % 4 != 0 else "abnormal" for i in range(len(fov_names))],
        }
    )
    csv_path = tmp_path / f"{experiment}_annotations.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _setup_dir_with_annotations(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create embeddings directory + annotation CSVs for exp_A and exp_B."""
    emb_dir = _make_embeddings_dir(tmp_path)
    ann_paths = {}
    for exp in ["exp_A", "exp_B"]:
        adata = ad.read_zarr(emb_dir / f"{exp}.zarr")
        ann_paths[exp] = _make_annotations(
            tmp_path,
            exp,
            adata.obs["fov_name"].tolist(),
            adata.obs["t"].tolist(),
            adata.obs["track_id"].tolist(),
        )
    return emb_dir, ann_paths["exp_A"], ann_paths["exp_B"]


def test_run_linear_classifiers_directory_mode(tmp_path):
    """Embeddings directory (post-split) is loaded and concatenated correctly."""
    emb_dir, ann_a, ann_b = _setup_dir_with_annotations(tmp_path)

    config = LinearClassifiersStepConfig(
        annotations=[
            AnnotationSource(experiment="exp_A", path=str(ann_a)),
            AnnotationSource(experiment="exp_B", path=str(ann_b)),
        ],
        tasks=[TaskSpec(task="infection_state")],
        use_scaling=True,
        split_train_data=0.8,
    )

    results = run_linear_classifiers(emb_dir, config, tmp_path / "out")

    assert not results.empty
    assert results.iloc[0]["task"] == "infection_state"
    assert results.iloc[0]["n_samples"] == 400  # 200 per experiment × 2
    assert (tmp_path / "out" / "metrics_summary.csv").exists()


def test_run_linear_classifiers_single_zarr_mode(tmp_path):
    """Single combined zarr (pre-split) is still accepted."""
    zarr_path = tmp_path / "embeddings.zarr"
    adata = _make_embeddings_zarr(zarr_path, experiment="exp_A")
    ann = _make_annotations(
        tmp_path,
        "exp_A",
        adata.obs["fov_name"].tolist(),
        adata.obs["t"].tolist(),
        adata.obs["track_id"].tolist(),
    )

    config = LinearClassifiersStepConfig(
        annotations=[AnnotationSource(experiment="exp_A", path=str(ann))],
        tasks=[TaskSpec(task="infection_state")],
        use_scaling=True,
        split_train_data=0.8,
    )

    results = run_linear_classifiers(zarr_path, config, tmp_path / "out")
    assert not results.empty


def test_run_linear_classifiers_fallback_join_no_id(tmp_path):
    """Annotation join falls back to (fov_name, t, track_id) when id column is absent."""
    zarr_path = tmp_path / "embeddings.zarr"
    adata = _make_embeddings_zarr(zarr_path, experiment="exp_A", use_id_col=False)

    assert "id" not in adata.obs.columns

    ann = _make_annotations(
        tmp_path,
        "exp_A",
        adata.obs["fov_name"].tolist(),
        adata.obs["t"].tolist(),
        adata.obs["track_id"].tolist(),
    )

    config = LinearClassifiersStepConfig(
        annotations=[AnnotationSource(experiment="exp_A", path=str(ann))],
        tasks=[TaskSpec(task="infection_state")],
        use_scaling=True,
        split_train_data=0.8,
    )

    results = run_linear_classifiers(zarr_path, config, tmp_path / "out")
    assert not results.empty
    assert results.iloc[0]["n_samples"] == 200


def test_run_linear_classifiers_multiple_tasks(tmp_path):
    """Multiple tasks produce one row each in results."""
    emb_dir, ann_a, ann_b = _setup_dir_with_annotations(tmp_path)

    config = LinearClassifiersStepConfig(
        annotations=[
            AnnotationSource(experiment="exp_A", path=str(ann_a)),
            AnnotationSource(experiment="exp_B", path=str(ann_b)),
        ],
        tasks=[
            TaskSpec(task="infection_state"),
            TaskSpec(task="organelle_state"),
        ],
        use_scaling=True,
        split_train_data=0.8,
    )

    results = run_linear_classifiers(emb_dir, config, tmp_path / "out")

    assert len(results) == 2
    assert set(results["task"].tolist()) == {"infection_state", "organelle_state"}


def test_run_linear_classifiers_marker_filter(tmp_path):
    """marker_filters restricts cells to those with matching marker."""
    emb_dir, ann_a, ann_b = _setup_dir_with_annotations(tmp_path)

    config = LinearClassifiersStepConfig(
        annotations=[
            AnnotationSource(experiment="exp_A", path=str(ann_a)),
            AnnotationSource(experiment="exp_B", path=str(ann_b)),
        ],
        tasks=[TaskSpec(task="infection_state", marker_filters=["Phase3D"])],
        use_scaling=True,
        split_train_data=0.8,
    )

    results = run_linear_classifiers(emb_dir, config, tmp_path / "out")

    assert not results.empty
    # Phase3D is half of each experiment → 100 per exp × 2 = 200
    assert results.iloc[0]["n_samples"] == 200


def test_run_linear_classifiers_missing_metadata_raises(tmp_path):
    """Raises ValueError when embeddings zarr lacks experiment/marker columns."""
    X = np.random.standard_normal((50, 8)).astype(np.float32)
    obs = pd.DataFrame({"fov_name": [f"A/1/FOV{i}" for i in range(50)]})
    obs["fov_name"] = obs["fov_name"].astype(object)
    obs.index = pd.Index([str(i) for i in range(50)], dtype=object)
    var = pd.DataFrame(index=pd.Index([str(i) for i in range(8)], dtype=object))
    zarr_path = tmp_path / "embeddings.zarr"
    ad.AnnData(X=X, obs=obs, var=var).write_zarr(zarr_path)

    config = LinearClassifiersStepConfig(
        annotations=[AnnotationSource(experiment="exp_A", path=str(tmp_path / "ann.csv"))],
        tasks=[TaskSpec(task="infection_state")],
    )

    with pytest.raises(ValueError, match="missing columns"):
        run_linear_classifiers(zarr_path, config, tmp_path / "out")


def test_run_linear_classifiers_unknown_marker_skipped(tmp_path):
    """If marker_filters matches no rows, task is skipped and result is empty."""
    emb_dir, ann_a, _ = _setup_dir_with_annotations(tmp_path)

    config = LinearClassifiersStepConfig(
        annotations=[AnnotationSource(experiment="exp_A", path=str(ann_a))],
        tasks=[TaskSpec(task="infection_state", marker_filters=["NonExistentMarker"])],
    )

    results = run_linear_classifiers(emb_dir, config, tmp_path / "out")
    assert results.empty
