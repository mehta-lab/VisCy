"""Tests for per-marker embedding-consistency QC."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.mmd.config import EmbeddingConsistencyConfig, MMDSettings
from dynaclr.evaluation.mmd.consistency import (
    corr_matrix_per_marker,
    mmd_matrix_per_marker,
    plot_consistency_matrix,
    run_consistency_qc,
)
from dynaclr.evaluation.paths import embedding_store


def _long_form() -> pd.DataFrame:
    """Minimal long-form pairwise MMD frame: 3 datasets, 2 markers, 2 conditions."""
    rows = []
    datasets = ["ds_a", "ds_b", "ds_c"]
    for marker in ["TOMM20", "Phase3D"]:
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                for cond, base in [("uninfected", 0.1), ("ZIKV", 0.2)]:
                    rows.append(
                        {
                            "marker": marker,
                            "exp_a": datasets[i],
                            "exp_b": datasets[j],
                            "condition": cond,
                            "mmd2": base + 0.05 * (j - i),
                            "p_value": 0.01,
                        }
                    )
    return pd.DataFrame(rows)


def test_matrix_is_symmetric_zero_diagonal():
    matrices = mmd_matrix_per_marker(_long_form())
    assert set(matrices) == {"TOMM20", "Phase3D"}
    for matrix in matrices.values():
        assert list(matrix.index) == list(matrix.columns) == ["ds_a", "ds_b", "ds_c"]
        values = matrix.to_numpy()
        assert np.allclose(np.diag(values), 0.0)
        assert np.allclose(values, values.T)


def test_matrix_averages_over_conditions():
    matrices = mmd_matrix_per_marker(_long_form())
    # ds_a vs ds_b: (0.1 + 0.05) and (0.2 + 0.05) -> mean 0.20
    assert matrices["TOMM20"].loc["ds_a", "ds_b"] == pytest.approx(0.20)


def test_plot_consistency_matrix_writes(tmp_path):
    matrix = mmd_matrix_per_marker(_long_form())["TOMM20"]
    out = tmp_path / "m.png"
    plot_consistency_matrix(matrix, "TOMM20", out)
    assert out.exists() and out.stat().st_size > 0


def _corr_dataset(experiment: str, mean_vec: np.ndarray, n_cells: int = 40) -> ad.AnnData:
    """Control-only dataset whose per-cell embeddings scatter tightly around ``mean_vec``."""
    rng = np.random.default_rng(abs(hash(experiment)) % (2**32))
    X = mean_vec[None, :] + rng.normal(scale=1e-3, size=(n_cells, mean_vec.size))
    obs = pd.DataFrame(
        {"experiment": experiment, "marker": "TOMM20", "perturbation": "uninfected"},
        index=[str(i) for i in range(n_cells)],
    )
    return ad.AnnData(X=X.astype(np.float32), obs=obs)


def test_corr_matrix_per_marker(tmp_path):
    # ds_a and ds_b share a mean direction (perfect +1); ds_c is its negation (-1).
    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    paths = []
    for name, vec in [("ds_a", base), ("ds_b", 2 * base), ("ds_c", -base)]:
        p = tmp_path / f"{name}.zarr"
        _corr_dataset(name, vec).write_zarr(str(p))
        paths.append(str(p))

    matrices = corr_matrix_per_marker(paths, {"perturbation": "uninfected"}, embedding_key=None)
    corr = matrices["TOMM20"]
    assert corr.loc["ds_a", "ds_b"] == pytest.approx(1.0, abs=1e-2)
    assert corr.loc["ds_a", "ds_c"] == pytest.approx(-1.0, abs=1e-2)


def _make_dataset_adata(experiment: str, shift: float, seed: int, n_features: int = 16) -> ad.AnnData:
    """One dataset's control+treated embeddings for a single marker, mean-shifted by ``shift``."""
    rng = np.random.default_rng(seed)
    rows = []
    embs = []
    for perturbation in ["uninfected", "ZIKV"]:
        for _ in range(60):
            embs.append(rng.normal(loc=shift, scale=1.0, size=n_features))
            rows.append(
                {
                    "experiment": experiment,
                    "marker": "TOMM20",
                    "perturbation": perturbation,
                    "hours_post_perturbation": 0.0,
                }
            )
    return ad.AnnData(X=np.stack(embs).astype(np.float32), obs=pd.DataFrame(rows))


def _write_datasets(datasets_root, model_family, run, ckpt) -> None:
    """Write ds_a/ds_b (similar) and ds_c (shifted) as per-marker zarrs in the canonical tree."""
    specs = [("ds_a", 0.0, 0), ("ds_b", 0.0, 1), ("ds_c", 5.0, 2)]
    for dataset, shift, seed in specs:
        store = embedding_store(dataset, model_family, run, ckpt, "TOMM20", datasets_root=datasets_root)
        store.parent.mkdir(parents=True, exist_ok=True)
        _make_dataset_adata(dataset, shift, seed).write_zarr(str(store))


def _qc_config(datasets_root, output_dir) -> EmbeddingConsistencyConfig:
    return EmbeddingConsistencyConfig(
        model_family="modelX",
        run="run1",
        ckpt_name="ckptA",
        datasets_root=str(datasets_root),
        output_dir=str(output_dir),
        obs_filter={"perturbation": "uninfected"},
        center_per_experiment=False,
        save_plots=True,
        mmd=MMDSettings(n_permutations=50, max_cells=200, min_cells=10, seed=0),
    )


def test_run_consistency_qc_end_to_end(tmp_path):
    datasets_root = tmp_path / "datasets"
    output_dir = tmp_path / "qc_out"
    _write_datasets(datasets_root, "modelX", "run1", "ckptA")

    df = run_consistency_qc(_qc_config(datasets_root, output_dir))

    assert set(df["exp_a"]) | set(df["exp_b"]) == {"ds_a", "ds_b", "ds_c"}
    assert (output_dir / "consistency_mmd_results.csv").exists()
    assert (output_dir / "consistency_mmd_matrix_TOMM20.csv").exists()
    assert (output_dir / "consistency_mmd_matrix_TOMM20.png").exists()
    assert (output_dir / "consistency_corr_matrix_TOMM20.csv").exists()
    assert (output_dir / "consistency_corr_matrix_TOMM20.png").exists()

    matrix = pd.read_csv(output_dir / "consistency_mmd_matrix_TOMM20.csv", index_col=0)
    # ds_c is mean-shifted; with center_per_experiment=False its off-diagonal MMD
    # to the two unshifted datasets must exceed the ds_a<->ds_b baseline.
    assert matrix.loc["ds_a", "ds_c"] > matrix.loc["ds_a", "ds_b"]
    assert matrix.loc["ds_b", "ds_c"] > matrix.loc["ds_a", "ds_b"]

    corr = pd.read_csv(output_dir / "consistency_corr_matrix_TOMM20.csv", index_col=0)
    assert list(corr.index) == list(corr.columns) == ["ds_a", "ds_b", "ds_c"]
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T)
    assert corr.to_numpy().min() >= -1.0 and corr.to_numpy().max() <= 1.0


def test_run_consistency_qc_requires_two_datasets(tmp_path):
    datasets_root = tmp_path / "datasets"
    store = embedding_store("ds_only", "modelX", "run1", "ckptA", "TOMM20", datasets_root=datasets_root)
    store.parent.mkdir(parents=True, exist_ok=True)
    _make_dataset_adata("ds_only", 0.0, 0).write_zarr(str(store))

    with pytest.raises(ValueError, match=">=2 dataset zarrs"):
        run_consistency_qc(_qc_config(datasets_root, tmp_path / "qc_out"))
