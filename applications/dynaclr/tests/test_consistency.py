"""Tests for per-marker embedding-consistency QC."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.mmd.config import EmbeddingConsistencyConfig, MMDSettings
from dynaclr.evaluation.mmd.consistency import (
    _dataset_mean_embedding,
    _frechet_distance,
    corr_matrix_per_marker,
    mmd_matrix_per_marker,
    plot_consistency_matrix,
    run_consistency_qc,
    temporal_std_per_marker,
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


def _make_dataset_adata(
    experiment: str, shift: float, seed: int, n_features: int = 16, microscope: str = ""
) -> ad.AnnData:
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
                    "microscope": microscope,
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


def test_dataset_mean_embedding_hpi_debiases_time_sampling():
    """HPI mean-of-means weights each time bin equally; grand mean is skewed by oversampling."""
    d = 3
    # Bin 0 (hpi<2): 100 cells at value 0. Bin 1 (2<=hpi<4): 1 cell at value 10.
    emb = np.vstack([np.zeros((100, d)), np.full((1, d), 10.0)]).astype(np.float32)
    hpi = np.concatenate([np.full(100, 0.5), np.full(1, 3.0)])

    grand = _dataset_mean_embedding(emb, hpi, hpi_bin_hours=None)
    binned = _dataset_mean_embedding(emb, hpi, hpi_bin_hours=2.0)

    # Grand mean ~ 10/101 ≈ 0.099 (dominated by the 100 cells).
    assert grand[0] == pytest.approx(10.0 / 101, abs=1e-4)
    # Mean-of-means: bin0=0, bin1=10 -> average 5.0 (each bin counts once).
    assert binned[0] == pytest.approx(5.0, abs=1e-4)


def test_dataset_mean_embedding_requires_hpi_when_binning():
    with pytest.raises(KeyError, match="hours_post_perturbation"):
        _dataset_mean_embedding(np.zeros((5, 3), dtype=np.float32), None, hpi_bin_hours=2.0)


def _drift_dataset(experiment: str, drift: float, seed: int, n_bins: int = 5, per_bin: int = 20) -> ad.AnnData:
    """A dataset whose control-cell embedding shifts by `drift` per HPI bin (0 = stable)."""
    d = 4
    rng = np.random.default_rng(seed)
    embs, rows = [], []
    for b in range(n_bins):
        center = drift * b
        for _ in range(per_bin):
            embs.append(rng.normal(loc=center, scale=0.01, size=d))
            rows.append(
                {
                    "experiment": experiment,
                    "marker": "TOMM20",
                    "perturbation": "uninfected",
                    "hours_post_perturbation": float(b * 2),  # bin width 2h -> one bin per b
                }
            )
    return ad.AnnData(X=np.stack(embs).astype(np.float32), obs=pd.DataFrame(rows))


def test_temporal_std_higher_for_drifting_dataset(tmp_path):
    """A dataset whose centroid drifts across HPI bins gets a larger temporal_std than a stable one."""
    stable = tmp_path / "stable.zarr"
    drifting = tmp_path / "drifting.zarr"
    _drift_dataset("ds_stable", drift=0.0, seed=0).write_zarr(str(stable))
    _drift_dataset("ds_drift", drift=1.0, seed=1).write_zarr(str(drifting))

    tables = temporal_std_per_marker(
        [str(stable), str(drifting)], {"perturbation": "uninfected"}, None, hpi_bin_hours=2.0
    )
    t = tables["TOMM20"]
    assert t.loc["ds_drift", "n_bins"] == 5
    assert t.loc["ds_stable", "temporal_std"] < 0.05  # ~0, only noise
    assert t.loc["ds_drift", "temporal_std"] > t.loc["ds_stable", "temporal_std"]  # drift dominates


def test_frechet_distance_uses_both_moments():
    """Fréchet distance is 0 for identical Gaussians and grows with BOTH a mean shift and a variance change."""
    d = 5
    mu0 = np.zeros(d)
    cov0 = np.eye(d)
    # Identical → 0.
    assert _frechet_distance(mu0, cov0, mu0, cov0) == pytest.approx(0.0, abs=1e-9)
    # Mean shift only (same covariance) → equals the Euclidean mean distance.
    mu1 = np.full(d, 2.0)
    assert _frechet_distance(mu0, cov0, mu1, cov0) == pytest.approx(np.linalg.norm(mu1 - mu0), abs=1e-6)
    # Variance change only (same mean) → strictly positive (Pearson-of-means would miss this).
    cov_wide = np.eye(d) * 4.0
    var_only = _frechet_distance(mu0, cov0, mu0, cov_wide)
    assert var_only > 1e-6
    # Symmetry.
    assert _frechet_distance(mu0, cov0, mu1, cov_wide) == pytest.approx(
        _frechet_distance(mu1, cov_wide, mu0, cov0), abs=1e-6
    )


def _write_scoped_datasets(datasets_root, model_family, run, ckpt) -> None:
    """Write 3 mantis_v2 + 1 mantis_v1 datasets (mirrors the real 9-v2 / 1-v1 cohort shape)."""
    specs = [
        ("ds_v2a", 0.0, 0, "mantis_v2"),
        ("ds_v2b", 0.0, 1, "mantis_v2"),
        ("ds_v2c", 0.2, 2, "mantis_v2"),
        ("ds_v1a", 5.0, 3, "mantis_v1"),  # lone v1 → within_v1 degenerate, skipped
    ]
    for dataset, shift, seed, scope in specs:
        store = embedding_store(dataset, model_family, run, ckpt, "TOMM20", datasets_root=datasets_root)
        store.parent.mkdir(parents=True, exist_ok=True)
        _make_dataset_adata(dataset, shift, seed, microscope=scope).write_zarr(str(store))


def test_run_consistency_qc_split_by_microscope(tmp_path):
    """split_by='microscope' emits a within-v2 block + a v1xv2 cross block; lone v1 within is skipped."""
    datasets_root = tmp_path / "datasets"
    output_dir = tmp_path / "qc_out"
    _write_scoped_datasets(datasets_root, "modelX", "run1", "ckptA")

    cfg = _qc_config(datasets_root, output_dir)
    cfg = cfg.model_copy(update={"split_by": "microscope"})
    run_consistency_qc(cfg)

    # Top-level full pairwise record still written.
    assert (output_dir / "consistency_mmd_results.csv").exists()

    # within-v2 block: 3 datasets → written.
    within_v2 = output_dir / "within_mantis_v2"
    assert within_v2.is_dir()
    assert (within_v2 / "consistency_mmd_matrix_TOMM20_mantis_v2.csv").exists()
    assert (within_v2 / "consistency_frechet_matrix_TOMM20_mantis_v2.csv").exists()
    m = pd.read_csv(within_v2 / "consistency_mmd_matrix_TOMM20_mantis_v2.csv", index_col=0)
    assert set(m.index) == {"ds_v2a", "ds_v2b", "ds_v2c"}  # no v1 dataset leaks in

    # lone-v1 within block: degenerate (1 dataset) → NOT written.
    assert not (output_dir / "within_mantis_v1").exists()

    # cross v1×v2 block: written, and contains ONLY cross pairs (v1a vs each v2).
    cross = output_dir / "cross_mantis_v1__mantis_v2"
    assert cross.is_dir()
    cross_df = pd.read_csv(cross / "consistency_mmd_results_mantis_v1__mantis_v2.csv")
    pairs = {frozenset((a, b)) for a, b in zip(cross_df["exp_a"], cross_df["exp_b"])}
    assert all("ds_v1a" in p for p in pairs)  # every kept pair straddles the v1/v2 boundary
    assert frozenset(("ds_v2a", "ds_v2b")) not in pairs  # within-v2 pairs excluded from cross
