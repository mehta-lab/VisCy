"""Tests for selectively exporting pooled MMD representations to Zarr."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.mmd.config import ComparisonSpec, MMDPooledConfig
from dynaclr.evaluation.mmd.export_representation import (
    export_pooled_representation,
    load_pooled_representation_config,
)


def _write_store(path: Path, experiment: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    for condition, shift in (("uninfected", 0.0), ("DENV", 1.0)):
        for index in range(48):
            rows.append(
                {
                    "experiment": experiment,
                    "marker": "SEC61B",
                    "perturbation": condition,
                    "hours_post_perturbation": float(index % 6),
                    "qc": "drop" if index == 0 else "keep",
                }
            )
            embeddings.append(rng.normal(shift, 1.0, size=10))
    values = np.asarray(embeddings, dtype=np.float32)
    adata = ad.AnnData(X=values.copy(), obs=pd.DataFrame(rows))
    adata.obsm["X_existing"] = np.ones((len(adata), 2), dtype=np.float32)
    adata.uns["existing"] = {"preserved": True}
    adata.write_zarr(path)
    return values


def _config(paths: list[Path], output_dir: Path) -> MMDPooledConfig:
    return MMDPooledConfig(
        input_paths=[str(path) for path in paths],
        condition_aliases={"uninfected": ["uninfected", "mock"]},
        output_dir=str(output_dir),
        comparisons=[
            ComparisonSpec(
                cond_a="uninfected",
                cond_b="DENV",
                label="uninfected_vs_DENV",
            )
        ],
        obs_filter={"qc": "keep"},
    )


def test_export_pooled_representation_updates_only_named_slots(tmp_path: Path):
    paths = [tmp_path / "exp_a.zarr", tmp_path / "exp_b.zarr"]
    originals = [
        _write_store(paths[0], "exp_a", seed=1),
        _write_store(paths[1], "exp_b", seed=2),
    ]
    artifact_dir = tmp_path / "artifacts"

    manifest = export_pooled_representation(
        _config(paths, tmp_path / "unused_mmd_output"),
        artifact_dir=artifact_dir,
    )

    assert len(manifest) == 2
    assert len(set(manifest["n_dimensions"])) == 1
    for path, original in zip(paths, originals, strict=True):
        result = ad.read_zarr(path)
        assert np.array_equal(np.asarray(result.X), original)
        assert np.array_equal(result.obsm["X_existing"], np.ones((len(result), 2)))
        assert result.uns["existing"]["preserved"]
        scores = result.obsm["X_normalized_pca80"]
        assert scores.shape[0] == len(result)
        assert scores.shape[1] <= original.shape[1]
        metadata = result.uns["X_normalized_pca80"]
        assert metadata["representation"] == "control_mad_pca80"
        assert metadata["source_embedding_key"] == "X"
        assert metadata["condition_column"] == "perturbation"
        assert list(metadata["condition_aliases"]["uninfected"]) == ["uninfected", "mock"]
        assert metadata["fit_cells"] == 188
        assert metadata["n_components_by_marker"]["SEC61B"] == scores.shape[1]

    assert (artifact_dir / "zarr_export_manifest.csv").exists()
    assert (artifact_dir / "zarr_export_metadata.json").exists()
    assert (artifact_dir / "pca_models" / "SEC61B.npz").exists()
    assert (artifact_dir / "marker_pca_scree.pdf").read_bytes().startswith(b"%PDF")


def test_export_uses_each_markers_exact_global_pca80_width(tmp_path: Path):
    rng = np.random.default_rng(12)
    paths = []
    for marker, correlated in (("CORRELATED", True), ("ISOTROPIC", False)):
        rows = []
        values = []
        for condition, shift in (("uninfected", 0.0), ("DENV", 0.3)):
            for index in range(160):
                if correlated:
                    latent = rng.normal(shift, 1.0)
                    vector = latent + rng.normal(0.0, 0.02, size=10)
                else:
                    vector = rng.normal(shift, 1.0, size=10)
                values.append(vector)
                rows.append(
                    {
                        "experiment": f"exp_{marker.lower()}",
                        "marker": marker,
                        "perturbation": condition,
                        "hours_post_perturbation": float(index % 8),
                        "qc": "keep",
                    }
                )
        path = tmp_path / f"{marker}.zarr"
        ad.AnnData(X=np.asarray(values, dtype=np.float32), obs=pd.DataFrame(rows)).write_zarr(path)
        paths.append(path)

    config = _config(paths, tmp_path / "output")
    manifest = export_pooled_representation(config, artifact_dir=tmp_path / "artifacts")

    widths = {}
    for path in paths:
        result = ad.read_zarr(path)
        marker = str(result.obs["marker"].iloc[0])
        width = result.obsm["X_normalized_pca80"].shape[1]
        widths[marker] = width
        metadata = result.uns["X_normalized_pca80"]
        assert width == metadata["n_components_by_marker"][marker]
        assert width == metadata["n_dimensions"]
    assert widths["CORRELATED"] < widths["ISOTROPIC"]
    assert set(manifest["n_dimensions"]) == set(widths.values())


def test_export_pooled_representation_can_refuse_replacement(tmp_path: Path):
    path = tmp_path / "exp.zarr"
    _write_store(path, "exp", seed=3)
    config = _config([path], tmp_path / "output")
    export_pooled_representation(config, artifact_dir=tmp_path / "first")

    with pytest.raises(FileExistsError, match="already exists"):
        export_pooled_representation(
            config,
            artifact_dir=tmp_path / "second",
            overwrite=False,
        )


def test_default_key_rejects_misleading_representation_name(tmp_path: Path):
    path = tmp_path / "exp.zarr"
    _write_store(path, "exp", seed=4)
    config = _config([path], tmp_path / "output")
    config.representation.pca_variance = 0.90

    with pytest.raises(ValueError, match="reserved for control_mad"):
        export_pooled_representation(config)


def test_canonical_recipe_exposes_pooled_representation_section():
    recipe = Path(__file__).parents[1] / "configs/evaluation/recipes/witness_gmm_pooled_joint_pca80.yaml"

    config = load_pooled_representation_config(recipe)

    assert config.representation.normalization == "control_mad"
    assert config.representation.pca_variance == 0.80
    assert config.representation.pca_max_cells_per_dataset_class == 1900
    assert config.mmd.n_permutations == 1000
    assert config.mmd.balance_samples
