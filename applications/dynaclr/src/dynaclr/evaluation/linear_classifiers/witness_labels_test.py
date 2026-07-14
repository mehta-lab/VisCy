"""Tests for MMD-witness weak labeling of linear classifiers."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from dynaclr.evaluation.evaluate_config import (
    LinearClassifiersStepConfig,
    WitnessLabelSource,
    WitnessSettings,
)
from dynaclr.evaluation.linear_classifiers.orchestrated import run_linear_classifiers
from dynaclr.evaluation.linear_classifiers.witness_labels import (
    _well_prefix_mask,
    build_witness_labels,
)


def _make_separable_embeddings(
    path: Path | None,
    n_per_well: int = 60,
    n_features: int = 16,
    experiment: str = "exp_A",
    marker: str = "Phase3D",
) -> ad.AnnData:
    """Embeddings with control well A/1 and perturbed well B/2 well-separated in feature space.

    Control cells cluster near +2 on feature 0, perturbed near -2, so the
    witness cleanly assigns positive scores to control and negative to
    perturbed. An unrelated well C/3 sits at the origin (ambiguous).
    """
    rng = np.random.default_rng(0)
    wells = ["A/1"] * n_per_well + ["B/2"] * n_per_well + ["C/3"] * n_per_well
    total = len(wells)
    X = rng.standard_normal((total, n_features)).astype(np.float32) * 0.3
    X[:n_per_well, 0] += 2.0  # control
    X[n_per_well : 2 * n_per_well, 0] -= 2.0  # perturbed
    # C/3 stays near origin

    obs = pd.DataFrame(
        {
            "fov_name": [f"{w}/000000" for w in wells],
            "t": [i % 5 for i in range(total)],
            "track_id": list(range(total)),
            "experiment": [experiment] * total,
            "marker": [marker] * total,
            "hours_post_perturbation": [float(i % 5) * 24.0 for i in range(total)],
        }
    )
    # pandas 3 defaults string columns to ArrowStringArray, which anndata's
    # zarr writer cannot serialize — cast to object (matches orchestrated_test).
    for col in obs.select_dtypes("string").columns:
        obs[col] = obs[col].astype(object)
    obs.index = pd.Index([str(i) for i in range(total)], dtype=object)
    var = pd.DataFrame(index=pd.Index([str(i) for i in range(n_features)], dtype=object))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if path is not None:
        adata.write_zarr(path)
    return adata


def test_well_prefix_mask_no_spurious_prefix_match():
    """'A/1' must not match 'A/10/...' — matching is on path components, not string prefix."""
    fov = pd.Series(["A/1/000000", "A/10/000000", "B/2/000000"])
    mask = _well_prefix_mask(fov, ["A/1"])
    assert mask.tolist() == [True, False, False]


def test_build_witness_labels_separates_control_and_perturbed():
    """Witness scores gate the two separated clusters into control/perturbed."""
    adata = _make_separable_embeddings(None)
    labels = build_witness_labels(
        adata,
        [WitnessLabelSource(experiment="exp_A", control_wells=["A/1"], perturbed_wells=["B/2"])],
        WitnessSettings(dead_zone=0.0),
    )
    # Every cell labeled (dead_zone=0), two classes present.
    assert labels.n_obs == adata.n_obs
    col = labels.obs["witness_state"]
    assert set(col.unique()) == {"control", "perturbed"}

    # Control well cells score as control; perturbed well cells as perturbed.
    is_ctrl_well = labels.obs["fov_name"].str.startswith("A/1")
    is_pert_well = labels.obs["fov_name"].str.startswith("B/2")
    assert (col[is_ctrl_well] == "control").mean() > 0.95
    assert (col[is_pert_well] == "perturbed").mean() > 0.95


def test_build_witness_labels_dead_zone_drops_ambiguous():
    """A positive dead-zone drops the lowest-|score| cells (the ambiguous C/3 cluster)."""
    adata = _make_separable_embeddings(None)
    labels = build_witness_labels(
        adata,
        [WitnessLabelSource(experiment="exp_A", control_wells=["A/1"], perturbed_wells=["B/2"])],
        WitnessSettings(dead_zone=0.3),
    )
    assert labels.n_obs < adata.n_obs
    # Dropped cells should be disproportionately the near-origin C/3 well.
    kept_wells = labels.obs["fov_name"].str.split("/").str[0]
    assert (kept_wells == "C").mean() < (1.0 / 3.0)


def test_build_witness_labels_empty_when_reference_missing():
    """No control or no perturbed cells → empty AnnData (skipped downstream)."""
    adata = _make_separable_embeddings(None)
    labels = build_witness_labels(
        adata,
        [WitnessLabelSource(experiment="exp_A", control_wells=["Z/9"], perturbed_wells=["B/2"])],
        WitnessSettings(),
    )
    assert labels.n_obs == 0


def test_run_linear_classifiers_witness_mode(tmp_path):
    """End-to-end witness path: config → weak labels → trained classifier + metrics."""
    zarr_path = tmp_path / "embeddings.zarr"
    _make_separable_embeddings(zarr_path)

    config = LinearClassifiersStepConfig(
        label_source="witness",
        witness_labels=[
            WitnessLabelSource(experiment="exp_A", control_wells=["A/1"], perturbed_wells=["B/2"]),
        ],
        witness=WitnessSettings(marker_filters=["Phase3D"], dead_zone=0.1),
        split_train_data=0.8,
    )

    results = run_linear_classifiers(zarr_path, config, tmp_path / "out")

    assert len(results) == 1
    assert results.iloc[0]["task"] == "witness_state"
    assert results.iloc[0]["marker_filter"] == "Phase3D"
    # Separable control/perturbed clusters → classifier well above chance.
    # (The ambiguous C/3 well, sign-labeled, caps this below 1.0.)
    assert results.iloc[0]["val_accuracy"] > 0.8
    assert (tmp_path / "out" / "metrics_summary.csv").exists()
    assert (tmp_path / "out" / "witness_state_summary.pdf").exists()
