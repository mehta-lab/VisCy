"""Tests for Stage A: MMD-witness + GMM annotation generation."""

import anndata as ad
import numpy as np
import pandas as pd

from dynaclr.evaluation.evaluate_config import WitnessGmmExperiment, WitnessGmmLabelsConfig
from dynaclr.evaluation.linear_classifiers.witness_gmm_labels import (
    _balanced_reference_indices,
    _extract_embedding_matrix,
    _well_prefix_mask,
    compute_marker_scores,
    fit_pooled_witness_gmm,
    generate_witness_gmm_annotation,
    label_marker,
    obs_filter_mask,
)
from viscy_utils.evaluation.annotation import load_annotation_anndata


def build_marker_annotation(adata, experiments, config):
    """Test helper: compose the two-pass Stage-A labeling (all conditions
    treated as significant unless the raw MMD gate would exclude them).

    Mirrors the run-level flow for a single marker: score → per-condition MMD
    p-value → treat p ≤ threshold as significant → GMM-label. Returns None when
    the marker has no references or no condition survives both gates.
    """
    ms = compute_marker_scores(adata, experiments, config)
    if ms is None:
        return None
    significant = {c for c, p in ms.cond_pvalues.items() if p <= config.mmd_pvalue_threshold}
    result = label_marker(ms, significant, config)
    return result.frame if result is not None else None


def _make_separable_embeddings(
    n_per_well: int = 80,
    n_features: int = 16,
    experiment: str = "exp_A",
    marker: str = "viral_sensor",
) -> ad.AnnData:
    """Embeddings: control well A/1, perturbed well B/2 as a *mixture*.

    Control cells cluster near +2 on feature 0. The perturbed well is a mix: half
    the cells are remodeled (near -2), half still resemble control (near +2) —
    the realistic case the per-condition GMM is designed to gate. The witness
    scores of the perturbed cells are therefore bimodal and the GMM can split
    them into confident-remodeled vs unaffected. obs carries the annotation key
    columns (``id``, ``fov_name``) and a ``perturbation`` condition column.
    """
    rng = np.random.default_rng(0)
    wells = ["A/1"] * n_per_well + ["B/2"] * n_per_well
    total = len(wells)
    X = rng.standard_normal((total, n_features)).astype(np.float32) * 0.3
    X[:n_per_well, 0] += 2.0  # control — all near +2
    half = n_per_well // 2
    X[n_per_well : n_per_well + half, 0] -= 2.0  # perturbed & remodeled — near -2
    X[n_per_well + half :, 0] += 2.0  # perturbed but still control-like — near +2

    well_to_pert = {"A/1": "uninfected", "B/2": "DENV"}
    obs = pd.DataFrame(
        {
            "fov_name": [f"{w}/000000" for w in wells],
            "id": list(range(total)),
            "t": [i % 5 for i in range(total)],
            "track_id": list(range(total)),
            "experiment": [experiment] * total,
            "marker": [marker] * total,
            "perturbation": [well_to_pert[w] for w in wells],
            "hours_post_perturbation": [float(i % 5) * 24.0 for i in range(total)],
        }
    )
    for col in obs.select_dtypes("string").columns:
        obs[col] = obs[col].astype(object)
    obs.index = pd.Index([str(i) for i in range(total)], dtype=object)
    var = pd.DataFrame(index=pd.Index([str(i) for i in range(n_features)], dtype=object))
    return ad.AnnData(X=X, obs=obs, var=var)


def _config(output_dir, experiment="exp_A", embeddings_zarr="unused.zarr", annotation_format="csv"):
    return WitnessGmmLabelsConfig(
        experiments=[
            WitnessGmmExperiment(
                experiment=experiment,
                embeddings_zarr=embeddings_zarr,
                control_filter={"perturbation": "uninfected"},
                perturbed_filter={"perturbation": "DENV"},
            )
        ],
        marker_filters=["viral_sensor"],
        label_column="infection_state",
        class_map={"positive": "infected", "negative": "uninfected"},
        condition_column="perturbation",
        output_dir=str(output_dir),
        annotation_format=annotation_format,
    )


def test_well_prefix_mask_no_spurious_prefix_match():
    """'A/1' must not match 'A/10/...' — matching is on path components."""
    fov = pd.Series(["A/1/000000", "A/10/000000", "B/2/000000"])
    assert _well_prefix_mask(fov, ["A/1"]).tolist() == [True, False, False]


def test_obs_filter_mask_forms():
    """obs_filter_mask supports scalar, list, range window, and well-prefix routing."""
    obs = pd.DataFrame(
        {
            "fov_name": ["A/1/0", "A/2/0", "A/10/0", "A/2/1"],
            "perturbation": ["uninfected", "DENV", "DENV", "DENV"],
            "hours_post_perturbation": [3.0, 26.0, 8.0, 30.0],
        }
    )
    assert obs_filter_mask(obs, {"perturbation": "DENV"}).tolist() == [False, True, True, True]
    assert obs_filter_mask(obs, {"perturbation": ["uninfected"]}).tolist() == [True, False, False, False]
    assert obs_filter_mask(obs, {"hours_post_perturbation": {"ge": 24, "le": 36}}).tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert obs_filter_mask(obs, {"well": "A/2", "hours_post_perturbation": {"ge": 24}}).tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_build_marker_annotation_maps_class_vocabulary(tmp_path):
    """The annotation frame carries the named state column with the real vocabulary."""
    adata = _make_separable_embeddings()
    frame = build_marker_annotation(
        adata, _config(tmp_path / "labels.csv").experiments, _config(tmp_path / "labels.csv")
    )
    assert frame is not None
    assert "infection_state" in frame.columns
    assert set(frame["infection_state"].unique()) == {"infected", "uninfected"}
    # All control-well cells labeled uninfected.
    ctrl = frame[frame["fov_name"].str.startswith("A/1")]
    assert (ctrl["infection_state"] == "uninfected").all()
    # The perturbed well contributes BOTH classes: confident positives, plus the
    # cells the gate is confidently negative about (pre-onset / bystanders). Before
    # the symmetric-threshold fix only positives were kept here, so every negative
    # came from a different well than the positives.
    pert = frame[frame["fov_name"].str.startswith("B/2")]
    assert set(pert["infection_state"].unique()) == {"infected", "uninfected"}
    # Key columns present for the annotation join.
    assert {"experiment", "fov_name", "id"}.issubset(frame.columns)


def test_generate_writes_annotation_file(tmp_path):
    """generate_witness_gmm_annotation writes a parquet/csv annotation file."""
    zarr_path = tmp_path / "embeddings.zarr"
    _make_separable_embeddings().write_zarr(zarr_path)
    out = generate_witness_gmm_annotation(
        _config(tmp_path / "ckpt", embeddings_zarr=str(zarr_path), annotation_format="parquet")
    )
    assert out.exists()
    assert out == tmp_path / "ckpt" / "labels" / "viral_sensor_infection_state.parquet"
    df = pd.read_parquet(out)
    assert "infection_state" in df.columns
    assert set(df["infection_state"].unique()) == {"infected", "uninfected"}
    # Full tracking metadata carried through (not just exp/fov/id/t/state).
    assert {"track_id", "marker", "perturbation"}.issubset(df.columns)
    # Per-cell provenance: raw witness score + GMM posterior (confidence weight).
    assert {"witness_score", "gmm_posterior"}.issubset(df.columns)
    # Control cells (uninfected) are the clean reference → posterior 1.0; positives ∈ (0, 1].
    ctrl = df["infection_state"] == "uninfected"
    assert (df.loc[ctrl, "gmm_posterior"] == 1.0).all()
    assert (df.loc[~ctrl, "gmm_posterior"] > 0).all() and (df.loc[~ctrl, "gmm_posterior"] <= 1.0).all()
    # Diagnostic plots written alongside the labels.
    plots = tmp_path / "ckpt" / "labels" / "plots"
    assert (plots / "witness_gmm_viral_sensor_DENV.png").exists()
    assert (plots / "mmd_null_viral_sensor_DENV.png").exists()
    assert (plots / "remodeling_vs_time_viral_sensor.png").exists()
    # Population-level provenance sidecar.
    mmd = pd.read_csv(tmp_path / "ckpt" / "labels" / "viral_sensor_infection_state_mmd.csv")
    assert {
        "marker",
        "condition",
        "mmd2",
        "p_raw",
        "p_adjusted",
        "mmd_significant",
        "embedding_key",
        "reference_sampling",
        "gmm_fit_population",
        "gmm_covariance_type",
    }.issubset(mmd.columns)


def test_annotation_joins_by_key_under_shuffle(tmp_path):
    """The Stage-A file is a valid annotation: labels land on the right cells even
    when the embeddings rows are shuffled (join by key, not row order)."""
    adata = _make_separable_embeddings()
    zarr_path = tmp_path / "embeddings.zarr"
    adata.write_zarr(zarr_path)
    out = generate_witness_gmm_annotation(_config(tmp_path / "ckpt", embeddings_zarr=str(zarr_path)))

    # Shuffle the embedding rows, then join the annotation back by key.
    rng = np.random.default_rng(3)
    perm = rng.permutation(adata.n_obs)
    shuffled = adata[perm].copy()
    joined = load_annotation_anndata(shuffled, str(out), "infection_state")

    # The join is by key, so each cell must carry the label Stage A gave THAT cell.
    # Compare against the annotation file itself rather than assuming a well maps to
    # one class — perturbed wells legitimately carry both (confident positives plus
    # pre-onset/bystander negatives).
    labeled = joined.obs["infection_state"].notna()
    expected = pd.read_csv(out).set_index(["fov_name", "id"])["infection_state"]
    got = joined.obs.loc[labeled].copy()
    got["fov_name"] = got["fov_name"].astype(object).str.strip("/")
    keys = list(zip(got["fov_name"], got["id"], strict=False))
    assert (got["infection_state"].to_numpy() == expected.loc[keys].to_numpy()).all()
    # Control cells are negative by well identity, so that direction still holds.
    ctrl = got["fov_name"].str.startswith("A/1")
    assert (got.loc[ctrl, "infection_state"] == "uninfected").all()


def test_build_marker_annotation_none_when_reference_missing(tmp_path):
    """No control or no perturbed reference → None (marker skipped)."""
    adata = _make_separable_embeddings()
    cfg = WitnessGmmLabelsConfig(
        experiments=[
            WitnessGmmExperiment(
                experiment="exp_A",
                embeddings_zarr="unused.zarr",
                control_filter={"perturbation": "nonexistent"},
                perturbed_filter={"perturbation": "DENV"},
            )
        ],
        marker_filters=["viral_sensor"],
        label_column="infection_state",
        class_map={"positive": "infected", "negative": "uninfected"},
        output_dir=str(tmp_path / "ckpt"),
    )
    assert build_marker_annotation(adata, cfg.experiments, cfg) is None


def test_build_marker_annotation_mmd_gate_skips_nonsignificant(tmp_path):
    """When control and perturbed clouds are indistinguishable, the MMD
    significance gate skips the condition (no positives → None)."""
    rng = np.random.default_rng(0)
    n = 120
    wells = ["A/1"] * n + ["B/2"] * n
    total = len(wells)
    # Both wells drawn from the SAME distribution — no real separation.
    X = rng.standard_normal((total, 16)).astype(np.float32)
    well_to_pert = {"A/1": "uninfected", "B/2": "DENV"}
    obs = pd.DataFrame(
        {
            "fov_name": [f"{w}/000000" for w in wells],
            "id": list(range(total)),
            "t": [i % 5 for i in range(total)],
            "track_id": list(range(total)),
            "experiment": ["exp_A"] * total,
            "marker": ["viral_sensor"] * total,
            "perturbation": [well_to_pert[w] for w in wells],
        }
    )
    for col in obs.select_dtypes("string").columns:
        obs[col] = obs[col].astype(object)
    obs.index = pd.Index([str(i) for i in range(total)], dtype=object)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[str(i) for i in range(16)]))

    cfg = _config(tmp_path / "labels.csv")
    assert build_marker_annotation(adata, cfg.experiments, cfg) is None


def test_time_matched_witness_scores_all_cells(tmp_path):
    """Time-matched witness (witness_time_bin_hours set) scores every cell and
    still separates the classes — labels land as with the pooled reference."""
    adata = _make_separable_embeddings()
    cfg = _config(tmp_path / "ckpt")
    cfg = cfg.model_copy(update={"witness_time_bin_hours": 24.0})
    ms = compute_marker_scores(adata, cfg.experiments, cfg)
    assert ms is not None
    # Every cell got a finite witness score (no bin left a cell unscored).
    assert np.isfinite(ms.scores).all()
    assert ms.scores.shape[0] == adata.n_obs
    # The class signal survives time-matching: control cells lean positive-score
    # (control reference), the remodeled half of perturbed leans negative.
    frame = build_marker_annotation(adata, cfg.experiments, cfg)
    assert frame is not None
    assert set(frame["infection_state"].unique()) == {"infected", "uninfected"}


def test_balanced_control_perturbed_gmm_fit_uses_both_populations(tmp_path):
    """The pooled mode fits equal control/perturbed samples but scores the condition."""
    adata = _make_separable_embeddings(n_per_well=80)
    cfg = _config(tmp_path / "ckpt").model_copy(update={"gmm_fit_population": "balanced_control_perturbed"})
    marker_scores = compute_marker_scores(adata, cfg.experiments, cfg)
    assert marker_scores is not None
    result = label_marker(marker_scores, {"DENV"}, cfg)
    assert result is not None
    gmm = result.cond_gmm["DENV"]
    assert gmm.n_fit_samples == 160
    assert len(gmm.posterior) == 80
    assert gmm.separated


def test_joint_control_perturbed_gmm_fit_uses_every_cell(tmp_path):
    """Literal joint mode concatenates all control and condition witness scores."""
    adata = _make_separable_embeddings(n_per_well=60)
    cfg = _config(tmp_path / "ckpt").model_copy(update={"gmm_fit_population": "joint_control_perturbed"})
    marker_scores = compute_marker_scores(adata, cfg.experiments, cfg)
    assert marker_scores is not None
    result = label_marker(marker_scores, {"DENV"}, cfg)
    assert result is not None
    gmm = result.cond_gmm["DENV"]
    assert gmm.n_fit_samples == 120
    assert len(gmm.posterior) == 60


def test_joint_gmm_applies_symmetric_gate_to_control_cells(tmp_path):
    """Joint mode does not force a perturbed-like control cell negative."""
    adata = _make_separable_embeddings(n_per_well=80)
    cfg = _config(tmp_path / "ckpt").model_copy(
        update={
            "gmm_fit_population": "joint_control_perturbed",
            "gmm_covariance_type": "tied",
        }
    )
    marker_scores = compute_marker_scores(adata, cfg.experiments, cfg)
    assert marker_scores is not None
    remodeled = marker_scores.scores[marker_scores.perturbed_mask]
    marker_scores.scores[0] = remodeled.min()

    result = label_marker(marker_scores, {"DENV"}, cfg)

    assert result is not None
    forced_outlier = result.frame[result.frame["id"] == 0]
    assert forced_outlier["infection_state"].tolist() == ["infected"]
    assert result.control_pos[0]


def test_function_api_fits_joint_tied_pooled_teacher(tmp_path):
    """The script-level API returns annotations without CLI or file I/O."""
    adata = _make_separable_embeddings(n_per_well=60)
    cfg = _config(tmp_path / "ckpt").model_copy(
        update={
            "gmm_fit_population": "joint_control_perturbed",
            "gmm_covariance_type": "tied",
            "reference_sampling": "balanced_by_experiment",
            "max_reference_cells": 50,
        }
    )
    fit = fit_pooled_witness_gmm(adata, cfg)

    assert not fit.annotations.empty
    assert set(fit.annotations["infection_state"]) == {
        "infected",
        "uninfected",
    }
    result = fit.marker_labels["viral_sensor"]
    assert result is not None
    assert result.cond_gmm["DENV"].gmm.covariance_type == "tied"


def test_stored_normalized_pca_representation_is_used(tmp_path):
    """Stage A can consume persisted normalized PCA scores without touching raw X."""
    adata = _make_separable_embeddings()
    adata.obsm["X_normalized_pca80"] = np.asarray(adata.X).copy()
    adata.X = np.zeros_like(np.asarray(adata.X))
    cfg = _config(tmp_path / "ckpt").model_copy(
        update={
            "embedding_key": "X_normalized_pca80",
            "gmm_fit_population": "joint_control_perturbed",
            "gmm_covariance_type": "tied",
        }
    )
    marker_scores = compute_marker_scores(adata, cfg.experiments, cfg)
    assert marker_scores is not None
    assert np.isfinite(marker_scores.scores).all()
    result = label_marker(marker_scores, {"DENV"}, cfg)
    assert result is not None
    assert result.cond_gmm["DENV"].gmm.covariance_type == "tied"


def test_balanced_reference_sampling_is_equal_per_experiment(tmp_path):
    """A feasible cap gives every experiment equal control and perturbed references."""
    first = _make_separable_embeddings(n_per_well=50, experiment="exp_A")
    second = _make_separable_embeddings(n_per_well=35, experiment="exp_B")
    pooled = ad.concat([first, second], index_unique="-")
    sources = [
        WitnessGmmExperiment(
            experiment=name,
            embeddings_zarr="unused.zarr",
            control_filter={"perturbation": "uninfected"},
            perturbed_filter={"perturbation": "DENV"},
        )
        for name in ("exp_A", "exp_B")
    ]
    control = pooled.obs["perturbation"].to_numpy() == "uninfected"
    perturbed = pooled.obs["perturbation"].to_numpy() == "DENV"
    control_idx, perturbed_idx = _balanced_reference_indices(
        pooled.obs,
        control,
        perturbed,
        sources,
        max_per_experiment_class=30,
        rng=np.random.default_rng(7),
    )
    experiments = pooled.obs["experiment"].astype(str).to_numpy()
    for name in ("exp_A", "exp_B"):
        assert np.sum(experiments[control_idx] == name) == 30
        assert np.sum(experiments[perturbed_idx] == name) == 30

    assert _extract_embedding_matrix(first, None).shape == first.X.shape
