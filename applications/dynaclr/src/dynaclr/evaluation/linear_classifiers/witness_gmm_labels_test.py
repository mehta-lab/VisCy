"""Tests for Stage A: MMD-witness + GMM annotation generation."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.evaluate_config import WitnessGmmExperiment, WitnessGmmLabelsConfig
from dynaclr.evaluation.linear_classifiers.witness_gmm_labels import (
    _hpi_bin_edges,
    _well_prefix_mask,
    compute_marker_scores,
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


def _config(output_dir, experiment="exp_A", embeddings_zarr="unused.zarr", annotation_format="csv", **overrides):
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
        **overrides,
    )


def test_hpi_bin_edges_include_maximum_on_boundary():
    hpi = np.asarray([0.0, 2.0, 4.0])

    edges = _hpi_bin_edges(hpi, width=2.0)

    assert edges.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert any(lo <= hpi.max() < hi for lo, hi in zip(edges[:-1], edges[1:]))


@pytest.mark.parametrize("field", ["witness_time_bin_hours", "mmd_hpi_bin_hours"])
@pytest.mark.parametrize("width", [0.0, -1.0, np.inf, np.nan])
def test_hpi_bin_width_must_be_finite_and_positive(tmp_path, field, width):
    with pytest.raises(ValueError, match=field):
        _config(tmp_path, **{field: width})


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
    # Every *labeled* perturbed-well cell is a confident positive (infected); the
    # unaffected half of the perturbed well is dropped (not in the frame).
    pert = frame[frame["fov_name"].str.startswith("B/2")]
    assert len(pert) > 0
    assert (pert["infection_state"] == "infected").all()
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
    assert {"marker", "condition", "mmd2", "p_raw", "p_adjusted", "mmd_significant"}.issubset(mmd.columns)


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

    # Every control-well cell that received a label reads "uninfected"; perturbed "infected".
    labeled = joined.obs["infection_state"].notna()
    ctrl = joined.obs["fov_name"].astype(object).str.strip("/").str.startswith("A/1")
    pert = joined.obs["fov_name"].astype(object).str.strip("/").str.startswith("B/2")
    assert (joined.obs.loc[labeled & ctrl, "infection_state"] == "uninfected").all()
    assert (joined.obs.loc[labeled & pert, "infection_state"] == "infected").all()


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
