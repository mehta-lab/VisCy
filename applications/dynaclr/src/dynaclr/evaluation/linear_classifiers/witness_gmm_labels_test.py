"""Tests for Stage A: MMD-witness + GMM annotation generation."""

import anndata as ad
import numpy as np
import pandas as pd

from dynaclr.evaluation.evaluate_config import WitnessGmmExperiment, WitnessGmmLabelsConfig
from dynaclr.evaluation.linear_classifiers.witness_gmm_labels import (
    _well_prefix_mask,
    build_marker_annotation,
    generate_witness_gmm_annotation,
    obs_filter_mask,
)
from viscy_utils.evaluation.annotation import load_annotation_anndata


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


def _config(output_path, experiment="exp_A", embeddings_zarr="unused.zarr"):
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
        output_path=str(output_path),
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
    out = generate_witness_gmm_annotation(_config(tmp_path / "labels.parquet", embeddings_zarr=str(zarr_path)))
    assert out.exists()
    df = pd.read_parquet(out)
    assert "infection_state" in df.columns
    assert set(df["infection_state"].unique()) == {"infected", "uninfected"}


def test_annotation_joins_by_key_under_shuffle(tmp_path):
    """The Stage-A file is a valid annotation: labels land on the right cells even
    when the embeddings rows are shuffled (join by key, not row order)."""
    adata = _make_separable_embeddings()
    zarr_path = tmp_path / "embeddings.zarr"
    adata.write_zarr(zarr_path)
    out = generate_witness_gmm_annotation(_config(tmp_path / "labels.csv", embeddings_zarr=str(zarr_path)))

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
        output_path=str(tmp_path / "labels.csv"),
    )
    assert build_marker_annotation(adata, cfg.experiments, cfg) is None
