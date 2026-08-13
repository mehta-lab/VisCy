"""Unit tests for the eval-config generators and per-model output_dir routing.

Covers the two functions the witness-GMM eval step depends on:

- ``_embeddings_dir`` — resolve the canonical ``{ckpt}/embeddings`` dir from the
  matrix ``embeddings_glob`` (else fall back to ``{output_dir}/embeddings``).
- ``_generate_witness_gmm_yaml`` — emit one Stage-A YAML per label source with
  the embeddings zarr, output_dir, and reference filters wired through.

Plus the ``output_dir`` derivation used by ``prepare_configs`` in the matrix
``eval_from_embeddings`` flow: the glob's ``{ckpt}`` (parent of the embeddings
dir) is where labels/classifiers land per model.
"""

from pathlib import Path

import pytest
import yaml

from dynaclr.evaluation.evaluate import (
    _embeddings_dir,
    _generate_witness_gmm_yaml,
    _resolve_lc_annotations,
)
from dynaclr.evaluation.evaluate_config import EvaluationConfig

_CKPT = "/data/2026_04_14_A549_SEC61B_DENV/2-phenotyping/predictions/DynaCLR/run/epoch105-step84800"
_GLOB = f"{_CKPT}/embeddings/*.zarr"


def _witness_eval_cfg(output_dir: str) -> EvaluationConfig:
    return EvaluationConfig(
        training_config="/cfg/DynaCLR.yml",
        output_dir=output_dir,
        steps=["witness_gmm", "linear_classifiers"],
        witness_gmm={
            "label_sources": [
                {
                    "marker": "SEC61B",
                    "label_column": "organelle_remodeling_state",
                    "class_map": {"positive": "remodel", "negative": "noremodel"},
                },
                {
                    "marker": "pAL40",
                    "label_column": "infection_state",
                    "class_map": {"positive": "infected", "negative": "uninfected"},
                },
            ],
            "experiments": ["2026_04_14_A549_SEC61B_DENV"],
            "control_filter": {"perturbation": "uninfected"},
            "perturbed_filter": {"perturbation": ["DENV"]},
            "embedding_key": "X_normalized_pca80",
            "reference_sampling": "balanced_by_experiment",
            "gmm_fit_population": "joint_control_perturbed",
            "gmm_covariance_type": "tied",
            "max_reference_cells": 1900,
        },
        linear_classifiers={
            # annotations intentionally empty — witness_gmm auto-fills them.
            "tasks": [
                {"task": "organelle_remodeling_state", "marker_filters": ["SEC61B"]},
                {"task": "infection_state", "marker_filters": ["pAL40"]},
            ],
            "split_groups_by": ["experiment", "fov_name", "track_id"],
        },
    )


def test_embeddings_dir_from_glob_ignores_output_dir():
    """With a glob, the embeddings dir is the glob parent, not {output_dir}/embeddings."""
    got = _embeddings_dir(output_dir=Path("/some/static/out"), embeddings_glob=_GLOB)
    assert str(got) == f"{_CKPT}/embeddings"


def test_embeddings_dir_fallback_without_glob():
    """No glob (legacy predict flow) falls back to {output_dir}/embeddings."""
    got = _embeddings_dir(output_dir=Path("/some/static/out"), embeddings_glob=None)
    assert str(got) == "/some/static/out/embeddings"


def test_output_dir_derived_from_glob_is_ckpt_parent():
    """The per-model output_dir is the glob's {ckpt} tree (parent of embeddings dir)."""
    derived = _embeddings_dir(output_dir=Path("/static"), embeddings_glob=_GLOB).parent
    assert str(derived) == _CKPT


def test_generate_witness_gmm_yaml_one_per_source(tmp_path):
    """One Stage-A YAML per label source, pointing at canonical {ckpt}/embeddings/{marker}.zarr."""
    (tmp_path / "configs").mkdir()
    cfg = _witness_eval_cfg(output_dir=str(tmp_path))

    paths = _generate_witness_gmm_yaml(cfg, output_dir=tmp_path, embeddings_glob=_GLOB)

    assert [p.name for p in paths] == ["witness_gmm_SEC61B.yaml", "witness_gmm_pAL40.yaml"]

    sec = yaml.safe_load(paths[0].read_text())["witness_gmm_labels"]
    assert sec["marker_filters"] == ["SEC61B"]
    assert sec["label_column"] == "organelle_remodeling_state"
    assert sec["class_map"] == {"positive": "remodel", "negative": "noremodel"}
    # Embeddings zarr resolves to the canonical {ckpt}/embeddings dir from the glob,
    # NOT {output_dir}/embeddings (which the matrix flow never populates).
    exp = sec["experiments"][0]
    assert exp["experiment"] == "2026_04_14_A549_SEC61B_DENV"
    assert exp["embeddings_zarr"] == f"{_CKPT}/embeddings/SEC61B.zarr"
    assert exp["control_filter"] == {"perturbation": "uninfected"}
    assert exp["perturbed_filter"] == {"perturbation": ["DENV"]}
    # Stage A appends labels/ under output_dir → {ckpt}/labels/ in the matrix flow.
    assert sec["output_dir"] == str(tmp_path)
    assert sec["embedding_key"] == "X_normalized_pca80"
    assert sec["reference_sampling"] == "balanced_by_experiment"
    assert sec["gmm_fit_population"] == "joint_control_perturbed"
    assert sec["gmm_covariance_type"] == "tied"
    assert sec["max_reference_cells"] == 1900

    sensor = yaml.safe_load(paths[1].read_text())["witness_gmm_labels"]
    assert sensor["experiments"][0]["embeddings_zarr"] == f"{_CKPT}/embeddings/pAL40.zarr"
    assert sensor["label_column"] == "infection_state"


def test_lc_annotations_autofilled_from_witness_sources():
    """Empty LC annotations are filled from witness (experiment × label source)."""
    out = Path(_CKPT)
    cfg = _witness_eval_cfg(output_dir="/static")

    anns = _resolve_lc_annotations(cfg, output_dir=out)

    # 1 experiment × 2 label sources → 2 annotation entries, each at {ckpt}/labels/.
    assert anns == [
        {
            "experiment": "2026_04_14_A549_SEC61B_DENV",
            "path": f"{_CKPT}/labels/SEC61B_organelle_remodeling_state.csv",
        },
        {
            "experiment": "2026_04_14_A549_SEC61B_DENV",
            "path": f"{_CKPT}/labels/pAL40_infection_state.csv",
        },
    ]


def test_lc_explicit_annotations_win_over_witness():
    """An explicit annotations list is used verbatim, not overridden by witness."""
    cfg = EvaluationConfig(
        training_config="/cfg/DynaCLR.yml",
        output_dir="/static",
        steps=["witness_gmm", "linear_classifiers"],
        witness_gmm={
            "label_sources": [
                {
                    "marker": "SEC61B",
                    "label_column": "organelle_remodeling_state",
                    "class_map": {"positive": "remodel", "negative": "noremodel"},
                },
            ],
            "experiments": ["exp_A"],
            "control_filter": {"perturbation": "uninfected"},
            "perturbed_filter": {"perturbation": ["DENV"]},
        },
        linear_classifiers={
            "annotations": [{"experiment": "exp_A", "path": "/hand/labels.csv"}],
            "tasks": [{"task": "organelle_remodeling_state"}],
        },
    )
    anns = _resolve_lc_annotations(cfg, output_dir=Path("/static"))
    assert anns == [{"experiment": "exp_A", "path": "/hand/labels.csv"}]


def test_lc_without_annotations_or_witness_rejected():
    """LC step with no annotations and no witness step fails config validation."""
    with pytest.raises(ValueError, match="linear_classifiers step needs annotations"):
        EvaluationConfig(
            training_config="/cfg/DynaCLR.yml",
            output_dir="/static",
            steps=["linear_classifiers"],
            linear_classifiers={"tasks": [{"task": "infection_state"}]},
        )
