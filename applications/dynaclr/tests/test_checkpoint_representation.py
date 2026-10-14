"""Tests for checkpoint-wide pooled representation planning."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from dynaclr.evaluation.mmd import checkpoint_representation as checkpoint


def _complete(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "zarr.json").write_text("{}")
    return path


def _patch_plan(monkeypatch, tmp_path, *, missing_experiment=False):
    collection = SimpleNamespace(experiments=[SimpleNamespace(name="DS_A"), SimpleNamespace(name="DS_B")])
    first = _complete(tmp_path / "DS_A_SEC61B.zarr")
    second = _complete(tmp_path / "DS_B_SEC61B.zarr")
    runs = [SimpleNamespace(experiment="DS_A", output_path=first)]
    if not missing_experiment:
        runs.append(SimpleNamespace(experiment="DS_B", output_path=second))
    monkeypatch.setattr(checkpoint, "load_collection", lambda _: collection)
    monkeypatch.setattr(checkpoint, "plan_predict_runs", lambda *args, **kwargs: runs)
    return first, second


def test_plan_checkpoint_representation_resolves_full_pool(monkeypatch, tmp_path):
    first, second = _patch_plan(monkeypatch, tmp_path)
    plan = checkpoint.plan_checkpoint_representation(
        tmp_path / "collection.yml",
        model_family="family",
        run="run",
        ckpt_name="last",
        datasets_root=tmp_path,
        markers=["SEC61B"],
    )
    assert plan.input_paths == (first, second)
    assert plan.experiments == ("DS_A", "DS_B")
    assert plan.artifact_dir == (tmp_path / "_pooled_representation/family/run/last/X_normalized_pca80")


def test_plan_checkpoint_representation_rejects_unrepresented_experiment(monkeypatch, tmp_path):
    _patch_plan(monkeypatch, tmp_path, missing_experiment=True)
    with pytest.raises(ValueError, match="DS_B"):
        checkpoint.plan_checkpoint_representation(
            tmp_path / "collection.yml",
            model_family="family",
            run="run",
            ckpt_name="last",
            datasets_root=tmp_path,
        )


def test_plan_checkpoint_representation_rejects_incomplete_store(monkeypatch, tmp_path):
    _patch_plan(monkeypatch, tmp_path)
    incomplete = tmp_path / "DS_B_SEC61B.zarr"
    (incomplete / "zarr.json").unlink()
    with pytest.raises(FileNotFoundError, match="DS_B_SEC61B"):
        checkpoint.plan_checkpoint_representation(
            tmp_path / "collection.yml",
            model_family="family",
            run="run",
            ckpt_name="last",
            datasets_root=tmp_path,
        )


def test_canonical_recipe_aliases_legacy_mock_controls():
    config = checkpoint.load_pooled_representation_config(checkpoint.DEFAULT_RECIPE)
    assert config.condition_aliases["uninfected"] == ["uninfected", "mock"]
