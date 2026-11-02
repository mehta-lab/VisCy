"""Eligibility forwarding and cache compatibility without real jobs or images."""

import json
import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from dynaclr.evaluation.orchestration import matrix, predict_batch
from dynaclr.evaluation.phenotyping_eligibility import (
    ELIGIBILITY_COLUMNS,
    eligibility_cache_matches,
    load_eligibility,
)
from dynaclr.evaluation.predict_triplet import plan_predict_runs
from viscy_data.collection import load_collection

WORKSPACE = Path(__file__).resolve().parents[3]


@pytest.fixture
def eligibility(tmp_path):
    table = pd.DataFrame(
        [["DS_A", "A/1/000", 1, 0, "GFP", pd.NA, "not reviewed"]],
        columns=ELIGIBILITY_COLUMNS,
    )
    table["eligible"] = table["eligible"].astype("boolean")
    path = tmp_path / "review eligibility.parquet"
    table.to_parquet(path)
    return path


@pytest.fixture
def model(tmp_path, eligibility):
    collection = tmp_path / "collection.yml"
    collection.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "experiments": [
                    {
                        "name": dataset,
                        "data_path": str(tmp_path / "datasets" / dataset / f"{dataset}.zarr"),
                        "tracks_path": str(tmp_path / "tracks" / dataset),
                        "channels": [{"name": "GFP", "marker": "SEC61B"}, {"name": "Phase3D", "marker": "Phase3D"}],
                        "perturbation_wells": {"uninfected": ["A/1"]},
                    }
                    for dataset in ("DS_A", "DS_B")
                ],
            }
        )
    )
    return {
        "collection": str(collection),
        "family": "family",
        "run": "run",
        "ckpt_name": "last",
        "datasets_root": str(tmp_path / "datasets"),
        "eligibility": str(eligibility),
        "eval_config": "evaluation.yml",
    }


def _stores(model, ckpt_name="last"):
    return plan_predict_runs(
        load_collection(Path(model["collection"])),
        model_family=model["family"],
        run=model["run"],
        ckpt_name=ckpt_name,
        datasets_root=model["datasets_root"],
    )


def _write_caches(model, eligibility, *, ckpt_name="last", filtered=True, normalized=True):
    _, provenance = load_eligibility(eligibility)
    audit = {**provenance, "policy": "exclude_explicit_false_only"}
    for run in _stores(model, ckpt_name):
        store = ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))
        if normalized:
            store.obsm["X_normalized_pca80"] = np.zeros((1, 1), dtype=np.float32)
            store.uns["X_normalized_pca80"] = {"fit_cells": 1}
        if filtered:
            store.uns["phenotyping_eligibility"] = json.dumps(audit)
        run.output_path.parent.mkdir(parents=True, exist_ok=True)
        store.write_zarr(run.output_path)
    return audit


def _update_audit(path, value):
    from contextlib import closing

    import zarr
    from anndata.io import write_elem

    with closing(zarr.storage.LocalStore(str(path))) as store:
        group = zarr.open_group(store=store, mode="a", use_consolidated=False)
        write_elem(group["uns"], "phenotyping_eligibility", value)


def _snapshot(model):
    return {
        path: sha256(path.read_bytes()).hexdigest()
        for run in _stores(model)
        for path in run.output_path.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("model_type", ["dynaclr", "foundation", "katamari"])
def test_batch_command_forwards_explicit_path(model, eligibility, model_type):
    cmd = predict_batch.build_predict_cmd(
        Path(model["collection"]),
        Path("model.ckpt"),
        "family",
        "run",
        "last",
        "/datasets",
        model_type=model_type,
        training_config="training.yml",
        eligibility=eligibility,
    )
    assert cmd[cmd.index("--eligibility") + 1] == str(eligibility.resolve())
    assert cmd.count("--eligibility") == 1
    plain = predict_batch.build_predict_cmd(
        Path("collection.yml"), Path("model.ckpt"), "family", "run", "last", "/datasets"
    )
    assert "--eligibility" not in plain


def test_batch_cli_forwards_eligibility_without_launch(model, eligibility):
    result = CliRunner().invoke(
        predict_batch.main,
        [
            "-c",
            model["collection"],
            "--model-family",
            "family",
            "--run",
            "run",
            "--eligibility",
            str(eligibility),
            "--print-cmd",
            "--skip-preflight",
        ],
    )
    assert result.exit_code == 0, result.output
    args = result.output.splitlines()
    assert args[args.index("--eligibility") + 1] == str(eligibility.resolve())


def test_defaults_eligibility_is_merged_and_exported_without_new_positionals(model, eligibility, monkeypatch):
    monkeypatch.delenv("DYNACLR_ELIGIBILITY", raising=False)
    defaults = {**model, "model_type": "foundation", "training_config": "training.yml"}
    resolved = matrix.resolve_model({}, defaults)
    cmd = matrix.build_predict_cmd(resolved, "last", "")
    assert cmd[1] == f"--export=ALL,DYNACLR_ELIGIBILITY={eligibility.resolve()}"
    plain = matrix.build_predict_cmd({**resolved, "eligibility": None}, "last", "")
    assert cmd[:1] + cmd[2:] == plain
    override = matrix.resolve_model({"eligibility": None}, defaults)
    assert matrix.build_predict_cmd(override, "last", "") == plain


def test_matrix_rejects_conflicting_inherited_eligibility(model, monkeypatch):
    monkeypatch.setenv("DYNACLR_ELIGIBILITY", "/unexpected.parquet")
    with pytest.raises(ValueError, match="conflicting DYNACLR_ELIGIBILITY"):
        matrix.build_predict_cmd({**model, "eligibility": None}, "last", "")
    with pytest.raises(ValueError, match="conflicting DYNACLR_ELIGIBILITY"):
        matrix.build_predict_cmd(model, "last", "")
    monkeypatch.setenv("DYNACLR_ELIGIBILITY", str(Path(model["eligibility"]).resolve()))
    assert f"--export=ALL,DYNACLR_ELIGIBILITY={model['eligibility']}" in matrix.build_predict_cmd(model, "last", "")


@pytest.mark.parametrize("model_type", ["dynaclr", "katamari"])
def test_sbatch_forwards_environment_path_before_runtime_cd(model, eligibility, tmp_path, monkeypatch, model_type):
    fake_srun = tmp_path / "srun"
    fake_srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_srun.chmod(0o755)
    monkeypatch.delenv("DYNACLR_ELIGIBILITY", raising=False)
    configured = {
        **model,
        "model_type": model_type,
        "evolutionaryscale_root": str(tmp_path),
        "pixi_executable": str(fake_srun),
    }
    cmd = matrix.build_predict_cmd(configured, "last", "model.ckpt")
    script_index = cmd.index(str(matrix._PREDICT_SBATCH))
    exports = next(arg for arg in cmd if arg.startswith("--export="))
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["WORKSPACE_DIR"] = str(WORKSPACE)
    env["DYNACLR_ELIGIBILITY"] = exports.split("DYNACLR_ELIGIBILITY=", 1)[1]
    result = subprocess.run(["bash", *cmd[script_index:]], env=env, check=True, capture_output=True, text=True)
    args = result.stdout.splitlines()
    assert args[args.index("--eligibility") + 1] == str(eligibility.resolve())
    assert args.count("--eligibility") == 1
    assert args[args.index("--model-type") + 1] == model_type


@pytest.mark.parametrize("entry", ["matrix_command", "matrix_cache", "batch_command"])
def test_rejects_eligibility_inside_predict_flags(model, eligibility, entry):
    configured = {**model, "predict_flags": {"eligibility": str(eligibility)}}
    with pytest.raises(ValueError, match="predict_flags.eligibility"):
        if entry == "matrix_command":
            matrix.build_predict_cmd(configured, "last", "")
        elif entry == "matrix_cache":
            matrix.resolve_datasets_to_run([configured], overwrite=False)
        else:
            predict_batch.build_predict_cmd(
                Path("c.yml"), None, "family", "run", "last", "/d", predict_flags=configured["predict_flags"]
            )


def test_missing_path_fails_before_submitting(model):
    configured = {**model, "eligibility": "/missing/eligibility.parquet"}
    with pytest.raises(FileNotFoundError, match="Eligibility parquet"):
        matrix.resolve_datasets_to_run([configured], overwrite=False)
    with pytest.raises(FileNotFoundError, match="Eligibility parquet"):
        matrix.build_predict_cmd(configured, "last", "")


def test_matching_hashes_skip_raw_and_normalized_caches_without_array_reads(model, eligibility, monkeypatch):
    _write_caches(model, eligibility)
    before = _snapshot(model)
    monkeypatch.setattr(ad, "read_zarr", lambda *_: pytest.fail("Cache check read embedding arrays"))
    resolved = matrix.resolve_datasets_to_run([model], overwrite=False, preserve_complete_rows=True)[0]
    assert resolved["_skip_predict"] is True
    assert resolved["_skip_normalize_ckpts"] == frozenset({"last"})
    assert [stage for stage, _ in matrix.build_stage_cmds(resolved, ("predict", "normalize", "eval"))] == ["eval"]
    assert _snapshot(model) == before


def test_matching_raw_cache_requires_normalization_only_when_requested(model, eligibility):
    _write_caches(model, eligibility, normalized=False)
    before = _snapshot(model)
    resolved = matrix.resolve_datasets_to_run([model], overwrite=False, preserve_complete_rows=True)[0]
    assert resolved["_skip_predict"] is True
    assert not resolved["_skip_normalize_ckpts"]
    assert matrix.build_stage_cmds(resolved, ("predict",)) == []
    assert [stage for stage, _ in matrix.build_stage_cmds(resolved, ("predict", "normalize"))] == ["normalize"]
    assert _snapshot(model) == before


@pytest.mark.parametrize("stages", ["normalize", "eval", "normalize,eval"])
@pytest.mark.parametrize("normalized", [False, True])
def test_downstream_only_cli_checks_identity_before_any_stage(model, eligibility, monkeypatch, stages, normalized):
    _write_caches(model, eligibility, normalized=normalized)
    monkeypatch.setattr(matrix, "load_matrix", lambda _: [model])
    calls = []
    monkeypatch.setattr(matrix, "run_model", lambda *args: calls.append(args))
    before = _snapshot(model)
    args = ["--matrix", model["collection"], "--stages", stages, "--dry-run"]
    accepted = CliRunner().invoke(matrix.main, args)
    assert accepted.exit_code == 0, accepted.output
    assert len(calls) == 1
    assert calls[0][0] == model
    calls.clear()
    table = pd.read_parquet(eligibility)
    table.loc[0, "eligible"] = False
    table.to_parquet(eligibility)
    rejected = CliRunner().invoke(matrix.main, args)
    assert rejected.exit_code != 0
    assert "identity conflicts" in str(rejected.exception)
    assert not calls
    assert _snapshot(model) == before
    assert not list(Path(model["collection"]).parent.glob("*.pending-*.yml"))


def test_downstream_only_requires_complete_raw_inputs(model):
    with pytest.raises(ValueError, match="Missing raw prediction embeddings"):
        matrix.resolve_datasets_to_run([model], overwrite=False, validate_only=True)


@pytest.mark.parametrize("change", ["decision", "file_only"])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_changed_eligibility_requires_distinct_identity_for_raw_and_normalized_caches(
    model, eligibility, change, normalized, overwrite
):
    original = _write_caches(model, eligibility, normalized=normalized)
    table = pd.read_parquet(eligibility)
    if change == "decision":
        table.loc[0, "eligible"] = False
    table.to_parquet(eligibility, compression="gzip")
    _, updated = load_eligibility(eligibility)
    assert updated["eligibility_file_sha256"] != original["eligibility_file_sha256"]
    assert (updated["eligibility_records_sha256"] == original["eligibility_records_sha256"]) == (change == "file_only")
    before = _snapshot(model)
    with pytest.raises(ValueError, match="distinct run/output identity"):
        matrix.resolve_datasets_to_run([model], overwrite=overwrite, preserve_complete_rows=True)
    distinct = {**model, "run": "run-eligibility-v2", "checkpoint": "/original/run/checkpoints/last.ckpt"}
    resolved = matrix.resolve_datasets_to_run([distinct], overwrite=False, preserve_complete_rows=True)[0]
    assert not resolved.get("_skip_predict", False)
    assert resolved["_skip_normalize_ckpts"] == frozenset()
    assert resolved["collection"] == model["collection"]
    assert not any(run.output_path.exists() for run in _stores(distinct))
    predict = matrix.build_predict_cmd(resolved, "last", distinct["checkpoint"])
    assert distinct["run"] in predict and distinct["checkpoint"] in predict
    assert distinct["run"] in matrix.build_normalize_cmd(resolved, "last")
    assert distinct["run"] in matrix.build_eval_cmd(resolved, "last")
    assert [stage for stage, _ in matrix.build_stage_cmds(resolved, ("predict",))] == ["predict"]
    assert [stage for stage, _ in matrix.build_stage_cmds(resolved, ("predict", "eval"))] == ["predict", "eval"]
    assert [stage for stage, _ in matrix.build_stage_cmds(resolved, ("predict", "normalize", "eval"))] == [
        "predict",
        "normalize",
        "eval",
    ]
    assert _snapshot(model) == before


@pytest.mark.parametrize("direction", ["enable", "disable"])
@pytest.mark.parametrize("normalized", [False, True])
def test_switching_filter_mode_never_reuses_incompatible_caches(model, eligibility, direction, normalized):
    _write_caches(model, eligibility, filtered=direction == "disable", normalized=normalized)
    configured = {**model, "eligibility": None} if direction == "disable" else model
    before = _snapshot(model)
    with pytest.raises(ValueError, match="identity conflicts"):
        matrix.resolve_datasets_to_run([configured], overwrite=False, preserve_complete_rows=True)
    assert _snapshot(model) == before


@pytest.mark.parametrize(
    "field", ["eligibility_file_sha256", "eligibility_records_sha256", "policy", "invalid_json", "missing_hash"]
)
def test_one_incompatible_reporter_rejects_entire_row_without_mutation(model, eligibility, field):
    audit = _write_caches(model, eligibility)
    run = next(run for run in _stores(model) if run.experiment == "DS_B" and run.marker == "SEC61B")
    if field == "missing_hash":
        del audit["eligibility_records_sha256"]
    elif field != "invalid_json":
        audit[field] = "incompatible"
    _update_audit(run.output_path, "invalid json" if field == "invalid_json" else json.dumps(audit))
    before = _snapshot(model)
    with pytest.raises(ValueError, match="identity conflicts"):
        matrix.resolve_datasets_to_run([model], overwrite=False, preserve_complete_rows=True)
    assert _snapshot(model) == before


def test_missing_reporter_prunes_dataset_and_refits_full_pool(model, eligibility):
    _write_caches(model, eligibility)
    run = next(run for run in _stores(model) if run.experiment == "DS_B" and run.marker == "SEC61B")
    shutil.rmtree(run.output_path)
    before = _snapshot(model)
    resolved = matrix.resolve_datasets_to_run([model], overwrite=False, preserve_complete_rows=True)[0]
    pending = load_collection(Path(resolved["collection"]))
    assert [entry.name for entry in pending.experiments] == ["DS_B"]
    assert resolved["_normalization_collection"] == model["collection"]
    assert resolved["_skip_normalize_ckpts"] == frozenset()
    assert matrix.build_normalize_cmd(resolved, "last")[2] == model["collection"]
    assert _snapshot(model) == before


def test_legacy_dict_audit_matches_but_missing_provenance_does_not(model, eligibility):
    audit = _write_caches(model, eligibility)
    run = _stores(model)[0]
    _update_audit(run.output_path, audit)
    assert eligibility_cache_matches(run.output_path, audit)
    assert not eligibility_cache_matches(run.output_path, None)
    _update_audit(run.output_path, json.dumps({"policy": "exclude_explicit_false_only"}))
    assert not eligibility_cache_matches(run.output_path, audit)


def test_shared_pending_collection_invalidates_normalization_for_every_checkpoint(model, eligibility):
    model = {**model, "ckpt_names": ["last", "epoch2"]}
    _write_caches(model, eligibility)
    _write_caches(model, eligibility, ckpt_name="epoch2")
    run = next(run for run in _stores(model, "epoch2") if run.experiment == "DS_B")
    shutil.rmtree(run.output_path)
    resolved = matrix.resolve_datasets_to_run([model], overwrite=False, preserve_complete_rows=True)[0]
    assert [entry.name for entry in load_collection(Path(resolved["collection"])).experiments] == ["DS_B"]
    assert resolved["_skip_normalize_ckpts"] == frozenset()


def test_overwrite_clears_prior_skip_decisions(model):
    stale = {**model, "_skip_predict": True, "_skip_normalize_ckpts": frozenset({"last"})}
    resolved = matrix.resolve_datasets_to_run([stale], overwrite=True)[0]
    assert "_skip_predict" not in resolved
    assert "_skip_normalize_ckpts" not in resolved
    assert "_skip_predict" in stale
