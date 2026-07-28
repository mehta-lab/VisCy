"""Unit tests for submit_matrix: .sh parsing, defaults merge, chained commands."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from dynaclr.evaluation.orchestration import matrix as submit_matrix

_SH = """#!/bin/bash
#SBATCH --job-name=foo
export PROJECT="DynaCLR-2D-MIP-BagOfChannels"
export RUN_NAME="2d-mip-fix-shuffler"
export CONFIGS="DynaCLR-2D/base.yml DynaCLR-2D/single-marker.yml"
source train.sh
"""


def _write_sh(tmp_path, text=_SH) -> Path:
    p = tmp_path / "model.sh"
    p.write_text(text)
    return p


def test_parse_train_sbatch(tmp_path):
    got = submit_matrix.parse_train_sbatch(_write_sh(tmp_path))
    assert got["family"] == "DynaCLR-2D-MIP-BagOfChannels"
    assert got["run"] == "2d-mip-fix-shuffler"
    assert got["train_configs"] == ["DynaCLR-2D/base.yml", "DynaCLR-2D/single-marker.yml"]


def test_parse_missing_export_raises(tmp_path):
    p = tmp_path / "bad.sh"
    p.write_text('export PROJECT="X"\nexport RUN_NAME="y"\n')  # no CONFIGS
    with pytest.raises(ValueError, match="CONFIGS"):
        submit_matrix.parse_train_sbatch(p)


def test_resolve_model_merges_defaults_and_parses(tmp_path):
    sh = _write_sh(tmp_path)
    defaults = {"ckpt_name": "last", "collection": "c.yml", "predict_flags": {"z_reduction": "mip"}}
    model = {"train_sbatch": str(sh)}
    r = submit_matrix.resolve_model(model, defaults)
    assert r["family"] == "DynaCLR-2D-MIP-BagOfChannels"
    assert r["run"] == "2d-mip-fix-shuffler"
    assert r["ckpt_name"] == "last"  # from defaults
    assert r["collection"] == "c.yml"  # from defaults
    assert r["predict_flags"]["z_reduction"] == "mip"


def test_resolve_model_explicit_overrides_parsed(tmp_path):
    sh = _write_sh(tmp_path)
    r = submit_matrix.resolve_model({"train_sbatch": str(sh), "run": "custom-run"}, {"ckpt_name": "last"})
    assert r["run"] == "custom-run"  # explicit wins over parsed RUN_NAME


def test_load_matrix_and_chain(tmp_path):
    sh_a = _write_sh(tmp_path)
    sh_b = tmp_path / "b.sh"
    sh_b.write_text(_SH.replace("fix-shuffler", "vits-boc"))
    matrix = tmp_path / "matrix.yml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "defaults": {
                    "ckpt_name": "last",
                    "collection": "c.yml",
                    "eval_config": "e.yaml",
                    "datasets_root": "/d",
                },
                "models": [{"train_sbatch": str(sh_a)}, {"train_sbatch": str(sh_b)}],
            }
        )
    )
    models = submit_matrix.load_matrix(matrix)
    assert len(models) == 2

    cmds = submit_matrix.build_stage_cmds(models[0], ("train", "predict", "eval"))
    assert [stage for stage, _ in cmds] == ["train", "predict", "eval"]
    # predict stage carries the resolved identity + collection + datasets_root
    predict_cmd = dict(cmds)["predict"]
    assert "DynaCLR-2D-MIP-BagOfChannels" in predict_cmd
    assert "c.yml" in predict_cmd
    assert "/d" in predict_cmd


def test_stages_subset_skips_train(tmp_path):
    sh = _write_sh(tmp_path)
    model = submit_matrix.resolve_model(
        {"train_sbatch": str(sh)},
        {"ckpt_name": "last", "collection": "c.yml", "eval_config": "e.yaml", "datasets_root": "/d"},
    )
    cmds = submit_matrix.build_stage_cmds(model, ("predict", "eval"))
    assert [stage for stage, _ in cmds] == ["predict", "eval"]


def test_predict_passes_explicit_checkpoint(tmp_path):
    """An explicit `checkpoint:` in the model entry is forwarded to predict.sbatch."""
    sh = _write_sh(tmp_path)
    ckpt = "/hpc/.../jbrwhzr3/checkpoints/epoch=105-step=84800.ckpt"
    model = submit_matrix.resolve_model(
        {"train_sbatch": str(sh), "checkpoint": ckpt},
        {"ckpt_name": "epoch105-step84800", "collection": "c.yml", "eval_config": "e.yaml", "datasets_root": "/d"},
    )
    predict_cmd = dict(submit_matrix.build_stage_cmds(model, ("predict",)))["predict"]
    assert ckpt in predict_cmd


def test_predict_empty_checkpoint_when_derived(tmp_path):
    """No explicit checkpoint → an empty positional (predict.sbatch then derives)."""
    sh = _write_sh(tmp_path)
    model = submit_matrix.resolve_model(
        {"train_sbatch": str(sh)},
        {"ckpt_name": "last", "collection": "c.yml", "eval_config": "e.yaml", "datasets_root": "/d"},
    )
    predict_cmd = dict(submit_matrix.build_stage_cmds(model, ("predict",)))["predict"]
    assert predict_cmd[7] == ""  # empty checkpoint positional
    assert predict_cmd[8] == ""  # empty markers positional


def test_predict_passes_markers(tmp_path):
    """A matrix row's `markers:` list is forwarded as a comma-joined positional."""
    sh = _write_sh(tmp_path)
    model = submit_matrix.resolve_model(
        {"train_sbatch": str(sh), "markers": ["SEC61B", "Phase3D"]},
        {"ckpt_name": "last", "collection": "c.yml", "eval_config": "e.yaml", "datasets_root": "/d"},
    )
    predict_cmd = dict(submit_matrix.build_stage_cmds(model, ("predict",)))["predict"]
    assert predict_cmd[8] == "SEC61B,Phase3D"


def test_matrix_forwards_focus_centered_z_window():
    """Focus window and physical Z reference ride the final positionals."""
    model = {
        "collection": "c.yml",
        "family": "family",
        "run": "run",
        "datasets_root": "/datasets",
        "eval_config": "e.yml",
        "predict_flags": {
            "z_window": 30,
            "focus_channel": "Phase3D",
            "z_focus_offset": 0.3,
            "z_reduction": "mip",
            "reference_pixel_size_z_um": 0.174,
        },
    }
    predict_cmd = submit_matrix.build_predict_cmd(model, "last", "")
    assert predict_cmd[-4:] == ["30", "Phase3D", "0.3", "0.174"]
    # focus-centered → no fixed z_range: z_start/z_end positionals are empty
    assert predict_cmd[9] == "" and predict_cmd[10] == ""


def test_matrix_z_range_and_z_window_mutually_exclusive():
    model = {
        "collection": "c.yml",
        "family": "family",
        "run": "run",
        "datasets_root": "/datasets",
        "eval_config": "e.yml",
        "predict_flags": {"z_range": [15, 45], "z_window": 30},
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        submit_matrix.build_predict_cmd(model, "last", "")


def test_matrix_forwards_predict_flags_and_eval_root():
    model = {
        "collection": "c.yml",
        "family": "family",
        "run": "run",
        "datasets_root": "/datasets",
        "eval_config": "e.yml",
        "predict_flags": {
            "z_range": [15, 45],
            "z_reduction": "mip",
            "reference_pixel_size": 0.1494,
            "batch_size": 64,
        },
    }

    predict_cmd = submit_matrix.build_predict_cmd(model, "last", "")
    # predict_flags positionals, model_type + training_config (dynaclr: empty),
    # then focus-centered z (empty here — fixed z_range was used).
    assert predict_cmd[-10:] == ["15", "45", "mip", "0.1494", "64", "dynaclr", "", "", "", ""]

    eval_cmd = submit_matrix.build_eval_cmd(model, "last")
    assert eval_cmd[-1] == "/datasets"


def test_resolve_foundation_model_needs_no_train_sbatch():
    """A foundation row resolves from explicit identity + training_config (no .sh)."""
    model = {
        "model_type": "foundation",
        "family": "MorphEm-frozen",
        "run": "frozen",
        "ckpt_name": "frozen",
        "training_config": "cfg.yaml",
    }
    r = submit_matrix.resolve_model(model, {"collection": "c.yml"})
    assert r["model_type"] == "foundation"
    assert r["family"] == "MorphEm-frozen"
    assert "train_sbatch" not in r


def test_resolve_foundation_model_missing_field_raises():
    with pytest.raises(ValueError, match="training_config"):
        submit_matrix.resolve_model({"model_type": "foundation", "family": "f", "run": "r", "ckpt_name": "c"}, {})


def test_foundation_predict_cmd_carries_model_type_and_config():
    """Foundation predict positionals: empty checkpoint, model_type, training_config."""
    model = {
        "model_type": "foundation",
        "collection": "c.yml",
        "family": "MorphEm-frozen",
        "run": "frozen",
        "datasets_root": "/d",
        "training_config": "cfg.yaml",
        "markers": ["SEC61B"],
    }
    predict_cmd = submit_matrix.build_predict_cmd(model, "frozen", "")
    # model_type + training_config, then the 3 focus-centered z positionals (empty here).
    assert predict_cmd[-5:] == ["foundation", "cfg.yaml", "", "", ""]
    # checkpoint positional stays empty for foundation rows
    assert predict_cmd[7] == ""


def test_foundation_run_model_skips_train(capsys):
    """Foundation rows never emit a train stage even when 'train' is requested."""
    model = {
        "model_type": "foundation",
        "collection": "c.yml",
        "family": "MorphEm-frozen",
        "run": "frozen",
        "ckpt_name": "frozen",
        "eval_config": "e.yaml",
        "datasets_root": "/d",
        "training_config": "cfg.yaml",
    }
    submit_matrix.run_model(model, ("train", "predict", "eval"), print_only=True)
    out = capsys.readouterr().out
    assert "[train]" not in out
    assert out.count("[predict]") == 1
    assert out.count("[eval]") == 1


def test_predict_sbatch_repeats_marker_option(tmp_path):
    """Each marker must have its own Click ``--markers`` option."""
    fake_srun = tmp_path / "srun"
    fake_srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_srun.chmod(0o755)

    workspace = Path(__file__).parents[3]
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["WORKSPACE_DIR"] = str(workspace)
    result = subprocess.run(
        [
            "bash",
            str(workspace / "applications/dynaclr/tools/predict.sbatch"),
            "collection.yml",
            "family",
            "run",
            "last",
            "/datasets",
            "",
            "SEC61B,TOMM20",
            "",
            "",
            "mip",
            "0.1494",
            "64",
            "",  # model_type
            "",  # training_config
            "30",
            "Phase3D",
            "0.3",
            "0.174",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    args = result.stdout.splitlines()
    marker_positions = [i for i, arg in enumerate(args) if arg == "--markers"]
    assert [args[i + 1] for i in marker_positions] == ["SEC61B", "TOMM20"]
    assert "--z-range" not in args
    assert args[args.index("--z-window") + 1] == "30"
    assert args[args.index("--z-reduction") + 1] == "mip"
    assert args[args.index("--reference-pixel-size") + 1] == "0.1494"
    assert args[args.index("--reference-pixel-size-z-um") + 1] == "0.174"
    assert args[args.index("--batch-size") + 1] == "64"


def test_predict_sbatch_forwards_foundation_flags(tmp_path):
    """The $13 model_type / $14 training_config positionals become CLI flags."""
    fake_srun = tmp_path / "srun"
    fake_srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_srun.chmod(0o755)

    workspace = Path(__file__).parents[3]
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["WORKSPACE_DIR"] = str(workspace)
    result = subprocess.run(
        [
            "bash",
            str(workspace / "applications/dynaclr/tools/predict.sbatch"),
            "collection.yml",
            "MorphEm-frozen",
            "frozen",
            "frozen",
            "/datasets",
            "",  # checkpoint (unused for foundation)
            "SEC61B",
            "15",
            "45",
            "mip",
            "0.1494",
            "",  # batch_size
            "foundation",
            "cfg.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    args = result.stdout.splitlines()
    assert args[args.index("--model-type") + 1] == "foundation"
    assert args[args.index("--training-config") + 1] == "cfg.yaml"
    assert "--checkpoint" not in args


def test_eval_sbatch_forwards_datasets_root(tmp_path):
    fake_module = tmp_path / "module"
    fake_module.write_text("#!/bin/bash\nexit 0\n")
    fake_module.chmod(0o755)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_uv.chmod(0o755)

    workspace = Path(__file__).parents[3]
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["WORKSPACE_DIR"] = str(workspace)
    model = {
        "eval_config": "e.yml",
        "family": "family",
        "run": "run",
        "datasets_root": "/datasets",
    }
    cmd = submit_matrix.build_eval_cmd(model, "last")
    result = subprocess.run(
        ["bash", *cmd[1:]],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    args = result.stdout.splitlines()
    assert args[args.index("--datasets-root") + 1] == "/datasets"


# --- matrix-level preflight -------------------------------------------------
import numpy as np  # noqa: E402
from iohub.ngff import open_ome_zarr  # noqa: E402


def _zarr(path, *, normalization, focus_slice):
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["Phase3D"]) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_zeros("0", shape=(1, 1, 4, 8, 8), dtype=np.float32)
        if normalization:
            pos.zattrs["normalization"] = {"Phase3D": {"fov_statistics": {"mean": 0.0, "std": 1.0}}}
        if focus_slice:
            pos.zattrs["focus_slice"] = {"Phase3D": {"fov_statistics": {"z_focus_mean": 2}}}


def _collection_for(tmp_path, zarr_path, name="ds"):
    coll = tmp_path / f"{name}.yml"
    coll.write_text(
        yaml.safe_dump(
            {
                "name": "c",
                "experiments": [
                    {
                        "name": name,
                        "data_path": str(zarr_path),
                        "tracks_path": str(tmp_path / "t"),
                        "channels": [{"name": "Phase3D", "marker": "Phase3D"}],
                        "perturbation_wells": {"uninfected": ["A/1"]},
                    }
                ],
            }
        )
    )
    return coll


def test_matrix_preflight_passes_when_ready(tmp_path):
    z = tmp_path / "ready.zarr"
    _zarr(z, normalization=True, focus_slice=True)
    models = [{"collection": str(_collection_for(tmp_path, z))}]
    submit_matrix.matrix_preflight(models)  # should not raise


def test_matrix_preflight_raises_on_missing_focus(tmp_path):
    z = tmp_path / "nofocus.zarr"
    _zarr(z, normalization=True, focus_slice=False)
    models = [{"collection": str(_collection_for(tmp_path, z))}]
    with pytest.raises(ValueError, match="focus_slice"):
        submit_matrix.matrix_preflight(models)


def test_resolve_checkpoints_scalar(tmp_path):
    sh = _write_sh(tmp_path)
    m = submit_matrix.resolve_model({"train_sbatch": str(sh)}, {"ckpt_name": "last"})
    assert submit_matrix.resolve_checkpoints(m) == [("last", "")]


def test_resolve_checkpoints_sweep_list(tmp_path):
    sh = _write_sh(tmp_path)
    m = submit_matrix.resolve_model({"train_sbatch": str(sh), "ckpt_names": ["last", "epoch80-step64000"]}, {})
    got = submit_matrix.resolve_checkpoints(m)
    assert [n for n, _ in got] == ["last", "epoch80-step64000"]


def test_run_model_sweep_fans_out(tmp_path, capsys):
    """ckpt sweep prints one train + a predict/eval pair per checkpoint (dry-run)."""
    sh = _write_sh(tmp_path)
    m = submit_matrix.resolve_model(
        {"train_sbatch": str(sh), "ckpt_names": ["last", "epoch80-step64000"]},
        {"collection": "c.yml", "eval_config": "e.yaml", "datasets_root": "/d"},
    )
    submit_matrix.run_model(m, ("train", "predict", "eval"), print_only=True)
    out = capsys.readouterr().out
    assert out.count("[train]") == 1  # train once
    assert out.count("[predict]") == 2  # one per checkpoint
    assert out.count("[eval]") == 2


# --- resolve_datasets_to_run: progressive-collection skip-existing pre-step -------------

_FAMILY = "DynaCLR-2D-MIP-BagOfChannels"
_RUN = "2d-mip-fix-shuffler"
_CKPT = "epoch105-step84800"
_MARKERS = ["SEC61B", "Phase3D"]


def _two_exp_collection_yaml(tmp_path) -> Path:
    """Write a 2-experiment collection (DS_A, DS_B); data_path parent = dataset name."""
    root = tmp_path / "datasets"

    def _exp(name):
        return {
            "name": name,
            "data_path": f"{root}/{name}/{name}.zarr",
            "tracks_path": f"{root}/{name}/tracking.zarr",
            "channels": [
                {"name": "raw GFP EX488 EM525-45", "marker": "SEC61B"},
                {"name": "Phase3D", "marker": "Phase3D"},
            ],
            "perturbation_wells": {"uninfected": ["A/1"], "ZIKV": ["B/1"]},
        }

    coll = tmp_path / "collection.yml"
    coll.write_text(yaml.safe_dump({"name": "prog", "experiments": [_exp("DS_A"), _exp("DS_B")]}))
    return coll


def _mark_done(tmp_path, dataset: str) -> None:
    """Create complete per-marker embedding zarrs (dir + zarr.json) for a dataset."""
    for marker in _MARKERS:
        d = (
            tmp_path
            / "datasets"
            / dataset
            / "2-phenotyping"
            / "predictions"
            / _FAMILY
            / _RUN
            / _CKPT
            / "embeddings"
            / f"{marker}.zarr"
        )
        d.mkdir(parents=True, exist_ok=True)
        (d / "zarr.json").write_text("{}")


def _model(coll, tmp_path) -> dict:
    return {
        "family": _FAMILY,
        "run": _RUN,
        "ckpt_name": _CKPT,
        "collection": str(coll),
        "markers": _MARKERS,
        "datasets_root": str(tmp_path / "datasets"),
    }


def test_resolve_drops_fully_done_row(tmp_path):
    coll = _two_exp_collection_yaml(tmp_path)
    _mark_done(tmp_path, "DS_A")
    _mark_done(tmp_path, "DS_B")
    survivors = submit_matrix.resolve_datasets_to_run([_model(coll, tmp_path)], overwrite=False)
    assert survivors == []  # both datasets done → row dropped


def test_resolve_prunes_partial_row(tmp_path):
    coll = _two_exp_collection_yaml(tmp_path)
    _mark_done(tmp_path, "DS_A")  # DS_A done, DS_B missing
    survivors = submit_matrix.resolve_datasets_to_run([_model(coll, tmp_path)], overwrite=False)
    assert len(survivors) == 1
    from viscy_data.collection import load_collection

    pruned_coll = load_collection(Path(survivors[0]["collection"]))
    assert [e.name for e in pruned_coll.experiments] == ["DS_B"]  # only the missing one
    assert survivors[0]["collection"] != str(coll)  # repointed to the pruned temp YAML


def test_resolve_keeps_row_when_none_done(tmp_path):
    coll = _two_exp_collection_yaml(tmp_path)  # nothing marked done
    survivors = submit_matrix.resolve_datasets_to_run([_model(coll, tmp_path)], overwrite=False)
    assert len(survivors) == 1
    assert survivors[0]["collection"] == str(coll)  # unpruned (all missing) → original collection


def test_resolve_overwrite_bypasses(tmp_path):
    coll = _two_exp_collection_yaml(tmp_path)
    _mark_done(tmp_path, "DS_A")
    _mark_done(tmp_path, "DS_B")
    models = [_model(coll, tmp_path)]
    survivors = submit_matrix.resolve_datasets_to_run(models, overwrite=True)
    assert survivors == models  # unchanged despite everything done


def test_resolve_partial_zarr_counts_as_missing(tmp_path):
    """A dir without zarr.json (crash mid-write) is NOT complete → dataset re-runs."""
    coll = _two_exp_collection_yaml(tmp_path)
    _mark_done(tmp_path, "DS_A")
    # DS_B: create only ONE marker's zarr, and one bare dir with no zarr.json
    partial = (
        tmp_path
        / "datasets"
        / "DS_B"
        / "2-phenotyping"
        / "predictions"
        / _FAMILY
        / _RUN
        / _CKPT
        / "embeddings"
        / "SEC61B.zarr"
    )
    partial.mkdir(parents=True, exist_ok=True)  # no zarr.json → incomplete
    survivors = submit_matrix.resolve_datasets_to_run([_model(coll, tmp_path)], overwrite=False)
    from viscy_data.collection import load_collection

    pruned_coll = load_collection(Path(survivors[0]["collection"]))
    assert [e.name for e in pruned_coll.experiments] == ["DS_B"]  # DS_B still to run
