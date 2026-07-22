"""Unit tests for submit_matrix: .sh parsing, defaults merge, chained commands."""

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
    # positionals end with: … checkpoint(""), markers("")
    assert predict_cmd[-2] == ""  # empty checkpoint positional
    assert predict_cmd[-1] == ""  # empty markers positional


def test_predict_passes_markers(tmp_path):
    """A matrix row's `markers:` list is forwarded as a comma-joined positional."""
    sh = _write_sh(tmp_path)
    model = submit_matrix.resolve_model(
        {"train_sbatch": str(sh), "markers": ["SEC61B", "Phase3D"]},
        {"ckpt_name": "last", "collection": "c.yml", "eval_config": "e.yaml", "datasets_root": "/d"},
    )
    predict_cmd = dict(submit_matrix.build_stage_cmds(model, ("predict",)))["predict"]
    assert predict_cmd[-1] == "SEC61B,Phase3D"


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
