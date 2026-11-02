"""Unit tests for submit_eval glob + command construction (pure logic)."""

from pathlib import Path

from dynaclr.evaluation.orchestration import eval_launch as submit_eval

MF = "DynaCLR-2D-MIP-BagOfChannels"
RUN = "2d-mip-fix-shuffler"
CKPT = "epoch105-step84800"
ROOT = "/base"


def test_glob_all_datasets_all_markers():
    g = submit_eval.build_embeddings_glob(MF, RUN, CKPT, datasets_root=ROOT)
    assert g == f"/base/*/2-phenotyping/predictions/{MF}/{RUN}/{CKPT}/*.zarr"


def test_glob_one_marker():
    g = submit_eval.build_embeddings_glob(MF, RUN, CKPT, marker="SEC61B", datasets_root=ROOT)
    assert g.endswith(f"/{CKPT}/SEC61B.zarr")
    assert "/*/2-phenotyping/" in g  # still all datasets


def test_glob_single_dataset_no_braces():
    g = submit_eval.build_embeddings_glob(MF, RUN, CKPT, datasets=["ds_a"], datasets_root=ROOT)
    assert "/base/ds_a/2-phenotyping/" in g
    assert "{" not in g


def test_glob_multi_dataset_brace_expansion():
    g = submit_eval.build_embeddings_glob(MF, RUN, CKPT, datasets=["ds_a", "ds_b"], datasets_root=ROOT)
    assert "/base/{ds_a,ds_b}/2-phenotyping/" in g


def test_nextflow_cmd_has_entry_and_glob():
    cmd = submit_eval.build_nextflow_cmd(Path("eval.yaml"), "/base/*/x/*.zarr", "/ws", resume=True)
    assert "eval_from_embeddings" in cmd
    assert "--embeddings_glob" in cmd
    assert cmd[cmd.index("--embeddings_glob") + 1] == "/base/*/x/*.zarr"
    assert "-resume" in cmd


def test_nextflow_cmd_no_resume():
    cmd = submit_eval.build_nextflow_cmd(Path("eval.yaml"), "g", "/ws", resume=False)
    assert "-resume" not in cmd
