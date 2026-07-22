"""Unit tests for submit_predict checkpoint resolution + command building."""

from pathlib import Path

from dynaclr.evaluation.orchestration import predict_batch as submit_predict

MF = "DynaCLR-2D-MIP-BagOfChannels"
RUN = "2d-mip-fix-shuffler"
ROOT = "/models"


def test_ckpt_last():
    p = submit_predict.checkpoint_path(MF, RUN, "last", models_root=ROOT)
    assert p == Path(f"/models/{MF}/{RUN}/checkpoints/last.ckpt")


def test_ckpt_epoch_label():
    p = submit_predict.checkpoint_path(MF, RUN, "epoch105-step84800", models_root=ROOT)
    assert p == Path(f"/models/{MF}/{RUN}/checkpoints/epoch=105-step=84800.ckpt")


def test_predict_cmd_core_flags():
    cmd = submit_predict.build_predict_cmd(
        Path("coll.yml"),
        Path("/models/x/last.ckpt"),
        MF,
        RUN,
        "last",
        "/data",
        predict_flags={"z_range": [15, 45], "z_reduction": "mip", "reference_pixel_size": 0.1494},
    )
    assert cmd[:2] == ["dynaclr", "predict-triplet"]
    assert cmd[cmd.index("--model-family") + 1] == MF
    assert cmd[cmd.index("--datasets-root") + 1] == "/data"
    assert cmd[cmd.index("--z-range") + 1 : cmd.index("--z-range") + 3] == ["15", "45"]
    assert cmd[cmd.index("--z-reduction") + 1] == "mip"
    assert cmd[cmd.index("--reference-pixel-size") + 1] == "0.1494"


def test_predict_cmd_markers_and_labelfree():
    cmd = submit_predict.build_predict_cmd(
        Path("coll.yml"),
        Path("/c.ckpt"),
        MF,
        RUN,
        "last",
        "/data",
        markers=["SEC61B", "TOMM20"],
        no_labelfree=True,
    )
    assert cmd[cmd.index("--markers") + 1] == "SEC61B,TOMM20"
    assert "--no-labelfree" in cmd


def test_predict_cmd_no_optional_flags():
    cmd = submit_predict.build_predict_cmd(Path("c.yml"), Path("/c.ckpt"), MF, RUN, "last", "/data")
    assert "--markers" not in cmd
    assert "--no-labelfree" not in cmd
    assert "--z-range" not in cmd
