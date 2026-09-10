"""Tests for prediction writer blending utilities."""

import hashlib

import numpy as np
import pytest
import torch
from iohub import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer

from viscy_data import HCSDataModule
from viscy_utils.callbacks.prediction_writer import HCSPredictionWriter, _blend_in
from viscy_utils.prediction_metadata import (
    PREDICTION_COMPLETE_KEY,
    clear_completion,
    completion_marker,
    mark_complete,
    prediction_complete,
    prediction_run,
    same_run,
)

Z_SIZE = 16
Z_WINDOW = 8


class _ZStampModule(LightningModule):
    """Predict a per-plane stamp ``10 * z_window_start + offset_in_window``.

    Every plane of the assembled volume then names the forward pass and the
    within-window position it came from, so the two ``z_reduction`` modes are
    distinguishable from the store alone.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        source = batch["source"]
        depth = source.shape[-3]
        offsets = torch.arange(depth, dtype=torch.float32, device=source.device)
        starts = torch.as_tensor([int(z) for z in batch["index"][2]], dtype=torch.float32, device=source.device)
        stamp = 10.0 * starts.view(-1, 1) + offsets.view(1, -1)
        return stamp.view(-1, 1, depth, 1, 1).expand(-1, 1, depth, *source.shape[-2:]).clone()


def _run_predict(tmp_path, z_reduction):
    """Predict a synthetic single-position plate and return the written CZYX volume."""
    source_store = tmp_path / f"source_{z_reduction}.zarr"
    with open_ome_zarr(source_store, layout="hcs", mode="w-", channel_names=["Phase3D", "Nuclei"]) as plate:
        position = plate.create_position("0", "0", "0")
        position.create_zeros("0", (1, 2, Z_SIZE, 8, 8), dtype=np.float32)
    data_module = HCSDataModule(
        data_path=str(source_store),
        source_channel=["Phase3D"],
        target_channel=["Nuclei"],
        z_window_size=Z_WINDOW,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[8, 8],
        normalizations=[],
        augmentations=[],
    )
    output_store = tmp_path / f"prediction_{z_reduction}.zarr"
    Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        callbacks=[HCSPredictionWriter(str(output_store), z_reduction=z_reduction)],
    ).predict(_ZStampModule(), datamodule=data_module, return_predictions=False)
    with open_ome_zarr(output_store, mode="r") as plate:
        return np.asarray(plate["0/0/0"]["0"][0, 0, :, 0, 0])


def test_predict_z_reduction_blend_averages_every_covering_window(tmp_path):
    """``blend`` makes each plane the unweighted mean of the windows covering it."""
    written = _run_predict(tmp_path, "blend")
    n_windows = Z_SIZE - Z_WINDOW + 1
    expected = [
        np.mean([10 * s + (p - s) for s in range(max(0, p - Z_WINDOW + 1), min(n_windows - 1, p) + 1)])
        for p in range(Z_SIZE)
    ]
    np.testing.assert_allclose(written, expected, rtol=1e-5)


def test_predict_z_reduction_center_takes_one_pass_per_plane(tmp_path):
    """``center`` gives each plane the window centered on it, and covers the edges."""
    written = _run_predict(tmp_path, "center")
    center = Z_WINDOW // 2
    last = Z_SIZE - Z_WINDOW
    expected = []
    for p in range(Z_SIZE):
        # Interior planes come from their own window's center; the leading planes
        # from window 0 and the trailing ones from the last window, since no
        # window is centered on them.
        start = min(max(0, p - center), last)
        expected.append(10 * start + (p - start))
    np.testing.assert_array_equal(written, expected)
    interior = range(center, last + center + 1)
    assert all(written[p] % 10 == center for p in interior)


def test_completion_marker_layout(tmp_path):
    """One position attribute maps each prediction channel to what produced it.

    The launcher's resume check and test fixtures read this literal layout, so a
    change here is a contract change, not a refactor.
    """
    _run_predict(tmp_path, "blend")
    run = prediction_run(array_key="0", z_window_size=Z_WINDOW, z_reduction="blend", checkpoint_path=None)
    with open_ome_zarr(tmp_path / "prediction_blend.zarr", mode="r") as plate:
        position = plate["0/0/0"]
        assert position.zattrs[PREDICTION_COMPLETE_KEY] == {
            "Nuclei_prediction": {
                "source_shape": [1, Z_SIZE, 8, 8],
                "array_key": "0",
                "z_window_size": Z_WINDOW,
                "z_reduction": "blend",
                "checkpoint_path": None,
                "checkpoint_sha256_12": None,
            }
        }
        assert prediction_complete(position, ["Nuclei_prediction"], completion_marker([1, Z_SIZE, 8, 8], run))
        assert not prediction_complete(position, ["Nuclei_prediction"], completion_marker([2, Z_SIZE, 8, 8], run))


def test_completion_helpers_keep_other_channels(tmp_path):
    """Marking or clearing one channel must leave the other channels' markers intact."""
    run = prediction_run(array_key="0", z_window_size=1, z_reduction="blend", checkpoint_path=None)
    marker = completion_marker([1, 2, 3, 4], run)
    with open_ome_zarr(tmp_path / "plate.zarr", layout="hcs", mode="w-", channel_names=["A", "B"]) as plate:
        position = plate.create_position("0", "0", "0")
        mark_complete(position, ["A"], marker)
        mark_complete(position, ["B"], marker)
        clear_completion(position, ["A"])
        assert position.zattrs[PREDICTION_COMPLETE_KEY] == {"B": marker}
        assert prediction_complete(position, ["B"], marker)
        assert not prediction_complete(position, ["A", "B"], marker)


class _OnesModule(LightningModule):
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return torch.ones_like(batch["source"])


def _write_source(path, fovs: list[str], z_size: int = 4) -> None:
    """Create a two-channel source plate with one ``(1, 2, Z, 8, 8)`` array per FOV."""
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=["Phase3D", "Nuclei"]) as plate:
        for fov in fovs:
            plate.create_position(*fov.split("/")).create_zeros("0", (1, 2, z_size, 8, 8), dtype=np.float32)


def _predict_ones(source, output, *, checkpoint_path=None, exclude=None, overwrite=False, limit_batches=None) -> None:
    """Predict all-ones for every FOV of ``source`` with depth-4 windows into ``output``."""
    data_module = HCSDataModule(
        data_path=str(source),
        source_channel=["Phase3D"],
        target_channel=["Nuclei"],
        z_window_size=4,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[8, 8],
        normalizations=[],
        augmentations=[],
        exclude_fov_names=exclude,
    )
    writer = HCSPredictionWriter(str(output), overwrite=overwrite, checkpoint_path=checkpoint_path)
    Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        limit_predict_batches=limit_batches,
        callbacks=[writer],
    ).predict(_OnesModule(), datamodule=data_module, return_predictions=False)


def test_appending_a_channel_keeps_existing_predictions(tmp_path):
    """A store holding another model's channel gains ours; its voxels and marker are untouched."""
    source = tmp_path / "source.zarr"
    output = tmp_path / "pred.zarr"
    _write_source(source, ["0/0/0", "0/0/1"])
    other_run = prediction_run(array_key="0", z_window_size=4, z_reduction="blend", checkpoint_path=None)
    with open_ome_zarr(output, layout="hcs", mode="w-", channel_names=["Other_prediction"]) as plate:
        for fov in ("0/0/0", "0/0/1"):
            position = plate.create_position(*fov.split("/"))
            position.create_zeros("0", (1, 1, 4, 8, 8), dtype=np.float32)[:] = 7
            mark_complete(position, ["Other_prediction"], completion_marker([1, 4, 8, 8], other_run))

    _predict_ones(source, output)

    with open_ome_zarr(output, mode="r") as plate:
        assert plate.channel_names == ["Other_prediction", "Nuclei_prediction"]
        for fov in ("0/0/0", "0/0/1"):
            position = plate[fov]
            assert position.channel_names == ["Other_prediction", "Nuclei_prediction"]
            np.testing.assert_array_equal(position["0"][:, 0], 7)
            np.testing.assert_array_equal(position["0"][:, 1], 1)
            marker = position.zattrs[PREDICTION_COMPLETE_KEY]
            assert marker["Other_prediction"] == completion_marker([1, 4, 8, 8], other_run)
            assert "Nuclei_prediction" in marker


def test_writer_refuses_positions_that_would_disagree_on_channel_order(tmp_path):
    """Channel indices are plate-wide, so positions must end up with one channel order; check before writing."""
    source = tmp_path / "source.zarr"
    output = tmp_path / "pred.zarr"
    _write_source(source, ["0/0/0", "0/0/1"])
    with open_ome_zarr(output, layout="hcs", mode="w-", channel_names=["Other_prediction"]) as plate:
        for fov in ("0/0/0", "0/0/1"):
            plate.create_position(*fov.split("/")).create_zeros("0", (1, 1, 4, 8, 8), dtype=np.float32)
    # Separate handles: positions created in one session share one channel list in memory.
    for fov, channel in (("0/0/0", "Nuclei_prediction"), ("0/0/1", "Third")):
        with open_ome_zarr(output, mode="r+") as plate:
            plate[fov].append_channel(channel, resize_arrays=True)

    with pytest.raises(ValueError, match="channel order"):
        _predict_ones(source, output, overwrite=True)

    with open_ome_zarr(output, mode="r") as plate:
        assert plate["0/0/1"].channel_names == ["Other_prediction", "Third"]


def test_interrupted_fov_carries_an_empty_marker(tmp_path):
    """A FOV the writer created but never finished is marked as such, unlike pre-marker outputs."""
    source = tmp_path / "source.zarr"
    _write_source(source, ["0/0/0"], z_size=8)
    _predict_ones(source, tmp_path / "pred.zarr", limit_batches=2)  # 2 of 5 depth windows

    with open_ome_zarr(tmp_path / "pred.zarr", mode="r") as plate:
        assert plate["0/0/0"].zattrs[PREDICTION_COMPLETE_KEY] == {}


def test_writer_refuses_fovs_predicted_before_markers_existed(tmp_path):
    """Channels without any completion attribute cannot be verified, so the store is off limits."""
    source = tmp_path / "source.zarr"
    output = tmp_path / "pred.zarr"
    _write_source(source, ["0/0/0", "0/0/1"])
    with open_ome_zarr(output, layout="hcs", mode="w-", channel_names=["Nuclei_prediction"]) as plate:
        for fov in ("0/0/0", "0/0/1"):
            plate.create_position(*fov.split("/")).create_zeros("0", (1, 1, 4, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="before completion markers"):
        _predict_ones(source, output, exclude=["0/0/0"], overwrite=True)

    with open_ome_zarr(output, mode="r") as plate:
        assert all(PREDICTION_COMPLETE_KEY not in pos.zattrs for _, pos in plate.positions())
        np.testing.assert_array_equal(plate["0/0/1/0"][:], 0)


def test_marker_records_the_checkpoint_content(tmp_path):
    """The marker names the checkpoint and hashes its content, so a moved copy still matches."""
    source = tmp_path / "source.zarr"
    _write_source(source, ["0/0/0"])
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    _predict_ones(source, tmp_path / "pred.zarr", checkpoint_path=str(ckpt))

    with open_ome_zarr(tmp_path / "pred.zarr", mode="r") as plate:
        marker = plate["0/0/0"].zattrs[PREDICTION_COMPLETE_KEY]["Nuclei_prediction"]
    assert marker["checkpoint_path"] == str(ckpt)
    assert marker["checkpoint_sha256_12"] == hashlib.sha256(b"weights-a").hexdigest()[:12]
    moved = tmp_path / "moved.ckpt"
    moved.write_bytes(b"weights-a")
    other = tmp_path / "b.ckpt"
    other.write_bytes(b"weights-b")
    assert same_run(marker, prediction_run(array_key="0", z_window_size=4, z_reduction="blend", checkpoint_path=moved))
    assert not same_run(
        marker, prediction_run(array_key="0", z_window_size=4, z_reduction="blend", checkpoint_path=other)
    )


def test_writer_refuses_to_mix_checkpoints_across_fovs(tmp_path):
    """FOVs outside the run that other weights completed make the store off limits, before any write."""
    source = tmp_path / "source.zarr"
    output = tmp_path / "pred.zarr"
    _write_source(source, ["0/0/0", "0/0/1"])
    ckpt_a = tmp_path / "a.ckpt"
    ckpt_a.write_bytes(b"weights-a")
    ckpt_b = tmp_path / "b.ckpt"
    ckpt_b.write_bytes(b"weights-b")
    _predict_ones(source, output, checkpoint_path=str(ckpt_a))
    with open_ome_zarr(output, mode="r") as plate:
        before = {name: pos.zattrs[PREDICTION_COMPLETE_KEY] for name, pos in plate.positions()}

    with pytest.raises(ValueError, match="different checkpoint"):
        _predict_ones(source, output, checkpoint_path=str(ckpt_b), exclude=["0/0/0"], overwrite=True)

    with open_ome_zarr(output, mode="r") as plate:
        assert {name: pos.zattrs[PREDICTION_COMPLETE_KEY] for name, pos in plate.positions()} == before


def test_writer_rejects_unknown_z_reduction():
    """An unknown reduction fails at construction, not silently mid-predict."""
    with pytest.raises(ValueError, match="z_reduction"):
        HCSPredictionWriter("unused.zarr", z_reduction="mean")  # type: ignore[arg-type]


def test_blend_in_consistency():
    """Verify _blend_in produces identical results for torch and numpy inputs."""
    depth = 5
    shape_4d = (2, depth, 8, 8)  # C, Z, Y, X (numpy from HCSPredictionWriter)

    rng = np.random.default_rng(42)
    old_np = rng.random(shape_4d).astype(np.float32)
    new_np = rng.random(shape_4d).astype(np.float32)
    old_torch = torch.from_numpy(old_np).unsqueeze(0)  # (1, C, Z, Y, X)
    new_torch = torch.from_numpy(new_np).unsqueeze(0)

    z_slice = slice(2, 2 + depth)

    result_np = _blend_in(old_np, new_np, z_slice)
    result_torch = _blend_in(old_torch, new_torch, z_slice)

    np.testing.assert_allclose(result_np, result_torch.squeeze(0).numpy(), rtol=1e-5, atol=1e-5)


def test_blend_in_zero_start():
    """Verify _blend_in returns new_stack unchanged when z_slice starts at 0."""
    old = np.ones((2, 5, 8, 8), dtype=np.float32)
    new = np.zeros((2, 5, 8, 8), dtype=np.float32)
    result = _blend_in(old, new, slice(0, 5))
    np.testing.assert_array_equal(result, new)


def test_blend_in_torch_preserves_dtype():
    """Verify _blend_in preserves torch tensor dtype."""
    old = torch.ones(1, 2, 5, 8, 8, dtype=torch.float32)
    new = torch.zeros(1, 2, 5, 8, 8, dtype=torch.float32)
    result = _blend_in(old, new, slice(2, 7))
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
