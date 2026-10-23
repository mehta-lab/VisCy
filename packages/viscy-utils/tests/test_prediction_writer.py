"""Tests for prediction writer blending utilities."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from iohub import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer

from viscy_data import HCSDataModule
from viscy_utils.callbacks.prediction_writer import HCSPredictionWriter, _blend_in
from viscy_utils.prediction_metadata import (
    PREDICTED_Z_PLANES_KEY,
    PREDICTION_COMPLETE_KEY,
    completion_marker,
    mark_complete,
    mark_started,
    plane_predict_unfinished,
    predicted_z_planes,
    prediction_complete,
    prediction_run,
    record_z_planes,
    same_run,
    started_marker,
    z_planes_sha256_12,
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
                "settings_sha256_12": None,
            }
        }
        assert prediction_complete(position, ["Nuclei_prediction"], completion_marker([1, Z_SIZE, 8, 8], run))
        assert not prediction_complete(position, ["Nuclei_prediction"], completion_marker([2, Z_SIZE, 8, 8], run))


def test_completion_helpers_keep_other_channels(tmp_path):
    """Completing or restarting one channel must leave the other channels' markers intact."""
    run = prediction_run(array_key="0", z_window_size=1, z_reduction="blend", checkpoint_path=None)
    marker = completion_marker([1, 2, 3, 4], run)
    with open_ome_zarr(tmp_path / "plate.zarr", layout="hcs", mode="w-", channel_names=["A", "B"]) as plate:
        position = plate.create_position("0", "0", "0")
        mark_complete(position, ["A"], marker)
        mark_complete(position, ["B"], marker)
        mark_started(position, ["A"], run)
        assert position.zattrs[PREDICTION_COMPLETE_KEY] == {"A": started_marker(run), "B": marker}
        assert prediction_complete(position, ["B"], marker)
        assert not prediction_complete(position, ["A", "B"], marker)


class _WindowFunctionModule(LightningModule):
    """Predict a nonlinear function of each window's input and position, so blending and window order matter.

    ``center_plane`` returns only the window's center plane (a 2D model, ``target_2d``); ``dtype`` sets the
    prediction dtype.
    """

    def __init__(self, center_plane: bool = False, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.center_plane = center_plane
        self.out_dtype = dtype

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        source = batch["source"]
        offsets = torch.arange(source.shape[-3], dtype=source.dtype, device=source.device).view(1, 1, -1, 1, 1)
        starts = torch.as_tensor([int(z) for z in batch["index"][2]], dtype=source.dtype, device=source.device)
        out = torch.sin(source * (1.0 + 0.1 * offsets)) + 0.01 * starts.view(-1, 1, 1, 1, 1)
        if self.center_plane:
            mid = out.shape[-3] // 2
            out = out[:, :, mid : mid + 1]
        return out.to(self.out_dtype)


class _PerWindowWriter(HCSPredictionWriter):
    """Reference: the per-window zarr read-modify-write (the ``write_input`` path) in every mode."""

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_predict_start(trainer, pl_module)
        self._write_once = False


#: Per-timepoint planes of the two FOVs of ``_write_random_source``.
RANDOM_SOURCE_PLANES = {"0/0/0": [[0, 1, 2], [4, 5, 6]], "0/0/1": [[8, 9, 10], [3, 4, 5]]}


def _write_random_source(path) -> None:
    """Two FOVs of random ``(2, 2, 11, 8, 8)`` float32 Phase3D/Nuclei."""
    rng = np.random.default_rng(0)
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=["Phase3D", "Nuclei"]) as plate:
        for fov in RANDOM_SOURCE_PLANES:
            plate.create_position(*fov.split("/")).create_image(
                "0", rng.normal(size=(2, 2, 11, 8, 8)).astype(np.float32)
            )


def _read_store(path) -> dict:
    with open_ome_zarr(path, mode="r") as plate:
        return {name: (np.asarray(pos["0"]), pos["0"].chunks, dict(pos.zattrs)) for name, pos in plate.positions()}


def _assert_same_bytes(reference: dict, buffered: dict) -> None:
    assert buffered.keys() == reference.keys()
    for name, (voxels, chunks, attrs) in reference.items():
        new_voxels, new_chunks, new_attrs = buffered[name]
        assert (new_voxels.shape, new_voxels.dtype, new_chunks) == (voxels.shape, voxels.dtype, chunks)
        assert new_voxels.tobytes() == voxels.tobytes()
        assert new_attrs == attrs


@pytest.mark.parametrize("mode", ["blend", "center", "target_2d", "z_planes"])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("limit_batches", [None, 5])
def test_buffered_volumes_store_the_same_bytes_as_per_window_writes(tmp_path, mode, batch_size, limit_batches):
    """Assembling each volume in memory and writing it once stores the bytes and markers per-window writes store.

    Covers 3D windows (blend, center), a 2D model's padded center plane (``target_2d``) and a plane selection.
    ``limit_batches=5`` stops inside a volume, so the partial volume flushed at predict end is compared too.
    """
    source = tmp_path / "source.zarr"
    _write_random_source(source)
    planes_file = None
    if mode == "z_planes":
        planes_file = tmp_path / "planes.json"
        planes_file.write_text(json.dumps({"positions": RANDOM_SOURCE_PLANES}))
    stores = {}
    for writer_cls in (_PerWindowWriter, HCSPredictionWriter):
        data_module = HCSDataModule(
            data_path=str(source),
            source_channel=["Phase3D"],
            target_channel=["Nuclei"],
            z_window_size=1 if mode == "z_planes" else 4,
            target_2d=mode == "target_2d",
            batch_size=batch_size,
            num_workers=0,
            yx_patch_size=[8, 8],
            normalizations=[],
            augmentations=[],
            predict_z_planes=planes_file,
        )
        output = tmp_path / f"pred_{writer_cls.__name__}.zarr"
        writer = writer_cls(str(output), z_reduction="center" if mode == "center" else "blend")
        Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            limit_predict_batches=limit_batches,
            callbacks=[writer],
        ).predict(
            _WindowFunctionModule(center_plane=mode == "target_2d"), datamodule=data_module, return_predictions=False
        )
        assert writer._write_once is (writer_cls is HCSPredictionWriter)
        stores[writer_cls] = _read_store(output)
    _assert_same_bytes(stores[_PerWindowWriter], stores[HCSPredictionWriter])


def test_buffered_volume_takes_the_dtype_of_an_existing_array(tmp_path):
    """Overwriting a float32 store with float16 predictions blends in float32, as per-window writes do."""
    source = tmp_path / "source.zarr"
    _write_random_source(source)
    stores = {}
    for writer_cls in (_PerWindowWriter, HCSPredictionWriter):
        for dtype, overwrite in ((torch.float32, False), (torch.float16, True)):
            output = tmp_path / f"pred_{writer_cls.__name__}.zarr"
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
            )
            Trainer(
                accelerator="cpu",
                logger=False,
                enable_progress_bar=False,
                callbacks=[writer_cls(str(output), overwrite=overwrite)],
            ).predict(_WindowFunctionModule(dtype=dtype), datamodule=data_module, return_predictions=False)
        stores[writer_cls] = _read_store(output)
    assert {voxels.dtype for voxels, _, _ in stores[HCSPredictionWriter].values()} == {np.dtype(np.float32)}
    _assert_same_bytes(stores[_PerWindowWriter], stores[HCSPredictionWriter])


class _OnesModule(LightningModule):
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return torch.ones_like(batch["source"])


def _write_source(path, fovs: list[str], z_size: int = 4) -> None:
    """Create a two-channel source plate with one ``(1, 2, Z, 8, 8)`` array per FOV."""
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=["Phase3D", "Nuclei"]) as plate:
        for fov in fovs:
            plate.create_position(*fov.split("/")).create_zeros("0", (1, 2, z_size, 8, 8), dtype=np.float32)


def _ones_data_module(source, *, exclude=None, target="Nuclei") -> HCSDataModule:
    """Depth-4 windows over every FOV of ``source``, predicting ``target``."""
    return HCSDataModule(
        data_path=str(source),
        source_channel=["Phase3D"],
        target_channel=[target],
        z_window_size=4,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[8, 8],
        normalizations=[],
        augmentations=[],
        exclude_fov_names=exclude,
    )


def _predict_ones_with(writer: HCSPredictionWriter, data_module: HCSDataModule, *, limit_batches=None) -> None:
    """Run ``_OnesModule`` through ``writer`` on the CPU."""
    Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        limit_predict_batches=limit_batches,
        callbacks=[writer],
    ).predict(_OnesModule(), datamodule=data_module, return_predictions=False)


def _predict_ones(
    source, output, *, checkpoint_path=None, exclude=None, overwrite=False, limit_batches=None, target="Nuclei"
) -> None:
    """Predict all-ones for every FOV of ``source`` with depth-4 windows into ``output``."""
    writer = HCSPredictionWriter(str(output), overwrite=overwrite, checkpoint_path=checkpoint_path)
    _predict_ones_with(writer, _ones_data_module(source, exclude=exclude, target=target), limit_batches=limit_batches)


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


def test_interrupted_fov_carries_a_started_marker(tmp_path):
    """A FOV the writer created but never finished names the run writing it, unlike pre-marker outputs."""
    source = tmp_path / "source.zarr"
    _write_source(source, ["0/0/0"], z_size=8)
    _predict_ones(source, tmp_path / "pred.zarr", limit_batches=2)  # 2 of 5 depth windows

    run = prediction_run(array_key="0", z_window_size=4, z_reduction="blend", checkpoint_path=None)
    with open_ome_zarr(tmp_path / "pred.zarr", mode="r") as plate:
        assert plate["0/0/0"].zattrs[PREDICTION_COMPLETE_KEY] == {"Nuclei_prediction": started_marker(run)}


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


def test_writer_refuses_a_legacy_channel_after_another_channel_was_marked(tmp_path):
    """Markers are per channel: predicting a new channel into a legacy store vouches for nothing else."""
    source = tmp_path / "source.zarr"
    output = tmp_path / "pred.zarr"
    _write_source(source, ["0/0/0", "0/0/1"])
    with open_ome_zarr(output, layout="hcs", mode="w-", channel_names=["Nuclei_prediction"]) as plate:
        for fov in ("0/0/0", "0/0/1"):
            plate.create_position(*fov.split("/")).create_zeros("0", (1, 1, 4, 8, 8), dtype=np.float32)[:] = 999
    _predict_ones(source, output, target="Other")
    with open_ome_zarr(output, mode="r") as plate:
        assert set(plate["0/0/0"].zattrs[PREDICTION_COMPLETE_KEY]) == {"Other_prediction"}

    with pytest.raises(ValueError, match="before completion markers"):
        _predict_ones(source, output, exclude=["0/0/0"], overwrite=True)

    with open_ome_zarr(output, mode="r") as plate:
        for fov in ("0/0/0", "0/0/1"):
            np.testing.assert_array_equal(plate[fov]["0"][:, 0], 999)
            assert set(plate[fov].zattrs[PREDICTION_COMPLETE_KEY]) == {"Other_prediction"}


def test_channels_appended_to_excluded_fovs_carry_this_runs_started_marker(tmp_path):
    """Appending a channel allocates zeros in every FOV; excluded FOVs record who did, so a resume can finish them."""
    source = tmp_path / "source.zarr"
    output = tmp_path / "pred.zarr"
    _write_source(source, ["0/0/0", "0/0/1"])
    _predict_ones(source, output)
    _predict_ones(source, output, exclude=["0/0/1"], target="Other")

    run = prediction_run(array_key="0", z_window_size=4, z_reduction="blend", checkpoint_path=None)
    complete = completion_marker([1, 4, 8, 8], run)
    with open_ome_zarr(output, mode="r") as plate:
        assert plate["0/0/0"].zattrs[PREDICTION_COMPLETE_KEY] == {
            "Nuclei_prediction": complete,
            "Other_prediction": complete,
        }
        assert plate["0/0/1"].zattrs[PREDICTION_COMPLETE_KEY] == {
            "Nuclei_prediction": complete,
            "Other_prediction": started_marker(run),
        }
        np.testing.assert_array_equal(plate["0/0/1/0"][:, 1], 0)

    _predict_ones(source, output, exclude=["0/0/0"], target="Other", overwrite=True)

    with open_ome_zarr(output, mode="r") as plate:
        for fov in ("0/0/0", "0/0/1"):
            assert plate[fov].zattrs[PREDICTION_COMPLETE_KEY] == {
                "Nuclei_prediction": complete,
                "Other_prediction": complete,
            }
            np.testing.assert_array_equal(plate[fov]["0"][:], 1)


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


def test_writer_refuses_a_checkpoint_replaced_after_construction(tmp_path):
    """The hash must name the weights the model loaded, so a file swapped after the writer saw it is refused."""
    source = tmp_path / "source.zarr"
    output = tmp_path / "pred.zarr"
    _write_source(source, ["0/0/0"])
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    writer = HCSPredictionWriter(str(output), checkpoint_path=str(ckpt))
    ckpt.write_bytes(b"weights-bb")

    with pytest.raises(RuntimeError, match="changed since this writer was constructed"):
        _predict_ones_with(writer, _ones_data_module(source))
    assert not output.exists()


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


def test_writer_refuses_multi_device_predict(tmp_path):
    """Per-process completion bitmaps cannot certify a FOV split across ranks; fail before opening the store."""
    writer = HCSPredictionWriter(str(tmp_path / "pred.zarr"))
    with pytest.raises(NotImplementedError, match="single device"):
        writer.on_predict_start(SimpleNamespace(world_size=2), None)
    assert not (tmp_path / "pred.zarr").exists()


class _SourcePlusOneModule(LightningModule):
    """Predict ``source + 1``, so every written plane names the source plane it came from."""

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return batch["source"] + 1.0


#: Per-timepoint planes for two FOVs of a (T=2, Z=7) source; caps and an interior slab.
SLAB_PLANES = {"0/0/0": [[0, 1], [3, 4, 5]], "0/0/1": [[5, 6], [2, 3, 4]]}


def _write_stamped_source(path) -> None:
    """Two FOVs of ``(2, 2, 7, 8, 8)`` whose Phase3D plane ``(t, z)`` holds ``100 * t + z + 1``."""
    stamp = (100 * np.arange(2)[:, None] + np.arange(7)[None, :] + 1).astype(np.float32)
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=["Phase3D", "Nuclei"]) as plate:
        for fov in SLAB_PLANES:
            data = np.zeros((2, 2, 7, 8, 8), np.float32)
            data[:, 0] = stamp[:, :, None, None]
            plate.create_position(*fov.split("/")).create_image("0", data)


def _predict_planes(source, output, planes_file=None, settings=None) -> HCSDataModule:
    data_module = HCSDataModule(
        data_path=str(source),
        source_channel=["Phase3D"],
        target_channel=["Nuclei"],
        z_window_size=1,
        batch_size=2,
        num_workers=0,
        yx_patch_size=[8, 8],
        normalizations=[],
        augmentations=[],
        predict_z_planes=planes_file,
    )
    Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        callbacks=[HCSPredictionWriter(str(output), overwrite=True, settings_sha256_12=settings)],
    ).predict(_SourcePlusOneModule(), datamodule=data_module, return_predictions=False)
    return data_module


def test_plane_restricted_predict_writes_only_the_listed_planes(tmp_path):
    """Listed planes equal a full predict's; the rest stay at the fill value; the store keeps the full extent."""
    source = tmp_path / "source.zarr"
    _write_stamped_source(source)
    planes_file = tmp_path / "planes.json"
    planes_file.write_text(json.dumps({"positions": SLAB_PLANES, "focus_anchor": "test"}))

    full, slab = tmp_path / "full.zarr", tmp_path / "slab.zarr"
    _predict_planes(source, full)
    data_module = _predict_planes(source, slab, planes_file)
    assert len(data_module.predict_dataset) == sum(len(p) for per_t in SLAB_PLANES.values() for p in per_t)

    with open_ome_zarr(full, mode="r") as full_plate, open_ome_zarr(slab, mode="r") as slab_plate:
        for fov, per_t in SLAB_PLANES.items():
            full_arr = np.asarray(full_plate[fov]["0"][:, 0])
            slab_arr = np.asarray(slab_plate[fov]["0"][:, 0])
            assert slab_arr.shape == full_arr.shape == (2, 7, 8, 8)
            for t, planes in enumerate(per_t):
                others = [z for z in range(7) if z not in planes]
                np.testing.assert_array_equal(slab_arr[t, planes], full_arr[t, planes])
                assert (slab_arr[t, others] == 0).all()
                # Discriminates: the full store holds a prediction on those planes.
                assert (full_arr[t, others] > 0).all()
            position = slab_plate[fov]
            assert predicted_z_planes(position, "Nuclei_prediction") == per_t
            marker = position.zattrs[PREDICTION_COMPLETE_KEY]["Nuclei_prediction"]
            assert marker["z_planes_sha256_12"] == z_planes_sha256_12(SLAB_PLANES)
            assert marker["source_shape"] == [2, 7, 8, 8]
            assert PREDICTED_Z_PLANES_KEY not in full_plate[fov].zattrs
            assert "z_planes_sha256_12" not in full_plate[fov].zattrs[PREDICTION_COMPLETE_KEY]["Nuclei_prediction"]


def test_full_predict_over_a_plane_store_clears_the_recorded_planes(tmp_path):
    """Re-predicting every plane into a plane-restricted store drops its plane record."""
    source = tmp_path / "source.zarr"
    _write_stamped_source(source)
    planes_file = tmp_path / "planes.json"
    planes_file.write_text(json.dumps({"positions": SLAB_PLANES}))
    output = tmp_path / "pred.zarr"
    _predict_planes(source, output, planes_file)
    _predict_planes(source, output)
    with open_ome_zarr(output, mode="r") as plate:
        for fov in SLAB_PLANES:
            assert predicted_z_planes(plate[fov], "Nuclei_prediction") is None
            assert (np.asarray(plate[fov]["0"][:, 0]) > 0).all()


def test_plane_restricted_predict_refuses_a_store_holding_other_planes(tmp_path):
    """A slab predict over another run's planes would leave them around the slab; over its own slab it may replace."""
    source = tmp_path / "source.zarr"
    _write_stamped_source(source)
    planes_file = tmp_path / "planes.json"
    planes_file.write_text(json.dumps({"positions": SLAB_PLANES}))
    full = tmp_path / "full.zarr"
    _predict_planes(source, full)
    with pytest.raises(ValueError, match="other planes"):
        _predict_planes(source, full, planes_file)
    slab = tmp_path / "slab.zarr"
    _predict_planes(source, slab, planes_file, settings="aaaaaaaaaaaa")
    # The same slab from other weights or settings (e.g. --ckpt best) rewrites every plane it holds.
    _predict_planes(source, slab, planes_file, settings="bbbbbbbbbbbb")
    other_planes = tmp_path / "other_planes.json"
    other_planes.write_text(json.dumps({"positions": {fov: [[2], [6]] for fov in SLAB_PLANES}}))
    with pytest.raises(ValueError, match="other planes"):
        _predict_planes(source, slab, other_planes, settings="cccccccccccc")


def test_an_interrupted_plane_predict_is_unfinished(tmp_path):
    """Until a plane run completes its marker has no source_shape, whatever planes are recorded."""
    source = tmp_path / "source.zarr"
    _write_stamped_source(source)
    planes_file = tmp_path / "planes.json"
    planes_file.write_text(json.dumps({"positions": SLAB_PLANES}))
    full, slab = tmp_path / "full.zarr", tmp_path / "slab.zarr"
    _predict_planes(source, full)
    _predict_planes(source, slab, planes_file)
    with open_ome_zarr(full, mode="r") as full_plate, open_ome_zarr(slab, mode="r+") as slab_plate:
        assert not plane_predict_unfinished(full_plate["0/0/0"], "Nuclei_prediction")
        position = slab_plate["0/0/0"]
        assert not plane_predict_unfinished(position, "Nuclei_prediction")
        marker = position.zattrs[PREDICTION_COMPLETE_KEY]["Nuclei_prediction"]
        # An interrupted re-predict of the same slab: started marker, the old planes still recorded.
        mark_started(position, ["Nuclei_prediction"], {k: v for k, v in marker.items() if k != "source_shape"})
        assert predicted_z_planes(position, "Nuclei_prediction") == SLAB_PLANES["0/0/0"]
        assert plane_predict_unfinished(position, "Nuclei_prediction")
        # An interrupted first predict: started marker, no planes yet.
        record_z_planes(position, ["Nuclei_prediction"], None)
        assert plane_predict_unfinished(position, "Nuclei_prediction")


def test_record_z_planes_keeps_other_channels(tmp_path):
    with open_ome_zarr(tmp_path / "p.zarr", layout="hcs", mode="w-", channel_names=["A", "B"]) as plate:
        position = plate.create_position("0", "0", "0")
        record_z_planes(position, ["A", "B"], [[3, 2, 2]])
        record_z_planes(position, ["A"], None)
        assert predicted_z_planes(position, "A") is None
        assert predicted_z_planes(position, "B") == [[2, 3]]
        record_z_planes(position, ["B"], None)
        assert PREDICTED_Z_PLANES_KEY not in position.zattrs


@pytest.mark.parametrize(
    ("planes", "match"),
    [
        ({"0/0/0": [[0]]}, "timepoints"),
        ({"0/0/0": [[0], [7]], "0/0/1": [[0], [0]]}, "outside"),
        ({"0/0/0": [[0], [0]]}, "no entry"),
    ],
)
def test_plane_file_must_cover_every_position_and_timepoint(tmp_path, planes, match):
    source = tmp_path / "source.zarr"
    _write_stamped_source(source)
    planes_file = tmp_path / "planes.json"
    planes_file.write_text(json.dumps({"positions": planes}))
    with pytest.raises((ValueError, KeyError), match=match):
        _predict_planes(source, tmp_path / "pred.zarr", planes_file)


def test_plane_restriction_needs_single_plane_windows(tmp_path):
    source = tmp_path / "source.zarr"
    _write_stamped_source(source)
    planes_file = tmp_path / "planes.json"
    planes_file.write_text(json.dumps({"positions": SLAB_PLANES}))
    data_module = _ones_data_module(source)
    data_module.predict_z_planes = planes_file
    with pytest.raises(ValueError, match="z_window_size=1"):
        data_module.setup("predict")


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
