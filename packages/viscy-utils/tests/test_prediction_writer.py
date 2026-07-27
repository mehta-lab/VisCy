"""Tests for prediction writer blending utilities."""

import numpy as np
import pytest
import torch
from iohub import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer

from viscy_data import HCSDataModule
from viscy_utils.callbacks.prediction_writer import HCSPredictionWriter, _blend_in

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
