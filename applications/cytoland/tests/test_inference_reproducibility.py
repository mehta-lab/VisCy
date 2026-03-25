"""Integration tests for inference reproducibility of modular vscyto3d.

Validates that the modular FcmaeUNet produces identical prediction results
to the reference predictions. Tests checkpoint loading and pixel-level
prediction exactness using the production pipeline (HCSDataModule +
HCSPredictionWriter + VisCyTrainer).

The test fixture is a single 512x512 FOV cropped from the mehta-lab
VSCyto3D test dataset with pre-computed normalization metadata.
The reference predictions were generated using the same code and checkpoint.

Tolerance rationale: GPU convolution non-determinism across CUDA/cuDNN
versions and hardware causes small numerical differences in deep ConvNeXt
models. We use the same tolerances as DynaCLR:
  - atol=0.02 for element-wise checks
  - Pearson correlation > 0.999 per channel
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr
from lightning.pytorch import seed_everything
from scipy import stats

from cytoland.engine import FcmaeUNet

# HPC path constants
CHECKPOINT_PATH = Path(
    "/hpc/projects/comp.micro/virtual_staining/models/fcmae-cyto3d-sensor/"
    "vscyto3d-logs/hek-a549-ipsc-finetune/checkpoints/"
    "epoch=83-step=14532-loss=0.492.ckpt"
)

DATA_ZARR_PATH = Path(
    "/hpc/projects/virtual_staining/datasets/mehta-lab/VS_datasets/VSCyto3D/test/vscyto3d_test_fixture.zarr"
)

REFERENCE_ZARR_PATH = Path(
    "/hpc/projects/virtual_staining/datasets/mehta-lab/VS_datasets/VSCyto3D/test/vscyto3d_test_reference.zarr"
)

HPC_PATHS_AVAILABLE = all(p.exists() for p in [CHECKPOINT_PATH, DATA_ZARR_PATH, REFERENCE_ZARR_PATH])
GPU_AVAILABLE = torch.cuda.is_available()

requires_hpc_and_gpu = pytest.mark.skipif(
    not (HPC_PATHS_AVAILABLE and GPU_AVAILABLE),
    reason="Requires HPC data paths and CUDA GPU",
)

# Model configuration — matches the fine-tuned vscyto3d checkpoint
# (from finetune_vscyto3d.py:163-174).
MODEL_CONFIG = {
    "in_channels": 1,
    "out_channels": 2,
    "encoder_blocks": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768],
    "decoder_conv_blocks": 2,
    "stem_kernel_size": (5, 4, 4),
    "in_stack_depth": 15,
    "pretraining": False,
}

# Source/target channel configuration.
SOURCE_CHANNEL = "Phase3D"
TARGET_CHANNELS = ["Membrane", "Nuclei"]

# GPU non-determinism tolerance for FCMAE/ConvNeXt convolutions.
ATOL = 0.02
RTOL = 1e-2
MIN_PEARSON_R = 0.999


def _build_module(checkpoint_path):
    """Build FcmaeUNet and load pretrained checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to Lightning checkpoint file.

    Returns
    -------
    tuple[FcmaeUNet, object]
        Module and load_state_dict result.
    """
    module = FcmaeUNet(model_config=MODEL_CONFIG)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    result = module.load_state_dict(ckpt["state_dict"])
    return module, result


@requires_hpc_and_gpu
@pytest.mark.hpc_integration
def test_checkpoint_loads_into_modular_fcmae_unet(checkpoint_path):
    """Checkpoint loads without state dict key mismatches."""
    seed_everything(42)
    module, result = _build_module(checkpoint_path)

    assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys}"
    assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"

    # Smoke-test forward pass with correct input shape.
    x = torch.randn(1, MODEL_CONFIG["in_channels"], MODEL_CONFIG["in_stack_depth"], 64, 64)
    module.eval()
    with torch.no_grad():
        output = module(x)
    assert output.shape[0] == 1
    assert output.shape[1] == MODEL_CONFIG["out_channels"]


@requires_hpc_and_gpu
@pytest.mark.hpc_integration
def test_predict_and_match_reference(
    tmp_path,
    checkpoint_path,
    data_zarr_path,
    reference_zarr_path,
):
    """Predict using production pipeline and compare against reference.

    Uses HCSDataModule + HCSPredictionWriter + VisCyTrainer,
    following the demo_vscyto3d.py pattern.
    """
    from viscy_data.hcs import HCSDataModule
    from viscy_transforms import NormalizeSampled
    from viscy_utils.callbacks import HCSPredictionWriter
    from viscy_utils.trainer import VisCyTrainer

    seed_everything(42)

    module, _ = _build_module(checkpoint_path)
    module.eval()

    # Single FOV path, following demo_vscyto3d.py pattern.
    fov_path = data_zarr_path / "plate/0/0"
    datamodule = HCSDataModule(
        data_path=str(fov_path),
        source_channel=SOURCE_CHANNEL,
        target_channel=TARGET_CHANNELS,
        z_window_size=MODEL_CONFIG["in_stack_depth"],
        batch_size=2,
        num_workers=0,
        normalizations=[
            NormalizeSampled(
                keys=[SOURCE_CHANNEL],
                level="fov_statistics",
                subtrahend="mean",
                divisor="std",
            )
        ],
    )

    output_path = tmp_path / "predictions.zarr"
    writer = HCSPredictionWriter(str(output_path))

    trainer = VisCyTrainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        callbacks=[writer],
        inference_mode=True,
        enable_progress_bar=False,
        logger=False,
    )

    trainer.predict(model=module, datamodule=datamodule, return_predictions=False)
    assert output_path.exists(), f"Output zarr not written at {output_path}"

    # --- Compare predictions against reference ---
    pred_plate = open_ome_zarr(str(output_path), mode="r")
    ref_plate = open_ome_zarr(str(reference_zarr_path), mode="r")

    pred_positions = dict(pred_plate.positions())
    ref_positions = dict(ref_plate.positions())

    assert set(pred_positions.keys()) == set(ref_positions.keys()), (
        f"Position mismatch: pred={set(pred_positions.keys())} vs ref={set(ref_positions.keys())}"
    )

    for pos_name in sorted(ref_positions.keys()):
        pred_pos = pred_positions[pos_name]
        ref_pos = ref_positions[pos_name]

        pred_img = np.asarray(pred_pos["0"][:], dtype=np.float32)
        ref_img = np.asarray(ref_pos["0"][:], dtype=np.float32)

        assert pred_img.shape == ref_img.shape, (
            f"Shape mismatch at {pos_name}: pred={pred_img.shape} vs ref={ref_img.shape}"
        )

        n_channels = pred_img.shape[1]
        for ch in range(n_channels):
            pred_ch = pred_img[:, ch].flatten().astype(np.float64)
            ref_ch = ref_img[:, ch].flatten().astype(np.float64)

            if np.all(ref_ch == 0) and np.all(pred_ch == 0):
                continue

            r, _ = stats.pearsonr(pred_ch, ref_ch)
            assert r > MIN_PEARSON_R, f"Pearson r={r:.6f} < {MIN_PEARSON_R} at position {pos_name}, channel {ch}"

            np.testing.assert_allclose(
                pred_img[:, ch],
                ref_img[:, ch],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Prediction exceeds tolerance at position {pos_name}, channel {ch}",
            )

    pred_plate.close()
    ref_plate.close()
