"""
Compare predictions from HCSPredictionWriter (out-of-core) vs predict_sliding_windows (in-memory).

This script loads:
1. Pre-computed predictions from a zarr store (generated via `viscy predict` CLI)
2. Raw phase data from the source zarr
3. Runs predict_sliding_windows on the phase data
4. Compares the outputs numerically
"""

# %%

from pathlib import Path

import numpy as np
import torch
from iohub import open_ome_zarr
from matplotlib import pyplot as plt

from viscy.translation.engine import AugmentedPredictionVSUNet, VSUNet

# =============================================================================
# CONFIGURATION - Edit these paths for your data
# =============================================================================

PREDICTION_ZARR = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/1-virtual-stain/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
)
SOURCE_ZARR = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/0-reconstruct/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
)
CHECKPOINT = Path(
    "/hpc/projects/comp.micro/virtual_staining/models/fcmae-cyto3d-sensor/vscyto3d-logs/hek-a549-ipsc-finetune/checkpoints/epoch=83-step=14532-loss=0.492.ckpt"
)

POSITION = "C/2/001000"
SOURCE_CHANNEL = "Phase3D"
PREDICTION_CHANNELS = ["nuclei_prediction", "membrane_prediction"]
TIMEPOINT = 0
ARRAY_KEY = "0"
STEP = 1

# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_predictions(
    reference: np.ndarray,
    computed: np.ndarray,
) -> dict:
    """Compare two prediction arrays and return metrics."""
    diff = np.abs(reference - computed)

    metrics = {
        "shape_match": reference.shape == computed.shape,
        "reference_shape": reference.shape,
        "computed_shape": computed.shape,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "std_abs_diff": float(diff.std()),
        "reference_range": (float(reference.min()), float(reference.max())),
        "computed_range": (float(computed.min()), float(computed.max())),
        "allclose_1e-5": bool(np.allclose(reference, computed, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(np.allclose(reference, computed, rtol=1e-4, atol=1e-4)),
        "allclose_1e-3": bool(np.allclose(reference, computed, rtol=1e-3, atol=1e-3)),
        "allclose_1e-2": bool(np.allclose(reference, computed, rtol=1e-2, atol=1e-2)),
    }

    correlations = []
    for c in range(reference.shape[0]):
        ref_flat = reference[c].flatten()
        comp_flat = computed[c].flatten()
        corr = np.corrcoef(ref_flat, comp_flat)[0, 1]
        correlations.append(float(corr))
    metrics["pearson_correlation_per_channel"] = correlations

    return metrics


def print_comparison_report(metrics: dict) -> None:
    """Print comparison metrics in markdown format."""
    print("\n## Prediction Comparison Report\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Shape Match | {metrics['shape_match']} |")
    print(f"| Reference Shape | {metrics['reference_shape']} |")
    print(f"| Computed Shape | {metrics['computed_shape']} |")
    print(f"| Max Absolute Difference | {metrics['max_abs_diff']:.6e} |")
    print(f"| Mean Absolute Difference | {metrics['mean_abs_diff']:.6e} |")
    print(f"| Std Absolute Difference | {metrics['std_abs_diff']:.6e} |")
    print(
        f"| Reference Range | [{metrics['reference_range'][0]:.4f}, {metrics['reference_range'][1]:.4f}] |"
    )
    print(
        f"| Computed Range | [{metrics['computed_range'][0]:.4f}, {metrics['computed_range'][1]:.4f}] |"
    )
    print(f"| Allclose (rtol=1e-5) | {metrics['allclose_1e-5']} |")
    print(f"| Allclose (rtol=1e-4) | {metrics['allclose_1e-4']} |")
    print(f"| Allclose (rtol=1e-3) | {metrics['allclose_1e-3']} |")
    print(f"| Allclose (rtol=1e-2) | {metrics['allclose_1e-2']} |")

    print("\n### Per-Channel Pearson Correlation\n")
    print("| Channel | Correlation |")
    print("|---------|-------------|")
    for i, corr in enumerate(metrics["pearson_correlation_per_channel"]):
        print(f"| {i} | {corr:.6f} |")


# %%
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load model (same pattern as demo_api.py)
    print(f"\nLoading model from: {CHECKPOINT}")
    model = (
        VSUNet(
            architecture="fcmae",
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "in_stack_depth": 15,
                "encoder_blocks": [3, 3, 9, 3],
                "dims": [96, 192, 384, 768],
                "decoder_conv_blocks": 2,
                "stem_kernel_size": [5, 4, 4],
                "pretraining": False,
            },
            ckpt_path=str(CHECKPOINT),
        )
        .to(DEVICE)
        .eval()
    )

    vs = (
        AugmentedPredictionVSUNet(
            model=model.model,
            forward_transforms=[lambda t: t],
            inverse_transforms=[lambda t: t],
        )
        .to(DEVICE)
        .eval()
    )

    # Load reference predictions (from viscy predict CLI)
    print(f"\nLoading reference predictions from: {PREDICTION_ZARR}")
    with open_ome_zarr(PREDICTION_ZARR, mode="r") as dataset:
        pos = dataset[POSITION]
        channel_indices = [dataset.get_channel_index(ch) for ch in PREDICTION_CHANNELS]
        reference_pred = np.asarray(pos[ARRAY_KEY][TIMEPOINT, channel_indices])
    print(f"  Reference shape: {reference_pred.shape}")

    # Load source data and normalization stats
    print(f"\nLoading source data from: {SOURCE_ZARR}")
    with open_ome_zarr(SOURCE_ZARR, mode="r") as dataset:
        pos = dataset[POSITION]
        channel_idx = dataset.get_channel_index(SOURCE_CHANNEL)
        # Load as (1, 1, Z, Y, X) to match demo_api.py pattern
        source_np = np.asarray(
            pos[ARRAY_KEY][TIMEPOINT : TIMEPOINT + 1, channel_idx : channel_idx + 1]
        )
        # Get normalization stats
        norm_meta = pos.zattrs["normalization"][SOURCE_CHANNEL]["fov_statistics"]
        median = norm_meta["median"]
        iqr = norm_meta["iqr"]
    print(f"  Source shape: {source_np.shape}")
    print(f"  Normalization: median={median:.4f}, iqr={iqr:.4f}")

    # Normalize and convert to tensor
    source_normalized = (source_np - median) / iqr
    vol = torch.from_numpy(source_normalized).float().to(DEVICE)

    # Run sliding window prediction
    print(f"\nRunning predict_sliding_windows (step={STEP})...")
    with torch.inference_mode():
        pred = vs.predict_sliding_windows(vol, out_channel=2, step=STEP)

    computed_pred = pred[0].cpu().numpy()  # (C, Z, Y, X)
    print(f"  Computed shape: {computed_pred.shape}")

    # Compare
    metrics = compare_predictions(reference_pred, computed_pred)
    print_comparison_report(metrics)

    # Summary
    print("\n## Summary\n")
    if metrics["allclose_1e-5"]:
        print("✓ Predictions are numerically identical (within rtol=1e-5)")
    elif metrics["allclose_1e-3"]:
        print("⚠ Predictions are close but not identical (within rtol=1e-3)")
    elif metrics["allclose_1e-2"]:
        print("⚠ Predictions are close but not identical (within rtol=1e-2)")
    else:
        print("✗ Predictions differ significantly")
        print("  This may indicate different blending logic or model state")

# %%
# Plot them side by side
# nucleui and membrane in 2x2 grid
Z_INDEX = 60
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(reference_pred[0, Z_INDEX])
axs[0, 0].set_title("Nuclei Reference")
axs[0, 1].imshow(computed_pred[0, Z_INDEX])
axs[0, 1].set_title("Nuclei Computed")
axs[1, 0].imshow(reference_pred[1, Z_INDEX])
axs[1, 0].set_title("Membrane Reference")
axs[1, 1].imshow(computed_pred[1, Z_INDEX])
axs[1, 1].set_title("Membrane Computed")

# %%
