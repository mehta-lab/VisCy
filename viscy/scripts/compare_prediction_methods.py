"""
Compare predictions from HCSPredictionWriter (out-of-core) vs predict_sliding_windows (in-memory).

Both methods now use the same linear feathering blending algorithm (_blend_in),
so predictions should be numerically identical when using the same model and input data.

This script:
1. Creates a temporary single-FOV zarr with source data
2. Runs HCSPredictionWriter via the trainer to generate reference predictions
3. Runs predict_sliding_windows on the same data
4. Compares the outputs numerically

Expected result: Predictions should match within floating point tolerance (rtol=1e-5).
"""

# %%

import tempfile
from pathlib import Path

import numpy as np
import torch
from iohub import open_ome_zarr
from matplotlib import pyplot as plt

from viscy.data.hcs import HCSDataModule
from viscy.trainer import VisCyTrainer
from viscy.translation.engine import AugmentedPredictionVSUNet, VSUNet
from viscy.translation.predict_writer import HCSPredictionWriter

# =============================================================================
# CONFIGURATION - Edit these paths for your data
# =============================================================================

SOURCE_ZARR = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/0-reconstruct/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
)
CHECKPOINT = Path(
    "/hpc/projects/comp.micro/virtual_staining/models/fcmae-cyto3d-sensor/vscyto3d-logs/hek-a549-ipsc-finetune/checkpoints/epoch=83-step=14532-loss=0.492.ckpt"
)

POSITION = "C/2/001000"
SOURCE_CHANNEL = "Phase3D"
TIMEPOINT = 0
ARRAY_KEY = "0"
Z_WINDOW_SIZE = 15

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

    # Load model
    print(f"\nLoading model from: {CHECKPOINT}")
    model = (
        VSUNet(
            architecture="fcmae",
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "in_stack_depth": Z_WINDOW_SIZE,
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

    # Load source data and normalization stats
    print(f"\nLoading source data from: {SOURCE_ZARR}")
    with open_ome_zarr(SOURCE_ZARR, mode="r") as dataset:
        pos = dataset[POSITION]
        channel_idx = dataset.get_channel_index(SOURCE_CHANNEL)
        source_np = np.asarray(
            pos[ARRAY_KEY][TIMEPOINT : TIMEPOINT + 1, channel_idx : channel_idx + 1]
        )
        norm_meta = pos.zattrs["normalization"][SOURCE_CHANNEL]["fov_statistics"]
        median = norm_meta["median"]
        iqr = norm_meta["iqr"]
    print(f"  Source shape: {source_np.shape}")
    print(f"  Normalization: median={median:.4f}, iqr={iqr:.4f}")

    # Create temporary zarr with single FOV for HCSPredictionWriter
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_zarr = tmp_path / "input.zarr"
        output_zarr = tmp_path / "output.zarr"

        # Write normalized source data to temp zarr
        print(f"\nCreating temporary input zarr: {input_zarr}")
        # IMPORTANT: Convert to float32 FIRST to ensure both pipelines use identical data
        source_normalized = ((source_np - median) / iqr).astype(np.float32)
        with open_ome_zarr(
            input_zarr,
            layout="hcs",
            mode="w",
            channel_names=[SOURCE_CHANNEL],
        ) as ds:
            pos = ds.create_position("0", "0", "0")
            pos.create_image(ARRAY_KEY, source_normalized)  # Already float32
            # Add normalization metadata (already normalized, so identity transform)
            pos.zattrs["normalization"] = {
                SOURCE_CHANNEL: {"fov_statistics": {"median": 0.0, "iqr": 1.0}}
            }

        # =================================================================
        # DEBUG: Compare single window predictions before blending
        # =================================================================
        print("\n" + "=" * 60)
        print("DEBUG: Comparing single Z-window predictions (no blending)")
        print("=" * 60)

        # Get first window input (z=0 to z=Z_WINDOW_SIZE)
        first_window = (
            torch.from_numpy(source_normalized[:, :, :Z_WINDOW_SIZE]).float().to(DEVICE)
        )
        print(f"  First window shape: {first_window.shape}")

        # Predict with VSUNet (same as HCSPredictionWriter uses)
        model.on_predict_start()  # Initialize _predict_pad
        with torch.inference_mode():
            padded = model._predict_pad(first_window)
            vsunet_pred = model.forward(padded)
            vsunet_pred = model._predict_pad.inverse(vsunet_pred)
        vsunet_window = vsunet_pred[0].cpu().numpy()
        print(f"  VSUNet prediction shape: {vsunet_window.shape}")

        # Predict with AugmentedPredictionVSUNet
        vs_debug = (
            AugmentedPredictionVSUNet(
                model=model.model,
                forward_transforms=[lambda t: t],
                inverse_transforms=[lambda t: t],
            )
            .to(DEVICE)
            .eval()
        )
        with torch.inference_mode():
            aug_pred = vs_debug._predict_with_tta(first_window)
        aug_window = aug_pred[0].cpu().numpy()
        print(f"  AugmentedPredictionVSUNet prediction shape: {aug_window.shape}")

        # Compare single window
        window_diff = np.abs(vsunet_window - aug_window)
        print("\n  Single window comparison:")
        print(f"    Max abs diff: {window_diff.max():.6e}")
        print(f"    Mean abs diff: {window_diff.mean():.6e}")
        print(
            f"    Allclose (1e-5): {np.allclose(vsunet_window, aug_window, rtol=1e-5, atol=1e-5)}"
        )
        print(
            f"    Allclose (1e-4): {np.allclose(vsunet_window, aug_window, rtol=1e-4, atol=1e-4)}"
        )
        print("=" * 60 + "\n")

        # =================================================================
        # Full pipeline comparison
        # =================================================================

        # Run HCSPredictionWriter
        print("\nRunning HCSPredictionWriter...")
        dm = HCSDataModule(
            data_path=input_zarr,
            source_channel=[SOURCE_CHANNEL],
            target_channel=["nuclei", "membrane"],  # Dummy, not used for prediction
            z_window_size=Z_WINDOW_SIZE,
            target_2d=False,
            batch_size=1,
            num_workers=0,
        )

        prediction_writer = HCSPredictionWriter(output_store=str(output_zarr))
        trainer = VisCyTrainer(
            logger=False,
            callbacks=[prediction_writer],
            default_root_dir=tmp_path,
        )
        trainer.predict(model, datamodule=dm)

        # Load reference predictions
        print(f"\nLoading reference predictions from: {output_zarr}")
        with open_ome_zarr(output_zarr, mode="r") as dataset:
            for _, pos in dataset.positions():
                reference_pred = np.asarray(pos[ARRAY_KEY][0])  # (C, Z, Y, X)
                break
        print(f"  Reference shape: {reference_pred.shape}")

        # Run predict_sliding_windows
        print("\nRunning predict_sliding_windows...")
        vs = (
            AugmentedPredictionVSUNet(
                model=model.model,
                forward_transforms=[lambda t: t],
                inverse_transforms=[lambda t: t],
            )
            .to(DEVICE)
            .eval()
        )

        vol = torch.from_numpy(source_normalized).float().to(DEVICE)
        with torch.inference_mode():
            pred = vs.predict_sliding_windows(vol, out_channel=2, step=1)

        computed_pred = pred[0].cpu().numpy()  # (C, Z, Y, X)
        print(f"  Computed shape: {computed_pred.shape}")

        # Compare
        metrics = compare_predictions(reference_pred, computed_pred)
        print_comparison_report(metrics)

        # Debug: Compare Z-slice by Z-slice to find where differences start
        print("\n## Z-slice comparison (all slices with max diff > 1e-3)\n")
        print("| Z | Max Abs Diff | Allclose (1e-5) |")
        print("|---|--------------|-----------------|")
        large_diff_slices = []
        for z in range(reference_pred.shape[1]):
            ref_slice = reference_pred[:, z]
            comp_slice = computed_pred[:, z]
            max_diff = np.abs(ref_slice - comp_slice).max()
            is_close = np.allclose(ref_slice, comp_slice, rtol=1e-5, atol=1e-5)
            if max_diff > 1e-3 or z < 20:  # Show first 20 + any with large diff
                print(f"| {z} | {max_diff:.6e} | {is_close} |")
            if max_diff > 1e-3:
                large_diff_slices.append((z, max_diff))

        if large_diff_slices:
            print(f"\nSlices with max diff > 1e-3: {len(large_diff_slices)}")
            print(
                f"Worst slice: Z={large_diff_slices[-1][0]}, diff={large_diff_slices[-1][1]:.4f}"
            )

        # Summary
        print("\n## Summary\n")
        if metrics["allclose_1e-5"]:
            print("✓ Predictions are numerically identical (within rtol=1e-5)")
            print("  Both methods use the same _blend_in linear feathering algorithm.")
        elif metrics["allclose_1e-3"]:
            print("⚠ Predictions are close but not identical (within rtol=1e-3)")
            print("  Small differences may be due to floating point precision.")
        elif metrics["allclose_1e-2"]:
            print("⚠ Predictions are close but not identical (within rtol=1e-2)")
            print("  Check normalization parameters and model state.")
        else:
            print("✗ Predictions differ significantly")
            print(
                "  Check: normalization stats, model checkpoint, input data alignment."
            )

        # Store for plotting
        reference_pred_plot = reference_pred.copy()
        computed_pred_plot = computed_pred.copy()

# %%
# Plot them side by side
Z_INDEX = min(60, reference_pred_plot.shape[1] - 1)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(reference_pred_plot[0, Z_INDEX])
axs[0, 0].set_title("Nuclei Reference (HCSPredictionWriter)")
axs[0, 1].imshow(computed_pred_plot[0, Z_INDEX])
axs[0, 1].set_title("Nuclei Computed (predict_sliding_windows)")
axs[1, 0].imshow(reference_pred_plot[1, Z_INDEX])
axs[1, 0].set_title("Membrane Reference (HCSPredictionWriter)")
axs[1, 1].imshow(computed_pred_plot[1, Z_INDEX])
axs[1, 1].set_title("Membrane Computed (predict_sliding_windows)")
plt.tight_layout()

# %%
