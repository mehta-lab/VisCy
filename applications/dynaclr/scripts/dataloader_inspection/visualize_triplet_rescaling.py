r"""Visualize scale-aware patch rescaling in TripletDataModule.

Loads a few cell patches from an OME-Zarr dataset using TripletDataModule
with ``reference_pixel_size`` set, then plots each patch in two columns:

  Left  — raw patch at ``initial_yx_patch_size`` (larger physical area sampled
           to match the reference pixel size)
  Right — the same patch bilinearly downscaled to ``final_yx_patch_size``
          (what the model actually receives)

Both columns use the same percentile-based grayscale contrast window so
that spatial content differences are visible rather than intensity shifts.
A physical-scale annotation (µm × µm) is printed below each patch.

Usage::

    python visualize_triplet_rescaling.py \\
        --data-path /path/to/data.zarr \\
        --tracks-path /path/to/tracks \\
        --source-channel Phase3D \\
        --z-range 0 5 \\
        --final-yx-patch-size 224 224 \\
        --reference-pixel-size 0.325 \\
        --n-samples 6 \\
        --output rescaling_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from viscy_data.triplet import TripletDataModule, _read_pixel_size

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Visualize TripletDataModule scale-aware patch rescaling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", required=True, help="Path to OME-Zarr plate/position.")
    p.add_argument("--tracks-path", required=True, help="Path to tracks CSV directory.")
    p.add_argument(
        "--source-channel",
        required=True,
        nargs="+",
        help="Channel name(s) to load (e.g. Phase3D).",
    )
    p.add_argument(
        "--z-range",
        required=True,
        type=int,
        nargs=2,
        metavar=("Z_START", "Z_STOP"),
        help="Z-slice range [start, stop).",
    )
    p.add_argument(
        "--final-yx-patch-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("Y", "X"),
        help="Target patch size fed to the model (pixels).",
    )
    p.add_argument(
        "--reference-pixel-size",
        type=float,
        required=True,
        help="X pixel size (µm/px) of the model's training dataset.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=6,
        help="Number of cell patches to visualize.",
    )
    p.add_argument(
        "--z-slice",
        type=int,
        default=None,
        help="Z index within the patch to display. Defaults to middle slice.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("rescaling_comparison.png"),
        help="Output PNG path.",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _percentile_norm(img: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    """Return (vmin, vmax) for percentile-based display."""
    vmin, vmax = np.percentile(img, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _rescale_yx(
    patch: torch.Tensor,
    target_yx: tuple[int, int],
) -> torch.Tensor:
    """Bilinear-rescale YX of a (C, Z, H, W) patch to target_yx."""
    c, z, h, w = patch.shape
    flat = patch.reshape(c * z, 1, h, w).float()
    out = torch.nn.functional.interpolate(
        flat,
        size=target_yx,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return out.reshape(c, z, *target_yx)


def _mid_slice(patch: torch.Tensor, z_idx: int | None) -> np.ndarray:
    """Return a 2-D (H, W) numpy array from (C, Z, H, W), channel 0, given z."""
    z_size = patch.shape[1]
    z = z_idx if z_idx is not None else z_size // 2
    z = max(0, min(z, z_size - 1))
    return patch[0, z].numpy()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    """Run the visualization CLI."""
    args = parse_args()

    final_yx = tuple(args.final_yx_patch_size)
    z_range = tuple(args.z_range)

    # Build the data module with scale-aware rescaling enabled.
    dm = TripletDataModule(
        data_path=args.data_path,
        tracks_path=args.tracks_path,
        source_channel=args.source_channel,
        z_range=z_range,
        final_yx_patch_size=final_yx,
        reference_pixel_size=args.reference_pixel_size,
        batch_size=args.n_samples,
        num_workers=0,
    )
    dm.setup("predict")

    inference_pixel_size = _read_pixel_size(args.data_path)
    initial_yx = dm.initial_yx_patch_size
    scale = args.reference_pixel_size / inference_pixel_size

    print(
        f"Reference pixel size : {args.reference_pixel_size:.4f} µm/px\n"
        f"Inference pixel size : {inference_pixel_size:.4f} µm/px\n"
        f"Scale factor         : {scale:.4f}\n"
        f"initial_yx_patch_size: {initial_yx}\n"
        f"final_yx_patch_size  : {final_yx}\n"
    )

    # Draw samples directly from the dataset (raw, before any transforms).
    n = min(args.n_samples, len(dm.predict_dataset))
    raw_batch = dm.predict_dataset.__getitems__(list(range(n)))
    raw_patches = raw_batch["anchor"]  # (B, C, Z, initial_Y, initial_X)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(8, 4 * n),
        squeeze=False,
    )

    for i in range(n):
        raw = raw_patches[i]  # (C, Z, initial_Y, initial_X)
        rescaled = _rescale_yx(raw, final_yx)  # (C, Z, final_Y, final_X)

        raw_2d = _mid_slice(raw, args.z_slice)
        rescaled_2d = _mid_slice(rescaled, args.z_slice)

        # Shared contrast from the raw patch so differences are spatial only.
        vmin, vmax = _percentile_norm(raw_2d)

        phys_raw_y = initial_yx[0] * inference_pixel_size
        phys_raw_x = initial_yx[1] * inference_pixel_size
        phys_final_y = final_yx[0] * args.reference_pixel_size
        phys_final_x = final_yx[1] * args.reference_pixel_size

        # Left column: raw patch
        ax = axes[i, 0]
        ax.imshow(raw_2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(
            f"Sample {i}  —  raw patch\n{initial_yx[0]}×{initial_yx[1]} px  ({phys_raw_y:.1f}×{phys_raw_x:.1f} µm)",
            fontsize=9,
        )
        ax.axis("off")

        # Right column: rescaled patch
        ax = axes[i, 1]
        ax.imshow(rescaled_2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(
            f"Sample {i}  —  rescaled (model input)\n"
            f"{final_yx[0]}×{final_yx[1]} px  "
            f"({phys_final_y:.1f}×{phys_final_x:.1f} µm)",
            fontsize=9,
        )
        ax.axis("off")

    axes[0, 0].annotate(
        "RAW (initial_yx_patch_size)",
        xy=(0.5, 1.12),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#e74c3c",
    )
    axes[0, 1].annotate(
        "RESCALED (final_yx_patch_size)",
        xy=(0.5, 1.12),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#2ecc71",
    )

    fig.suptitle(
        f"Scale-aware patch rescaling\n"
        f"reference={args.reference_pixel_size} µm/px  ·  "
        f"inference={inference_pixel_size:.4f} µm/px  ·  "
        f"scale={scale:.3f}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
