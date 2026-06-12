"""Generate a CELL-Diff ODE denoising trajectory for a single FOV and timepoint.

Reads a phase-contrast volume from an OME-Zarr store, runs the CELLDiff ODE
integrator for ``num_steps`` steps, and saves all intermediate states to a new
OME-Zarr with shape (num_steps, 1, D, H, W).

The output T axis indexes ODE steps: T=0 is pure Gaussian noise, T=-1 is the
final predicted fluorescence image.

Usage
-----
uv run python applications/dynacell/tests/generate_celldiff_trajectory.py \\
    --ckpt-path /path/to/last.ckpt \\
    --zarr-path /path/to/input.zarr \\
    --output-path /path/to/trajectory.zarr \\
    [--fov-idx 0] [--t-idx 0] [--phase-channel Phase3D] [--num-steps 100] [--device cuda]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from iohub.ngff import open_ome_zarr

from dynacell.engine import DynacellFlowMatching


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True, type=Path, help="Lightning checkpoint (.ckpt)")
    p.add_argument("--zarr-path", required=True, type=Path, help="Input OME-Zarr with phase data")
    p.add_argument("--output-path", required=True, type=Path, help="Output trajectory OME-Zarr")
    p.add_argument("--fov-idx", type=int, default=0, help="0-based FOV index (default: 0)")
    p.add_argument("--t-idx", type=int, default=0, help="Timepoint index within FOV (default: 0)")
    p.add_argument("--phase-channel", default="Phase3D", help="Phase channel name (default: Phase3D)")
    p.add_argument("--num-steps", type=int, default=100, help="ODE integration steps (default: 100)")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.ckpt_path}")
    model = DynacellFlowMatching.load_from_checkpoint(args.ckpt_path, map_location=args.device)
    model.eval()

    patch_d, patch_h, patch_w = model.model.net.input_spatial_size
    print(f"Model input_spatial_size: ({patch_d}, {patch_h}, {patch_w})")

    print(f"Reading zarr: {args.zarr_path}")
    with open_ome_zarr(args.zarr_path, mode="r") as plate:
        positions = list(plate.positions())
        if args.fov_idx >= len(positions):
            raise ValueError(f"fov_idx={args.fov_idx} out of range ({len(positions)} positions)")
        pos_name, pos = positions[args.fov_idx]
        print(f"  FOV [{args.fov_idx}]: {pos_name}  data shape: {pos.data.shape}")
        ch_idx = pos.get_channel_index(args.phase_channel)
        phase_vol = np.asarray(pos.data[args.t_idx, ch_idx])  # (D, H, W)

    print(f"  Extracted phase shape: {phase_vol.shape} at t={args.t_idx}, channel={args.phase_channel!r}")

    d, h, w = phase_vol.shape
    for axis, (size, patch) in enumerate(zip((d, h, w), (patch_d, patch_h, patch_w))):
        if size < patch:
            raise ValueError(
                f"Phase dim {axis} ({size}) is smaller than model patch size ({patch}). "
                "Provide a zarr with sufficient spatial extent."
            )

    phase_crop = phase_vol[:patch_d, :patch_h, :patch_w]
    print(f"  Cropped to model patch: {phase_crop.shape}")

    phase_tensor = torch.from_numpy(phase_crop).float().unsqueeze(0).unsqueeze(0).to(args.device)

    print(f"Generating trajectory ({args.num_steps} steps) on {args.device}...")
    trajectory = model.model.generate_trajectory(phase_tensor, num_steps=args.num_steps)
    # trajectory: (num_steps, B=1, C, D, H, W) → squeeze batch → (num_steps, C, D, H, W)
    trajectory_np = trajectory[:, 0].cpu().numpy().astype(np.float32)
    print(f"  Trajectory shape: {trajectory_np.shape}")

    print(f"Saving to {args.output_path}")
    with open_ome_zarr(
        args.output_path,
        layout="fov",
        mode="w",
        channel_names=["prediction"],
    ) as out:
        out.create_image(
            "0",
            trajectory_np,
            chunks=(1, 1, patch_d, patch_h, patch_w),
        )
    print(f"Done. Output shape: {trajectory_np.shape}  (T=ODE step, C=1, D, H, W)")


if __name__ == "__main__":
    main()
