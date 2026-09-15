"""Save the full CELL-Diff ODE trajectory for one FOV as per-step OME-Zarr stores.

For a single FOV of a test set, runs the CELL-Diff sliding-window ODE integrator
and writes each of the ``num_steps`` intermediate states as a separate
eval-compatible OME-Zarr store: ``step1.zarr`` (ODE index 0, near-noise) …
``step{num_steps}.zarr`` (ODE index -1, final prediction). Each store is an HCS
plate holding the single FOV at its original ``row/col/fov`` path, with the
prediction channel named ``<target_channel>_prediction`` — so the existing
``dynacell evaluate`` pipeline can be run on each store with ``limit_positions=1``
to trace how metrics evolve across sampling steps.

Normalization and FOV iteration reuse ``HCSDataModule`` in the ``predict`` stage,
so the model input matches the real predict runs exactly (e.g. CELL-Diff nucleus
uses ``MinMaxSampled(keys=[Phase3D], level=timepoint_statistics)``).

Tiling uses ``sliding_window`` semantics (independent, non-overlapping patches):
step ``i`` is at the same ODE time across all Z tiles, which keeps the per-step
volumes consistent for a metric-vs-step curve.

Usage
-----
uv run python applications/dynacell/tests/generate_celldiff_trajectory_eval.py \\
    --ckpt-path /path/to/last.ckpt \\
    --test-zarr /path/to/test/cell.zarr \\
    --output-dir /path/to/virtualize_sample_traj/nucleus \\
    [--fov-idx 0] [--num-steps 100] [--source-channel Phase3D] \\
    [--target-channel Nuclei] [--z-window 40] [--device cuda]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from iohub.ngff import open_ome_zarr

from dynacell.engine import DynacellFlowMatching
from viscy_data.hcs import HCSDataModule
from viscy_transforms import MinMaxSampled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True, type=Path, help="Lightning checkpoint (.ckpt)")
    p.add_argument("--test-zarr", required=True, type=Path, help="Test OME-Zarr HCS store (GT plate)")
    p.add_argument("--output-dir", required=True, type=Path, help="Directory for step*.zarr stores")
    p.add_argument("--fov-idx", type=int, default=0, help="0-based FOV index (default: 0)")
    p.add_argument("--num-steps", type=int, default=100, help="ODE integration steps (default: 100)")
    p.add_argument("--source-channel", default="Phase3D", help="Source channel name (default: Phase3D)")
    p.add_argument("--target-channel", default="Nuclei", help="Target channel name (default: Nuclei)")
    p.add_argument("--z-window", type=int, default=40, help="Z window size (default: 40)")
    p.add_argument(
        "--predict-method",
        default="sliding_window",
        choices=["sliding_window", "iterative"],
        help="Tiling method for the trajectory (default: sliding_window)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        nargs=3,
        default=[4, 256, 256],
        metavar=("OD", "OH", "OW"),
        help="Overlap (Z Y X) for --predict-method iterative (default: 4 256 256)",
    )
    p.add_argument(
        "--save-mode",
        default="trajectory",
        choices=["trajectory", "denoise"],
        help=(
            "trajectory: save raw ODE state xt per step (stepN.zarr). "
            "denoise: save the clean-target estimate x1 = xt + (1-t)*v per step "
            "(stepN_denoise.zarr); requires --predict-method iterative (default: trajectory)"
        ),
    )
    p.add_argument("--seed", type=int, default=None, help="Optional manual seed for the noise init")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available)",
    )
    return p.parse_args()


def select_fov_name(test_zarr: Path, fov_idx: int) -> str:
    """Return the ``row/col/fov`` position name at ``fov_idx`` in the plate."""
    with open_ome_zarr(test_zarr, mode="r") as plate:
        names = [name for name, _ in plate.positions()]
    if fov_idx >= len(names):
        raise ValueError(f"fov_idx={fov_idx} out of range ({len(names)} positions)")
    return names[fov_idx]


def save_gt(test_zarr: Path, fov_name: str, output_dir: Path) -> None:
    """Copy the chosen FOV (all channels, raw) to ``output_dir/gt.zarr``.

    The GT store mirrors the FOV's ``row/col/fov`` path and keeps the original
    channel names (including ``Nuclei``), so ``dynacell evaluate`` can pair it
    with the step predictions.
    """
    with open_ome_zarr(test_zarr, mode="r") as plate:
        pos = plate[fov_name]
        data = np.asarray(pos.data)  # (T, C, Z, Y, X)
        channel_names = list(pos.channel_names)
    row, col, fov = fov_name.split("/")
    gt_path = output_dir / "gt.zarr"
    with open_ome_zarr(gt_path, layout="hcs", mode="w", channel_names=channel_names) as out:
        position = out.create_position(row, col, fov)
        position.create_image("0", data)
    print(f"Wrote GT to {gt_path}  shape={data.shape}  channels={channel_names}")


def main() -> None:
    args = parse_args()

    if args.save_mode == "denoise" and args.predict_method != "iterative":
        raise ValueError("--save-mode denoise requires --predict-method iterative")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"Loading model from {args.ckpt_path}")
    model = DynacellFlowMatching.load_from_checkpoint(args.ckpt_path, map_location=args.device)
    model.eval()

    fov_name = select_fov_name(args.test_zarr, args.fov_idx)
    print(f"FOV [{args.fov_idx}]: {fov_name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # In denoise mode gt.zarr already exists and is identical; skip re-writing it to
    # avoid racing a concurrent eval sweep that reads gt.zarr.
    if args.save_mode == "denoise":
        print("save-mode=denoise: skipping gt.zarr write (already present)")
    else:
        save_gt(args.test_zarr, fov_name, args.output_dir)

    # Reuse the predict pipeline so normalization matches the real predict runs.
    dm = HCSDataModule(
        data_path=str(args.test_zarr),
        source_channel=args.source_channel,
        target_channel=args.target_channel,
        z_window_size=args.z_window,
        batch_size=1,
        num_workers=0,
        yx_patch_size=(512, 512),
        normalizations=[
            MinMaxSampled(keys=[args.source_channel], level="timepoint_statistics"),
        ],
        augmentations=[],
        include_fov_names=[fov_name],
    )
    dm.setup("predict")

    batch = next(iter(dm.predict_dataloader()))
    source = batch["source"].to(args.device)  # (1, 1, Z, Y, X), normalized
    print(f"Normalized source shape: {tuple(source.shape)}")

    print(f"Generating {args.save_mode} ({args.num_steps} steps, method={args.predict_method}) on {args.device}...")
    if args.save_mode == "denoise":
        trajectory = model.model.generate_iterative_denoise_trajectory(
            source, num_steps=args.num_steps, overlap_size=tuple(args.overlap)
        )
    elif args.predict_method == "iterative":
        trajectory = model.model.generate_iterative_trajectory(
            source, num_steps=args.num_steps, overlap_size=tuple(args.overlap)
        )
    else:
        trajectory = model.model.generate_sliding_window_trajectory(source, num_steps=args.num_steps)
    # (num_steps, B=1, C, Z, Y, X) → squeeze batch → (num_steps, C, Z, Y, X)
    trajectory = trajectory[:, 0].cpu().numpy().astype(np.float32)
    print(f"Trajectory shape: {trajectory.shape}")

    row, col, fov = fov_name.split("/")
    pred_channel = f"{args.target_channel}_prediction"
    suffix = "_denoise" if args.save_mode == "denoise" else ""

    print(f"Writing {args.num_steps} step{suffix} stores to {args.output_dir}")
    for i in range(args.num_steps):
        step_path = args.output_dir / f"step{i + 1}{suffix}.zarr"
        # (C, Z, Y, X) → (T=1, C, Z, Y, X) to match GT TCZYX layout.
        step_arr = trajectory[i][np.newaxis]
        with open_ome_zarr(step_path, layout="hcs", mode="w", channel_names=[pred_channel]) as plate:
            position = plate.create_position(row, col, fov)
            position.create_image("0", step_arr)

    print(f"Done. step1{suffix}..step{args.num_steps}{suffix}.zarr at {fov_name}, channel {pred_channel!r}")


if __name__ == "__main__":
    main()
