"""Generate the frozen pixel-metrics parity fixture.

Run once (with GPU) to pin the pre-migration numerical baseline:

    uv run python applications/dynacell/tests/data/_generate_pixel_metrics_golden.py

The output ``pixel_metrics_golden.npz`` is committed alongside this script.
Re-run and re-commit only when intentionally changing metric semantics.

Source data
-----------
- GT  : /hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/SEC61B.zarr
        channel "Structure"
- Pred: /hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/sec61b_celldiff_iterative.zarr
        channel "Structure_prediction"
- FOV : 4/38452/5187  t=0  crop (Z=20, Y=256, X=256)
- Spacing: [0.290, 0.108, 0.108] µm  (aics-hipsc manifest)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import iohub.ngff as ngff
import numpy as np
import torch

_GT_ZARR = Path("/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/SEC61B.zarr")
_PRED_ZARR = Path("/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/sec61b_celldiff_iterative.zarr")
_GT_CHANNEL = "Structure"
_PRED_CHANNEL = "Structure_prediction"
_FOV = "4/38452/5187"
_T = 0
_CROP_Z = slice(10, 30)  # 20 slices from the middle (≥11 for Gaussian SSIM kernel)
_CROP_Y = slice(128, 384)  # 256 px
_CROP_X = slice(128, 384)  # 256 px
_SPACING = [0.290, 0.108, 0.108]

_OUT = Path(__file__).parent / "pixel_metrics_golden.npz"


def _load_fov_array(zarr_path: Path, channel: str, fov: str, t: int) -> np.ndarray:
    with ngff.open_ome_zarr(zarr_path, mode="r") as plate:
        row, col, fov_name = fov.split("/")
        position = plate[row][col][fov_name]
        ch_idx = position.channel_names.index(channel)
        arr = position["0"][t, ch_idx]  # (Z, Y, X)
        return np.asarray(arr)


def main() -> None:
    from dynacell.evaluation.metrics import compute_pixel_metrics

    print(f"Loading GT  : {_GT_ZARR}  [{_GT_CHANNEL}]  fov={_FOV} t={_T}")
    gt_full = _load_fov_array(_GT_ZARR, _GT_CHANNEL, _FOV, _T)
    print(f"Loading Pred: {_PRED_ZARR}  [{_PRED_CHANNEL}]  fov={_FOV} t={_T}")
    pred_full = _load_fov_array(_PRED_ZARR, _PRED_CHANNEL, _FOV, _T)

    gt = gt_full[_CROP_Z, _CROP_Y, _CROP_X].astype(np.float32)
    pred = pred_full[_CROP_Z, _CROP_Y, _CROP_X].astype(np.float32)
    print(f"Cropped shape: {pred.shape}  dtype={pred.dtype}")

    use_gpu = torch.cuda.is_available()
    print(f"Running compute_pixel_metrics  use_gpu={use_gpu}")
    pred_t = torch.as_tensor(pred)
    gt_t = torch.as_tensor(gt)
    metrics = compute_pixel_metrics(
        pred_t,
        gt_t,
        spacing=_SPACING,
        fsc_kwargs={},
        spectral_pcc_kwargs={},
        use_gpu=use_gpu,
    )
    print("Metrics:", {k: v for k, v in metrics.items() if not k.startswith("_")})

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    np.savez_compressed(
        _OUT,
        pred=pred,
        target=gt,
        expected_pcc=metrics["PCC"],
        expected_ssim=metrics["SSIM"],
        expected_nrmse=metrics["NRMSE"],
        expected_psnr=metrics["PSNR"],
        expected_spectral_pcc=metrics["Spectral_PCC"],
        expected_xy_fsc_resolution=metrics["XY_FSC_Resolution"],
        expected_z_fsc_resolution=metrics["Z_FSC_Resolution"],
        _fov_id=_FOV,
        _t=_T,
        _spacing=_SPACING,
        _pin_dynacell_commit=git_sha,
        _pin_torch_version=torch.__version__,
    )
    print(f"Saved: {_OUT}  ({_OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
