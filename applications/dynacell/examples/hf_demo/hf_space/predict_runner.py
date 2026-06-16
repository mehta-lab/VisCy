"""Download checkpoints from HF Hub, generate configs, run dynacell predict, and generate trajectories."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import spaces
import zarr
from huggingface_hub import hf_hub_download

CHECKPOINT_REPO = "biohub/dynacell-checkpoints"
TEMPLATE_DIR = Path(__file__).parent / "config_templates"

# (model, organelle) → filename in the HF checkpoint repo
CHECKPOINT_FILES: dict[tuple[str, str], str] = {
    ("celldiff", "CAAX"): "celldiff_caax.ckpt",
    ("celldiff", "H2B"): "celldiff_h2b.ckpt",
    ("celldiff", "SEC61B"): "celldiff_sec61b.ckpt",
    ("celldiff", "TOMM20"): "celldiff_tomm20.ckpt",
    ("fnet3d", "CAAX"): "fnet3d_caax.ckpt",
    ("fnet3d", "H2B"): "fnet3d_h2b.ckpt",
    ("fnet3d", "SEC61B"): "fnet3d_sec61b.ckpt",
    ("fnet3d", "TOMM20"): "fnet3d_tomm20.ckpt",
    ("vscyto3d", "CAAX"): "vscyto3d_caax.ckpt",
    ("vscyto3d", "H2B"): "vscyto3d_h2b.ckpt",
    ("vscyto3d", "SEC61B"): "vscyto3d_sec61b.ckpt",
    ("vscyto3d", "TOMM20"): "vscyto3d_tomm20.ckpt",
}

TARGET_CHANNELS: dict[str, str] = {
    "CAAX": "Membrane",
    "H2B": "Nuclei",
    "SEC61B": "Structure",
    "TOMM20": "Structure",
}

ORGANELLE_LABELS: dict[str, str] = {
    "CAAX": "Membrane (CAAX)",
    "H2B": "Chromatin (H2B)",
    "SEC61B": "ER (SEC61B)",
    "TOMM20": "Mitochondria (TOMM20)",
}

FLUOR_CH = 2  # channel index for fluorescence in the input zarr

# Spectral PCC settings (volumetric; shared with app.py).
SPACING = [0.174, 0.1494, 0.1494]
SPECTRAL_KWARGS = dict(bin_delta=1.0, tail_fraction=0.2, apodization="tukey", nbins_low=3)

# Cache downloaded checkpoints in /tmp so the Space doesn't re-download each run
_ckpt_cache: dict[str, str] = {}


def get_checkpoint(model: str, organelle: str) -> str:
    """Download (or return cached) checkpoint path for a given model + organelle."""
    key = (model, organelle)
    filename = CHECKPOINT_FILES[key]
    if filename not in _ckpt_cache:
        print(f"Downloading {filename} from {CHECKPOINT_REPO} ...")
        local = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=filename)
        _ckpt_cache[filename] = local
    return _ckpt_cache[filename]


def preprocess_zarr(data_path: str) -> None:
    """Compute normalization statistics for the uploaded zarr via viscy preprocess."""
    subprocess.run(
        ["viscy", "preprocess", f"--data_path={data_path}", "--num_workers=1", "--block_size=32"],
        check=True,
    )


def create_single_timepoint_zarr(source_path: str, timepoint: int) -> str:
    """Copy source HCS zarr plate, keeping only the selected timepoint.

    Remaps timepoint_statistics in .zattrs so index "0" carries the selected
    timepoint's normalization stats (needed by celldiff's MinMaxSampled).
    """
    out_path = Path(tempfile.gettempdir()) / f"dynacell_t{timepoint}_{uuid.uuid4().hex[:8]}.zarr"
    shutil.copytree(source_path, str(out_path))

    src_store = zarr.open(source_path, mode="r")
    dst_store = zarr.open(str(out_path), mode="r+")

    def _trim(src_grp: zarr.Group, dst_grp: zarr.Group) -> None:
        for key in list(src_grp.keys()):
            item = src_grp[key]
            if isinstance(item, zarr.Array) and key == "0":
                # Write selected timepoint into index 0, then resize to T=1
                dst_arr = dst_grp[key]
                dst_arr[0] = item[timepoint]
                dst_arr.resize((1,) + item.shape[1:])
            elif isinstance(item, zarr.Group):
                _trim(item, dst_grp[key])

    _trim(src_store, dst_store)

    # Remap timepoint_statistics["<timepoint>"] → ["0"] in each FOV's .zattrs
    def _remap_tp_stats(zattrs_path: Path) -> None:
        if not zattrs_path.exists():
            return
        zattrs = json.loads(zattrs_path.read_text())
        norm = zattrs.get("normalization", {})
        changed = False
        for ch_data in norm.values():
            if "timepoint_statistics" in ch_data:
                tp_stats = ch_data["timepoint_statistics"]
                t_key = str(timepoint)
                if t_key in tp_stats:
                    ch_data["timepoint_statistics"] = {"0": tp_stats[t_key]}
                    changed = True
        if changed:
            zattrs_path.write_text(json.dumps(zattrs))

    for row in out_path.iterdir():
        if not row.is_dir():
            continue
        for col in row.iterdir():
            if not col.is_dir():
                continue
            for fov in col.iterdir():
                if fov.is_dir():
                    _remap_tp_stats(fov / ".zattrs")

    return str(out_path)


@spaces.GPU(duration=120)
def run_prediction(model: str, organelle: str, data_path: str, timepoint: int) -> str:
    """Run prediction for a single timepoint; return the output zarr path.

    Creates a single-timepoint subset of the source zarr, runs prediction on it,
    and returns the path to the output zarr (which has T=1). The `dynacell predict`
    subprocess inherits the ZeroGPU allocation from this decorated frame.
    """
    subset_path = create_single_timepoint_zarr(data_path, timepoint)

    ckpt_path = get_checkpoint(model, organelle)
    output_dir = Path(tempfile.gettempdir()) / f"dynacell_pred_{uuid.uuid4().hex[:8]}"
    output_store = str(output_dir / f"{organelle}_{model}.zarr")

    template = (TEMPLATE_DIR / f"{model}.yaml").read_text()
    config_text = template.format(
        ckpt_path=ckpt_path,
        data_path=subset_path,
        output_store=output_store,
        target_channel=TARGET_CHANNELS[organelle],
    )

    config_path = Path(tempfile.gettempdir()) / f"dynacell_cfg_{uuid.uuid4().hex[:8]}.yaml"
    config_path.write_text(config_text)

    print(f"Running dynacell predict: {model} / {organelle} / t={timepoint}")
    subprocess.run(["dynacell", "predict", "-c", str(config_path)], check=True)
    config_path.unlink(missing_ok=True)

    return output_store


@spaces.GPU(duration=120)
def compute_trajectory(
    organelle: str,
    data_path: str,
    timepoint: int = 0,
    num_steps: int = 50,
    progress=None,
) -> dict:
    """Run the CELL-Diff ODE; save trajectory to /tmp as .npy; return metadata dict.

    The returned dict contains everything needed to call render_trajectory_frame
    without re-running the ODE.
    """
    import numpy as np
    import torch
    from iohub.ngff import open_ome_zarr

    from dynacell.engine import DynacellFlowMatching
    from viscy_data._utils import _read_norm_meta

    if progress is not None:
        progress(0.05, desc="Downloading CELL-Diff checkpoint...")
    ckpt_path = get_checkpoint("celldiff", organelle)

    if progress is not None:
        progress(0.15, desc="Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DynacellFlowMatching.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    patch_d, patch_h, patch_w = model.model.net.input_spatial_size  # (8, 512, 512)

    if progress is not None:
        progress(0.25, desc="Reading phase data...")
    with open_ome_zarr(data_path, mode="r") as plate:
        _, pos = next(plate.positions())
        phase_ch = pos.get_channel_index("Phase3D")
        phase_raw = np.array(pos.data[timepoint, phase_ch])
        fluor_raw = np.array(pos.data[timepoint, FLUOR_CH])
        norm_meta = _read_norm_meta(pos)

    tp_stats = norm_meta["Phase3D"]["timepoint_statistics"][str(timepoint)]
    lo = tp_stats["p1"].item()
    hi = tp_stats["p99"].item()
    phase_norm = np.clip(phase_raw.astype(np.float32), lo, hi)
    phase_norm = 2.0 * (phase_norm - lo) / (hi - lo + 1e-8) - 1.0

    z_total = phase_norm.shape[0]
    z_start = (z_total - patch_d) // 2
    phase_crop = phase_norm[z_start : z_start + patch_d, :patch_h, :patch_w]
    # Raw phase + experimental fluorescence over the same window, for the
    # display panels and the per-step Spectral PCC.
    phase_disp = phase_raw[z_start : z_start + patch_d, :patch_h, :patch_w].astype(np.float32)
    gt_crop = fluor_raw[z_start : z_start + patch_d, :patch_h, :patch_w].astype(np.float32)

    if progress is not None:
        progress(0.35, desc=f"Generating {num_steps}-step ODE trajectory...")
    phase_tensor = torch.from_numpy(phase_crop).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        trajectory = model.model.generate_trajectory(phase_tensor, num_steps=num_steps)
    traj_np = trajectory[:, 0].cpu().numpy().astype(np.float32)  # (num_steps, 1, D, H, W)

    if progress is not None:
        progress(0.90, desc="Saving trajectory to disk...")
    traj_path = str(Path(tempfile.gettempdir()) / f"traj_{uuid.uuid4().hex[:8]}.npz")
    np.savez(traj_path, traj=traj_np, phase=phase_disp, gt=gt_crop)

    if progress is not None:
        progress(1.0, desc="Done.")

    return {
        "traj_path": traj_path,
        "z_start": z_start,
        "patch_d": patch_d,
        "organelle": organelle,
        "timepoint": timepoint,
        "num_steps": num_steps,
    }
