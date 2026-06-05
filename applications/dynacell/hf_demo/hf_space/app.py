"""DynaCell Virtual Staining Demo — Gradio Space.

Upload a zipped OME-Zarr HCS store once; then:
  Tab 1 — run CELL-Diff / FNet3D / VSCyto3D predictions on a selected timepoint,
           view a chosen Z slice, and see Spectral PCC metrics.
  Tab 2 — visualize the CELL-Diff ODE denoising trajectory as an animated GIF,
           with a Phase | Exp reference panel at the selected timepoint and Z slice.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from iohub.ngff import open_ome_zarr

sys.path.insert(0, str(Path(__file__).parent))
from predict_runner import ORGANELLE_LABELS, TARGET_CHANNELS, preprocess_zarr, run_prediction, run_trajectory

from cubic.metrics.bandlimited import spectral_pcc

ORGANELLES   = ["CAAX", "H2B", "SEC61B", "TOMM20"]
MODEL_KEYS   = ["celldiff", "fnet3d", "vscyto3d"]
MODEL_LABELS = {"celldiff": "CELL-Diff", "fnet3d": "FNet3D", "vscyto3d": "VSCyto3D"}
PHASE_CH = 0
FLUOR_CH = 2
_DEMO_REPO = "dihan-zheng/dynacell-demo-data"

SPACING = [0.174, 0.1494, 0.1494]
SPECTRAL_KWARGS = dict(bin_delta=1.0, tail_fraction=0.2, apodization="tukey", nbins_low=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_zarr_zip(zip_path: str) -> str:
    """Extract uploaded zip to a fresh temp dir; return the HCS zarr root path."""
    import json
    tmpdir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmpdir)
    for candidate in sorted(tmpdir.rglob(".zattrs")):
        root = candidate.parent
        try:
            zattrs = json.loads((root / ".zattrs").read_text())
            if "plate" in zattrs:
                return str(root)
        except Exception:
            pass
    for d in sorted(tmpdir.iterdir()):
        if d.is_dir():
            return str(d)
    raise ValueError("No zarr store found in zip.")


def get_data_shape(data_path: str) -> tuple[int, int]:
    """Return (n_timepoints, n_z_slices) from the first position in the plate."""
    with open_ome_zarr(data_path, mode="r") as plate:
        _, pos = next(plate.positions())
        return pos.data.shape[0], pos.data.shape[2]


def percentile_norm(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    p_lo, p_hi = np.percentile(img, [lo, hi])
    if p_hi == p_lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p_lo) / (p_hi - p_lo), 0, 1).astype(np.float32)


def compute_spectral_pcc(pred_zarr_path: str, gt_fluor_vol: np.ndarray) -> float | None:
    """Spectral PCC between the prediction (t=0) and the GT fluorescence volume."""
    try:
        with open_ome_zarr(pred_zarr_path, mode="r") as pred_plate:
            _, pred_pos = next(pred_plate.positions())
            pred_vol = np.array(pred_pos.data[0, 0], dtype=np.float32)
        return float(spectral_pcc(pred_vol, gt_fluor_vol, spacing=SPACING, **SPECTRAL_KWARGS))
    except Exception as e:
        print(f"spectral_pcc failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _make_slider_updates(data_path: str, organelle: str) -> tuple:
    """Read data shape and return slider updates + Phase|Exp figure."""
    n_tp, n_z = get_data_shape(data_path)
    z_mid = n_z // 2
    fig = render_phase_exp(data_path, 0, z_mid, organelle)
    return (
        gr.Slider(minimum=0, maximum=n_tp - 1, step=1, value=0),   # timepoint_slider
        gr.Slider(minimum=0, maximum=n_z - 1, step=1, value=z_mid), # z_slice_slider
        gr.Slider(minimum=0, maximum=n_tp - 1, step=1, value=0),   # traj_timepoint
        gr.Slider(minimum=0, maximum=n_z - 1, step=1, value=z_mid), # traj_z_slice
        fig,                                                          # traj_static
        n_tp, n_z,
    )


def load_demo_data(organelle: str, progress=gr.Progress()) -> tuple:
    """Download the demo zarr, extract it, and return updated UI state."""
    from huggingface_hub import hf_hub_download
    filename = f"{organelle}_mock.zarr.zip"
    progress(0.1, desc=f"Downloading {organelle} demo data...")
    zip_path = hf_hub_download(repo_id=_DEMO_REPO, filename=filename, repo_type="dataset")
    progress(0.8, desc="Extracting zarr...")
    data_path = extract_zarr_zip(zip_path)
    tp_sl, z_sl, traj_tp, traj_z, fig, n_tp, n_z = _make_slider_updates(data_path, organelle)
    progress(1.0, desc="Ready.")
    status = f"**Loaded:** {filename} (A549 cells, {n_tp} timepoints, {n_z} Z slices)"
    return data_path, status, tp_sl, z_sl, traj_tp, traj_z, fig


def on_upload(file, organelle: str) -> tuple:
    """Handle zarr zip upload: extract, read shape, update UI state."""
    if file is None:
        raise gr.Error("No file uploaded.")
    zip_path = file if isinstance(file, str) else file.name
    data_path = extract_zarr_zip(zip_path)
    tp_sl, z_sl, traj_tp, traj_z, fig, n_tp, n_z = _make_slider_updates(data_path, organelle)
    status = f"**Uploaded:** {Path(zip_path).name} ({n_tp} timepoints, {n_z} Z slices)"
    return data_path, status, tp_sl, z_sl, traj_tp, traj_z, fig


# ---------------------------------------------------------------------------
# Tab 2: Phase | Exp reference panel
# ---------------------------------------------------------------------------

def render_phase_exp(
    zarr_state: str | None,
    timepoint: int,
    z_slice: int,
    organelle: str,
) -> plt.Figure | None:
    """Render Phase and Experimental fluorescence side by side at (timepoint, z_slice)."""
    if zarr_state is None:
        return None
    with open_ome_zarr(zarr_state, mode="r") as plate:
        _, pos = next(plate.positions())
        n_tp = pos.data.shape[0]
        n_z  = pos.data.shape[2]
        tp = min(timepoint, n_tp - 1)
        z  = min(z_slice,   n_z  - 1)
        phase_img = np.array(pos.data[tp, PHASE_CH, z])
        fluor_img = np.array(pos.data[tp, FLUOR_CH, z])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    ax1.imshow(percentile_norm(phase_img), cmap="gray")
    ax1.set_title("Phase", fontsize=10)
    ax1.axis("off")
    ax2.imshow(percentile_norm(fluor_img), cmap="gray")
    ax2.set_title(f"Exp ({TARGET_CHANNELS[organelle]})", fontsize=10)
    ax2.axis("off")
    fig.suptitle(
        f"{ORGANELLE_LABELS[organelle]}  |  t={tp}  |  z={z}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Virtual Staining
# ---------------------------------------------------------------------------

def render_from_z(
    pred_info: dict | None,
    z_slice: int,
    zarr_state: str | None,
) -> plt.Figure | None:
    """Re-render the prediction comparison at a different Z slice."""
    if pred_info is None or zarr_state is None:
        return None

    organelle       = pred_info["organelle"]
    timepoint       = pred_info["timepoint"]
    selected_models = pred_info["selected_models"]
    pred_paths      = pred_info["paths"]
    pred_pccs       = pred_info["pccs"]
    n_z             = pred_info["n_z"]
    z = min(z_slice, n_z - 1)

    with open_ome_zarr(zarr_state, mode="r") as gt_plate:
        _, gt_pos = next(gt_plate.positions())
        phase_img = np.array(gt_pos.data[timepoint, PHASE_CH, z])
        fluor_img = np.array(gt_pos.data[timepoint, FLUOR_CH, z])

    cols = ["Phase", f"Exp ({TARGET_CHANNELS[organelle]})"] + [MODEL_LABELS[m] for m in selected_models]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.2))
    if len(cols) == 1:
        axes = [axes]

    axes[0].imshow(percentile_norm(phase_img), cmap="gray")
    axes[0].set_title("Phase", fontsize=10)
    axes[1].imshow(percentile_norm(fluor_img), cmap="gray")
    axes[1].set_title(f"Exp ({TARGET_CHANNELS[organelle]})", fontsize=10)

    for col_idx, model_key in enumerate(selected_models, start=2):
        label     = MODEL_LABELS[model_key]
        pred_path = pred_paths.get(model_key)
        pcc       = pred_pccs.get(model_key)
        if pred_path is not None:
            try:
                with open_ome_zarr(pred_path, mode="r") as pred_plate:
                    _, pred_pos = next(pred_plate.positions())
                    img = percentile_norm(np.array(pred_pos.data[0, 0, z]))
                title = f"{label}\nSpectral PCC={pcc:.3f}" if pcc is not None else label
            except Exception as e:
                img   = np.zeros_like(phase_img, dtype=np.float32)
                title = f"{label}\n(failed)"
                print(f"Render failed for {model_key}: {e}")
        else:
            img   = np.zeros_like(phase_img, dtype=np.float32)
            title = f"{label}\n(failed)"

        axes[col_idx].imshow(img, cmap="gray")
        axes[col_idx].set_title(title, fontsize=9)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{ORGANELLE_LABELS[organelle]}  |  t={timepoint}  |  z={z}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


def run_demo(
    zarr_zip,
    organelle: str,
    selected_models: list[str],
    timepoint: int,
    z_slice: int,
    zarr_state: str | None,
    progress=gr.Progress(),
) -> tuple[plt.Figure | None, list[list], str, dict]:
    if zarr_zip is None and not zarr_state:
        raise gr.Error("Please load demo data or upload a zarr zip file.")
    if not selected_models:
        raise gr.Error("Select at least one model.")

    if zarr_state:
        data_path = zarr_state
    else:
        progress(0.05, desc="Extracting zarr...")
        zip_path = zarr_zip if isinstance(zarr_zip, str) else zarr_zip.name
        data_path = extract_zarr_zip(zip_path)

    progress(0.10, desc="Computing normalization statistics...")
    preprocess_zarr(data_path)

    with open_ome_zarr(data_path, mode="r") as gt_plate:
        _, gt_pos = next(gt_plate.positions())
        n_tp, n_z = gt_pos.data.shape[0], gt_pos.data.shape[2]
        tp = min(timepoint, n_tp - 1)
        gt_fluor_vol = np.array(gt_pos.data[tp, FLUOR_CH], dtype=np.float32)

    pred_paths: dict[str, str | None] = {}
    pred_pccs:  dict[str, float | None] = {}
    n_models = len(selected_models)
    for i, model_key in enumerate(selected_models):
        progress(0.15 + 0.60 * i / n_models, desc=f"Running {MODEL_LABELS[model_key]}...")
        try:
            path = run_prediction(model_key, organelle, data_path, tp)
            pred_paths[model_key] = path
            pred_pccs[model_key]  = compute_spectral_pcc(path, gt_fluor_vol)
        except Exception as e:
            pred_paths[model_key] = None
            pred_pccs[model_key]  = None
            print(f"Prediction failed for {model_key}: {e}")

    pred_info = {
        "timepoint": tp, "organelle": organelle,
        "selected_models": selected_models,
        "paths": pred_paths, "pccs": pred_pccs, "n_z": n_z,
    }

    progress(0.80, desc="Rendering figure...")
    fig = render_from_z(pred_info, min(z_slice, n_z - 1), data_path)

    metrics_rows = [
        [MODEL_LABELS[m], "failed" if pred_paths.get(m) is None
         else (f"{pred_pccs[m]:.4f}" if pred_pccs.get(m) is not None else "N/A")]
        for m in selected_models
    ]

    progress(1.0, desc="Done.")
    return fig, metrics_rows, data_path, pred_info


# ---------------------------------------------------------------------------
# Tab 2: CellDiff Trajectory
# ---------------------------------------------------------------------------

def run_trajectory_demo(
    zarr_zip,
    organelle: str,
    timepoint: int,
    num_steps: int,
    z_slice: int,
    zarr_state: str | None,
    progress=gr.Progress(),
) -> tuple[str, str]:
    if zarr_zip is None and not zarr_state:
        raise gr.Error("Please load demo data or upload a zarr zip file.")

    if zarr_state:
        data_path = zarr_state
    else:
        progress(0.03, desc="Extracting zarr...")
        zip_path = zarr_zip if isinstance(zarr_zip, str) else zarr_zip.name
        data_path = extract_zarr_zip(zip_path)

    progress(0.08, desc="Computing normalization statistics...")
    preprocess_zarr(data_path)

    gif_path = run_trajectory(organelle, data_path, timepoint, num_steps, z_slice, progress)
    return gif_path, data_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="DynaCell Virtual Staining") as demo:
    gr.Markdown("## DynaCell Virtual Staining Demo")
    gr.Markdown(
        "**Tab 1** runs virtual staining predictions (CELL-Diff / FNet3D / VSCyto3D) "
        "on a phase-contrast OME-Zarr for a selected timepoint, and reports Spectral PCC. "
        "**Tab 2** visualizes the CELL-Diff ODE denoising trajectory."
    )

    zarr_state      = gr.State(value=None)
    pred_info_state = gr.State(value=None)

    # ---- Data source row -------------------------------------------------
    with gr.Row():
        organelle = gr.Dropdown(
            choices=[(ORGANELLE_LABELS[o], o) for o in ORGANELLES],
            value="CAAX", label="Organelle",
            info="Select the target organelle.",
        )
        load_demo_btn = gr.Button("Load Demo Data", variant="secondary", scale=1)
        zarr_upload = gr.File(
            label="Or upload your own zarr (.zip)",
            file_types=[".zip"],
            scale=2,
        )

    data_status = gr.Markdown("")

    # ---- Tabs ------------------------------------------------------------
    with gr.Tabs():

        with gr.Tab("Virtual Staining"):
            with gr.Row():
                model_selector = gr.CheckboxGroup(
                    choices=[(MODEL_LABELS[m], m) for m in MODEL_KEYS],
                    value=MODEL_KEYS,
                    label="Models to run",
                )
            with gr.Row():
                timepoint_slider = gr.Slider(
                    minimum=0, maximum=4, step=1, value=0,
                    label="Timepoint",
                    info="Range updates after loading data.",
                )
                z_slice_slider = gr.Slider(
                    minimum=0, maximum=99, step=1, value=15,
                    label="Z slice",
                    info="Range updates after loading data.",
                )
            run_btn = gr.Button("Run Predictions", variant="primary")
            output_plot = gr.Plot(label="Predictions")
            metrics_table = gr.Dataframe(
                headers=["Model", "Spectral PCC"],
                label="Spectral PCC (volumetric, vs experimental fluorescence)",
            )

        with gr.Tab("CELL-Diff Trajectory"):
            gr.Markdown(
                "Generate the CELL-Diff ODE denoising trajectory. "
                "T=0 is pure Gaussian noise; T=N is the final predicted fluorescence. "
                "The Phase | Exp panel updates live as you change the sliders."
            )
            with gr.Row():
                traj_timepoint = gr.Slider(
                    minimum=0, maximum=4, step=1, value=0,
                    label="Timepoint",
                    info="Range updates after loading data.",
                )
                traj_z_slice = gr.Slider(
                    minimum=0, maximum=99, step=1, value=15,
                    label="Z slice",
                    info="Range updates after loading data.",
                )
                traj_num_steps = gr.Slider(
                    minimum=10, maximum=100, step=10, value=50,
                    label="ODE steps",
                )
            traj_static = gr.Plot(label="Phase | Exp (reference)")
            traj_btn    = gr.Button("Generate Trajectory", variant="primary")
            traj_gif    = gr.Image(label="Animated trajectory (GIF)", type="filepath")

    # ---- Event wiring ----------------------------------------------------

    _data_outputs = [
        zarr_state, data_status,
        timepoint_slider, z_slice_slider,
        traj_timepoint, traj_z_slice,
        traj_static,
    ]

    load_demo_btn.click(
        fn=load_demo_data,
        inputs=[organelle],
        outputs=_data_outputs,
    )

    zarr_upload.upload(
        fn=on_upload,
        inputs=[zarr_upload, organelle],
        outputs=_data_outputs,
    )

    run_btn.click(
        fn=run_demo,
        inputs=[zarr_upload, organelle, model_selector, timepoint_slider, z_slice_slider, zarr_state],
        outputs=[output_plot, metrics_table, zarr_state, pred_info_state],
    )

    z_slice_slider.change(
        fn=render_from_z,
        inputs=[pred_info_state, z_slice_slider, zarr_state],
        outputs=[output_plot],
    )

    # Phase | Exp panel updates on any slider or organelle change
    for _trigger in (traj_timepoint, traj_z_slice, organelle):
        _trigger.change(
            fn=render_phase_exp,
            inputs=[zarr_state, traj_timepoint, traj_z_slice, organelle],
            outputs=[traj_static],
        )

    traj_btn.click(
        fn=run_trajectory_demo,
        inputs=[zarr_upload, organelle, traj_timepoint, traj_num_steps, traj_z_slice, zarr_state],
        outputs=[traj_gif, zarr_state],
    )

if __name__ == "__main__":
    demo.launch()
