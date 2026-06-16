"""DynaCell Virtual Staining Demo — Gradio Space.

Single-page layout, three stacked sections (controls left, render right):

  1. Data        — pick a demo organelle dataset; browse the
                   Phase | Experimental-fluorescence view by timepoint and Z.
  2. Regression  — deterministic models (FNet3D, VSCyto3D): predict + Spectral PCC.
  3. Generative  — CELL-Diff: ODE trajectory (Phase | Exp | prediction) with an
                   ODE-step slider and the per-step Spectral PCC.

Each section has its own Timepoint and Z-slice sliders. Inference runs on the
single selected timepoint only.
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
from cubic.metrics.bandlimited import spectral_pcc  # noqa: E402
from predict_runner import (  # noqa: E402
    ORGANELLE_LABELS,
    SPACING,
    SPECTRAL_KWARGS,
    TARGET_CHANNELS,
    compute_trajectory,
    preprocess_zarr,
    run_prediction,
)

ORGANELLES = ["CAAX", "H2B", "SEC61B", "TOMM20"]
REGRESSION_MODELS = ["fnet3d", "vscyto3d"]
MODEL_LABELS = {"celldiff": "CELL-Diff", "fnet3d": "FNet3D", "vscyto3d": "VSCyto3D"}
PHASE_CH = 0
FLUOR_CH = 2
_DEMO_REPO = "biohub/dynacell-demo-data"

PATCH_D = 8  # Z window CELL-Diff operates on (center of the stack)
PANEL_IN = 2.2  # per-panel width (inches) for data + regression → equal image heights
FIG_H = 2.8  # figure height (inches) for data + regression
# Generative panels are larger: its controls column is taller (extra ODE-step slider).
GEN_PANEL_IN = 3.0
GEN_FIG_H = 3.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_dark(request: gr.Request | None) -> bool:
    """Detect the client theme from the `__theme` query param (default: dark)."""
    try:
        return (request.query_params.get("__theme") or "dark").lower() != "light"
    except Exception:
        return True


def style_fig(fig, dark: bool) -> None:
    """Match the figure to the themed widget background: transparent panel, themed text."""
    fg = "white" if dark else "black"
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
        ax.title.set_color(fg)
    if fig._suptitle is not None:
        fig._suptitle.set_color(fg)


def extract_zarr_zip(zip_path: str) -> str:
    """Extract the demo zip to a fresh temp dir; return the HCS zarr root path."""
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
# Renderers (all transparent + themed; constant per-panel size → equal heights)
# ---------------------------------------------------------------------------


def render_phase_exp(
    zarr_state: str | None,
    timepoint: int,
    z_slice: int,
    organelle: str,
    dark: bool = True,
) -> plt.Figure | None:
    """Render Phase and Experimental fluorescence side by side at (timepoint, z_slice)."""
    if zarr_state is None:
        return None
    with open_ome_zarr(zarr_state, mode="r") as plate:
        _, pos = next(plate.positions())
        n_tp = pos.data.shape[0]
        n_z = pos.data.shape[2]
        tp = min(int(timepoint), n_tp - 1)
        z = min(int(z_slice), n_z - 1)
        phase_img = np.array(pos.data[tp, PHASE_CH, z])
        fluor_img = np.array(pos.data[tp, FLUOR_CH, z])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PANEL_IN * 2, FIG_H), layout="constrained")
    ax1.imshow(percentile_norm(phase_img), cmap="gray")
    ax1.set_title("Phase", fontsize=10)
    ax1.axis("off")
    ax2.imshow(percentile_norm(fluor_img), cmap="gray")
    ax2.set_title(f"Exp ({TARGET_CHANNELS[organelle]})", fontsize=10)
    ax2.axis("off")
    fig.suptitle(f"{ORGANELLE_LABELS[organelle]}  |  t={tp}  |  z={z}", fontsize=11)
    style_fig(fig, dark)
    return fig


def render_predictions(
    pred_info: dict | None,
    z_slice: int,
    zarr_state: str | None,
    dark: bool = True,
) -> plt.Figure | None:
    """Render Phase | Exp | <models> at the given Z slice (regression section)."""
    if pred_info is None or zarr_state is None:
        return None

    organelle = pred_info["organelle"]
    timepoint = pred_info["timepoint"]
    selected_models = pred_info["selected_models"]
    pred_paths = pred_info["paths"]
    pred_pccs = pred_info["pccs"]
    n_z = pred_info["n_z"]
    z = min(int(z_slice), n_z - 1)

    with open_ome_zarr(zarr_state, mode="r") as gt_plate:
        _, gt_pos = next(gt_plate.positions())
        phase_img = np.array(gt_pos.data[timepoint, PHASE_CH, z])
        fluor_img = np.array(gt_pos.data[timepoint, FLUOR_CH, z])

    cols = ["Phase", f"Exp ({TARGET_CHANNELS[organelle]})"] + [MODEL_LABELS[m] for m in selected_models]
    fig, axes = plt.subplots(1, len(cols), figsize=(PANEL_IN * len(cols), FIG_H), layout="constrained")
    if len(cols) == 1:
        axes = [axes]

    axes[0].imshow(percentile_norm(phase_img), cmap="gray")
    axes[0].set_title("Phase", fontsize=10)
    axes[1].imshow(percentile_norm(fluor_img), cmap="gray")
    axes[1].set_title(f"Exp ({TARGET_CHANNELS[organelle]})", fontsize=10)

    for col_idx, model_key in enumerate(selected_models, start=2):
        label = MODEL_LABELS[model_key]
        pred_path = pred_paths.get(model_key)
        pcc = pred_pccs.get(model_key)
        if pred_path is not None:
            try:
                with open_ome_zarr(pred_path, mode="r") as pred_plate:
                    _, pred_pos = next(pred_plate.positions())
                    img = percentile_norm(np.array(pred_pos.data[0, 0, z]))
                title = f"{label}\nSpectral PCC={pcc:.3f}" if pcc is not None else label
            except Exception as e:
                img = np.zeros_like(phase_img, dtype=np.float32)
                title = f"{label}\n(failed)"
                print(f"Render failed for {model_key}: {e}")
        else:
            img = np.zeros_like(phase_img, dtype=np.float32)
            title = f"{label}\n(failed)"

        axes[col_idx].imshow(img, cmap="gray")
        axes[col_idx].set_title(title, fontsize=9)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{ORGANELLE_LABELS[organelle]}  |  t={timepoint}  |  z={z}", fontsize=11)
    style_fig(fig, dark)
    return fig


def render_trajectory_frame(
    traj_info: dict | None,
    z_abs: int,
    step: int,
    dark: bool = True,
) -> plt.Figure | None:
    """Render Phase | Exp | CELL-Diff prediction at one ODE step + that step's Spectral PCC."""
    if traj_info is None:
        return None
    with np.load(traj_info["traj_path"]) as data:
        traj = data["traj"]  # (num_steps, 1, D, H, W)
        phase = data["phase"]  # (D, H, W)
        gt = data["gt"]  # (D, H, W)
    z_start = traj_info["z_start"]
    patch_d = traj_info["patch_d"]
    organelle = traj_info["organelle"]
    timepoint = traj_info["timepoint"]
    num_steps = traj_info["num_steps"]

    z_patch = max(0, min(int(z_abs) - z_start, patch_d - 1))
    step = max(0, min(int(step), num_steps - 1))
    pcc = float(spectral_pcc(traj[step, 0], gt, spacing=SPACING, **SPECTRAL_KWARGS))

    fig, (ax_p, ax_e, ax_t) = plt.subplots(1, 3, figsize=(GEN_PANEL_IN * 3, GEN_FIG_H), layout="constrained")
    for ax in (ax_p, ax_e, ax_t):
        ax.axis("off")
    ax_p.imshow(percentile_norm(phase[z_patch]), cmap="gray")
    ax_p.set_title("Phase", fontsize=10)
    ax_e.imshow(percentile_norm(gt[z_patch]), cmap="gray")
    ax_e.set_title(f"Exp ({TARGET_CHANNELS[organelle]})", fontsize=10)
    ax_t.imshow(percentile_norm(traj[step, 0, z_patch]), cmap="gray")
    ax_t.set_title(f"CELL-Diff\nStep {step}/{num_steps - 1} · PCC {pcc:.3f}", fontsize=9)
    fig.suptitle(f"{ORGANELLE_LABELS[organelle]}  |  t={timepoint}  |  z={z_start + z_patch}", fontsize=11)
    style_fig(fig, dark)
    return fig


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------


def load_demo_data(organelle: str, progress=gr.Progress(), request: gr.Request | None = None) -> tuple:
    """Download + extract the demo zarr; set every section's slider ranges; render view."""
    from huggingface_hub import hf_hub_download

    filename = f"{organelle}_mock.zarr.zip"
    progress(0.1, desc=f"Downloading {organelle} demo data...")
    zip_path = hf_hub_download(repo_id=_DEMO_REPO, filename=filename, repo_type="dataset")
    progress(0.7, desc="Extracting zarr...")
    data_path = extract_zarr_zip(zip_path)
    n_tp, n_z = get_data_shape(data_path)
    z_mid = n_z // 2
    z_start = (n_z - PATCH_D) // 2
    fig = render_phase_exp(data_path, 0, z_mid, organelle, is_dark(request))
    status = f"**Loaded:** {filename}  ·  {n_tp} timepoints · {n_z} Z slices"
    progress(1.0, desc="Ready.")

    def t_slider():
        return gr.Slider(minimum=0, maximum=n_tp - 1, step=1, value=0)

    return (
        data_path,  # zarr_state
        status,  # data_status
        t_slider(),  # data_t
        gr.Slider(minimum=0, maximum=n_z - 1, step=1, value=z_mid),  # data_z
        t_slider(),  # reg_t
        gr.Slider(minimum=0, maximum=n_z - 1, step=1, value=z_mid),  # reg_z
        t_slider(),  # gen_t
        gr.Slider(  # gen_z (center patch)
            minimum=z_start, maximum=z_start + PATCH_D - 1, step=1, value=z_start + PATCH_D // 2
        ),
        fig,  # data_view
    )


def on_data_slider(
    zarr_state: str | None,
    organelle: str,
    timepoint: int,
    z_slice: int,
    request: gr.Request | None = None,
) -> plt.Figure | None:
    """Re-render the data view on T/Z change."""
    if not zarr_state:
        return None
    return render_phase_exp(zarr_state, timepoint, z_slice, organelle, is_dark(request))


# ---------------------------------------------------------------------------
# 2. Regression models
# ---------------------------------------------------------------------------


def run_regression(
    zarr_state: str | None,
    organelle: str,
    selected_models: list[str],
    timepoint: int,
    z_slice: int,
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> tuple[plt.Figure | None, dict]:
    if not zarr_state:
        raise gr.Error("Load demo data first.")
    if not selected_models:
        raise gr.Error("Select at least one regression model.")

    progress(0.05, desc="Computing normalization statistics...")
    preprocess_zarr(zarr_state)

    with open_ome_zarr(zarr_state, mode="r") as gt_plate:
        _, gt_pos = next(gt_plate.positions())
        n_tp, n_z = gt_pos.data.shape[0], gt_pos.data.shape[2]
        tp = min(int(timepoint), n_tp - 1)  # single timepoint only
        gt_fluor_vol = np.array(gt_pos.data[tp, FLUOR_CH], dtype=np.float32)

    pred_paths: dict[str, str | None] = {}
    pred_pccs: dict[str, float | None] = {}
    n = len(selected_models)
    for i, model_key in enumerate(selected_models):
        progress(0.15 + 0.7 * i / n, desc=f"Running {MODEL_LABELS[model_key]}...")
        try:
            path = run_prediction(model_key, organelle, zarr_state, tp)
            pred_paths[model_key] = path
            pred_pccs[model_key] = compute_spectral_pcc(path, gt_fluor_vol)
        except Exception as e:
            pred_paths[model_key] = None
            pred_pccs[model_key] = None
            print(f"Prediction failed for {model_key}: {e}")

    pred_info = {
        "timepoint": tp,
        "organelle": organelle,
        "selected_models": list(selected_models),
        "paths": pred_paths,
        "pccs": pred_pccs,
        "n_z": n_z,
    }
    progress(0.9, desc="Rendering...")
    fig = render_predictions(pred_info, min(int(z_slice), n_z - 1), zarr_state, is_dark(request))
    progress(1.0, desc="Done.")
    return fig, pred_info


def rerender_regression(
    pred_info: dict | None,
    z_slice: int,
    zarr_state: str | None,
    request: gr.Request | None = None,
) -> plt.Figure | None:
    """Re-render regression predictions at a new Z slice (no re-prediction)."""
    if pred_info is None or not zarr_state:
        return None
    return render_predictions(pred_info, int(z_slice), zarr_state, is_dark(request))


# ---------------------------------------------------------------------------
# 3. Generative model — CELL-Diff
# ---------------------------------------------------------------------------


def run_generative(
    zarr_state: str | None,
    organelle: str,
    timepoint: int,
    num_steps: int,
    z_slice: int,
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> tuple:
    if not zarr_state:
        raise gr.Error("Load demo data first.")
    progress(0.05, desc="Computing normalization statistics...")
    preprocess_zarr(zarr_state)
    # Runs the ODE on the single selected timepoint only.
    traj_info = compute_trajectory(organelle, zarr_state, int(timepoint), int(num_steps), progress)
    n = int(num_steps)
    last = n - 1
    progress(0.95, desc="Rendering final step...")
    fig = render_trajectory_frame(traj_info, int(z_slice), last, is_dark(request))
    step_slider = gr.Slider(minimum=0, maximum=last, step=1, value=last, label="ODE step")
    progress(1.0, desc="Done.")
    return fig, traj_info, step_slider


def rerender_generative(
    traj_info: dict | None,
    z_slice: int,
    step: int,
    request: gr.Request | None = None,
) -> plt.Figure | None:
    """Re-render the trajectory frame at a new ODE step / Z slice (no ODE re-run)."""
    if traj_info is None:
        return None
    return render_trajectory_frame(traj_info, int(z_slice), int(step), is_dark(request))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="DynaCell Virtual Staining") as demo:
    gr.Markdown("## DynaCell Virtual Staining Demo")
    gr.Markdown(
        "Predict fluorescence from label-free phase-contrast 3-D microscopy of live "
        "A549 cells. Browse the data, run deterministic **regression** models, and "
        "explore the **generative** CELL-Diff ODE trajectory."
    )

    zarr_state = gr.State(value=None)
    reg_pred_state = gr.State(value=None)
    traj_state = gr.State(value=None)

    # ---- 1. Data ---------------------------------------------------------
    gr.Markdown("### 1.&nbsp; Data")
    with gr.Row():
        with gr.Column(scale=1):
            organelle = gr.Dropdown(
                choices=[(ORGANELLE_LABELS[o], o) for o in ORGANELLES],
                value="CAAX",
                label="Organelle",
                info="Select the target organelle.",
            )
            load_demo_btn = gr.Button("Load Demo Data", variant="primary")
            data_t = gr.Slider(0, 4, step=1, value=0, label="Timepoint")
            data_z = gr.Slider(0, 15, step=1, value=8, label="Z slice")
            data_status = gr.Markdown("")
        with gr.Column(scale=2):
            data_view = gr.Plot(label="Phase | Experimental fluorescence")

    # ---- 2. Regression models -------------------------------------------
    gr.Markdown("---")
    gr.Markdown("### 2.&nbsp; Regression models: FNet3D, VSCyto3D")
    with gr.Row():
        with gr.Column(scale=1):
            reg_models = gr.CheckboxGroup(
                choices=[("FNet3D", "fnet3d"), ("VSCyto3D", "vscyto3d")],
                value=REGRESSION_MODELS,
                label="Models",
            )
            reg_t = gr.Slider(0, 4, step=1, value=0, label="Timepoint")
            reg_z = gr.Slider(0, 15, step=1, value=8, label="Z slice")
            reg_run_btn = gr.Button("Run regression", variant="primary")
        with gr.Column(scale=2):
            reg_plot = gr.Plot(label="Predictions")

    # ---- 3. Generative model: CELL-Diff ---------------------------------
    gr.Markdown("---")
    gr.Markdown("### 3.&nbsp; Generative model: CELL-Diff")
    with gr.Row():
        with gr.Column(scale=1):
            gen_steps = gr.Slider(10, 100, step=10, value=50, label="ODE steps")
            gen_t = gr.Slider(0, 4, step=1, value=0, label="Timepoint")
            gen_z = gr.Slider(4, 11, step=1, value=8, label="Z slice")
            gr.Markdown("_CELL-Diff inference requires 8 input slices._")
            gen_btn = gr.Button("Generate", variant="primary")
            gen_step = gr.Slider(0, 1, step=1, value=0, label="ODE step", info="Slide after generating.")
        with gr.Column(scale=2):
            gen_plot = gr.Plot(label="Phase | Exp | Trajectory")

    # ---- Wiring ----------------------------------------------------------
    load_demo_btn.click(
        load_demo_data,
        [organelle],
        [zarr_state, data_status, data_t, data_z, reg_t, reg_z, gen_t, gen_z, data_view],
    )
    for _trigger in (data_t, data_z):
        _trigger.change(on_data_slider, [zarr_state, organelle, data_t, data_z], [data_view])

    reg_run_btn.click(
        run_regression,
        [zarr_state, organelle, reg_models, reg_t, reg_z],
        [reg_plot, reg_pred_state],
    )
    reg_z.change(rerender_regression, [reg_pred_state, reg_z, zarr_state], [reg_plot])

    gen_btn.click(
        run_generative,
        [zarr_state, organelle, gen_t, gen_steps, gen_z],
        [gen_plot, traj_state, gen_step],
    )
    for _trigger in (gen_step, gen_z):
        _trigger.change(rerender_generative, [traj_state, gen_z, gen_step], [gen_plot])


if __name__ == "__main__":
    demo.launch()
