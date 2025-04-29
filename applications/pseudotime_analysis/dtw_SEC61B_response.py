# %%
import ast
import os
import warnings
from glob import glob
from pathlib import Path
from typing import NamedTuple

import ffmpeg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from iohub import open_ome_zarr
from skimage.exposure import adjust_gamma, rescale_intensity

from utils import find_top_matching_tracks
from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_pca

plt.style.use("../contrastive_phenotyping/figures/figure.mplstyle")

# %%

# NOTE: swap this depending on the condition
ALIGNING_CONDITION = "organelle_only"  # "infection", "cell_division", "matching_inf_div", "infection_organelle","organelle_only"
EMBEDDING_CONDITION = "phase_n_organelle"  # "phase_n_organelle", "phase_n_infection", "phase_n_cell_division"

input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr"
)

features_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/figure/SEC61B"
)

if EMBEDDING_CONDITION == "organelle_only":
    # features only organelle
    feature_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_organelle_only_ntxent_192patch_ckpt52_rev8_GT.zarr"
elif EMBEDDING_CONDITION == "phase_n_organelle":
    # features only infection
    feature_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
else:
    raise ValueError(f"Embedding condition {EMBEDDING_CONDITION} not implemented")
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)

embeddings_dataset = read_embedding_dataset(feature_path)
feature_df = embeddings_dataset["sample"].to_dataframe().reset_index(drop=True)

cell_division_matching_lineages_path = (
    features_root / "20241107_SEC61B_cell_division_matching_lineages.csv"
)
infection_matching_lineages_path = (
    features_root / "20241107_SEC61B_infection_matching_lineages.csv"
)
organelle_only_matching_lineages_path = (
    features_root / "20241107_SEC61B_organelle_only_matching_lineages.csv"
)

cell_division_df = pd.read_csv(cell_division_matching_lineages_path)
infection_df = pd.read_csv(infection_matching_lineages_path)
organelle_only_df = pd.read_csv(organelle_only_matching_lineages_path)

# %%
# Filtered tracks that show both events
matching_tracks = find_top_matching_tracks(cell_division_df, infection_df, n_top=10)

# %%
# Cache the top 5 lineages

z_range = (15, 30)
yx_patch_size = (192, 192)

if ALIGNING_CONDITION == "infection":
    condition_df = infection_df
    channels_to_display = [
        "Phase3D",
        # "raw GFP EX488 EM525-45",
        "raw mCherry EX561 EM600-37",
    ]
elif ALIGNING_CONDITION == "cell_division":
    condition_df = cell_division_df
    channels_to_display = [
        "Phase3D",
        "raw GFP EX488 EM525-45",
        # "raw mCherry EX561 EM600-37",
    ]
elif ALIGNING_CONDITION == "matching_inf_div":
    condition_df = matching_tracks
    channels_to_display = [
        "Phase3D",
        "raw GFP EX488 EM525-45",
        "raw mCherry EX561 EM600-37",
    ]
elif ALIGNING_CONDITION == "infection_organelle":
    condition_df = infection_df
    channels_to_display = [
        "Phase3D",
        "raw GFP EX488 EM525-45",
        # "raw mCherry EX561 EM600-37",
    ]
elif ALIGNING_CONDITION == "organelle_only":
    condition_df = organelle_only_df
    channels_to_display = ["raw GFP EX488 EM525-45"]


output_data_dir = Path(
    features_root
    / f"SEC61B/aligned_response_{EMBEDDING_CONDITION}_by_{ALIGNING_CONDITION}"
)
output_data_dir.mkdir(parents=True, exist_ok=True)

output_data_unaligned_dir = Path(
    features_root
    / f"SEC61B/unaligned_response_{EMBEDDING_CONDITION}_by_{ALIGNING_CONDITION}"
)
output_data_unaligned_dir.mkdir(parents=True, exist_ok=True)
# %%
for i, row in condition_df.head(6).iterrows():
    fov_name = row.fov_name
    track_ids = ast.literal_eval(row.track_ids)

    data_module = TripletDataModule(
        data_path=input_data_path,
        tracks_path=tracks_path,
        include_fov_names=[fov_name] * len(track_ids),
        include_track_ids=track_ids,
        source_channel=channels_to_display,
        z_range=z_range,
        initial_yx_patch_size=yx_patch_size,
        final_yx_patch_size=yx_patch_size,
        batch_size=1,
        num_workers=16,
        normalizations=None,
        predict_cells=True,
    )
    data_module.setup("predict")

    img_stack = []

    for batch in data_module.predict_dataloader():
        images = batch["anchor"].numpy()[0]
        indices = batch["index"]
        t_idx = indices["t"].tolist()
        # Take the middle z-slice
        z_idx = images.shape[1] // 2
        C, Z, Y, X = images.shape

        image_out = np.zeros((C, 1, Y, X), dtype=np.float32)
        for c_idx, channel in enumerate(channels_to_display):
            if channel in ["Phase3D", "DIC", "BF"]:
                image_out[c_idx] = images[c_idx, z_idx]
                # Normalize 0-mean, unit variance
                image_out[c_idx] = (
                    image_out[c_idx] - image_out[c_idx].mean()
                ) / image_out[c_idx].std()
                # rescale intensity
                image_out[c_idx] = rescale_intensity(image_out[c_idx], out_range=(0, 1))
            else:
                # For fluoresnce max projection and then normalize
                image_out[c_idx] = np.max(images[c_idx], axis=0)
                # Percentile normalization
                lower, upper = np.percentile(image_out[c_idx], (50, 99))
                image_out[c_idx] = (image_out[c_idx] - lower) / (upper - lower)
                # Rescale intensity
                image_out[c_idx] = rescale_intensity(image_out[c_idx], out_range=(0, 1))

        img_stack.append(image_out)
    img_stack = np.stack(img_stack)
    with open_ome_zarr(
        output_data_dir / f"{fov_name[1:].replace('/', '_')}_track_{track_ids[0]}.zarr",
        mode="w",
        layout="fov",
        channel_names=channels_to_display,
    ) as dataset:
        dataset["0"] = img_stack

    with open_ome_zarr(
        output_data_unaligned_dir
        / f"{fov_name[1:].replace('/', '_')}_track_{track_ids[0]}.zarr",
        mode="w",
        layout="fov",
        channel_names=channels_to_display,
    ) as dataset:
        dataset["0"] = img_stack


# %%
# We want to align only the regions where these tracks matched the reference
# ref_infection_tracks = infection_df.iloc[0]
# ref_division_tracks = cell_division_df.iloc[0]
class Color(NamedTuple):
    r: float
    g: float
    b: float


BOP_ORANGE = Color(0.972549, 0.6784314, 0.1254902)
BOP_BLUE = Color(BOP_ORANGE.b, BOP_ORANGE.g, BOP_ORANGE.r)
GREEN = Color(0.0, 1.0, 0.0)
MAGENTA = Color(1.0, 0.0, 1.0)

gdrive_path = (
    Path().home()
    / "mydata/gdrive/publications/dynaCLR/2025_dynaCLR_paper/fig_manuscript_svg/figure_organelle_remodeling/figure_parts"
)
output_plot_dir = Path(
    gdrive_path / f"SEC61B/aligned_response_{condition}/alignment_pngs_{condition}"
)
output_plot_dir.mkdir(parents=True, exist_ok=True)

output_plot_unaligned_dir = Path(
    gdrive_path / f"SEC61B/unaligned_response_{condition}/unaligned_pngs_{condition}"
)
output_plot_unaligned_dir.mkdir(parents=True, exist_ok=True)


def create_movie_frames(input_zarr, output_dir, time_indices, condition):
    with open_ome_zarr(input_zarr) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        for t_idx in time_indices:
            if t_idx >= T:
                continue

            img_rgb = np.zeros((Y, X, 3), dtype=np.float32)

            if condition != "organelle_only":
                phase = dataset["0"][t_idx, 0, 0, :]
                img_rgb[:, :, 0] = phase
                img_rgb[:, :, 1] = phase
                img_rgb[:, :, 2] = phase

            if condition == "cell_division":
                fluo = dataset["0"][t_idx, 1, 0, :]
                img_rgb[:, :, 1] += fluo
            elif condition == "infection":
                fluo = dataset["0"][t_idx, 1, 0, :]
                img_rgb[:, :, 0] += fluo
                img_rgb[:, :, 2] += fluo
            elif condition == "infection_organelle":
                fluo = dataset["0"][t_idx, 1, 0, :]
                img_rgb[:, :, 1] += fluo
            elif condition == "organelle_only":
                fluo = dataset["0"][t_idx, 0, 0, :]
                img_rgb[:, :, 1] += fluo
            else:
                warnings.warn(f"Condition {condition} not implemented")

            img_rgb = np.clip(img_rgb, 0, 1)

            plt.figure(figsize=(5.5, 5.5), facecolor="none")
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.savefig(
                output_dir / f"t{t_idx}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )
            plt.close()


# %%
# Generate the aligned movie frames (using the warping path)
for i, row in condition_df.head(6).iterrows():
    fov_name = row.fov_name
    track_id = ast.literal_eval(row.track_ids)

    condition_warp_path = ast.literal_eval(row.warp_path)
    condition_start_timepoint = row.start_timepoint

    for ref_idx, query_idx in condition_warp_path:
        actual_idx = int(query_idx + condition_start_timepoint)
        # sample every timepoint
        if ref_idx % 1 == 0:
            output_dir_fov = (
                output_plot_dir
                / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}"
            )
            output_dir_fov.mkdir(parents=True, exist_ok=True)

            zarr_path = (
                output_data_dir
                / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}.zarr"
            )
            create_movie_frames(zarr_path, output_dir_fov, [actual_idx], condition)

    # Create aligned video
    video_path = (
        output_plot_dir / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}.mp4"
    )

    try:
        ffmpeg.input(
            output_plot_dir
            / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}"
            / "*.png",
            pattern_type="glob",
            framerate=10,
        ).output(
            str(video_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
        ).overwrite_output().run(
            quiet=True
        )
        print(f"Aligned video created successfully: {video_path}")
    except ffmpeg.Error as e:
        print(
            f"Error creating aligned video: {e.stderr.decode() if e.stderr else str(e)}"
        )

# %%
# Generate the unaligned movie frames (using sequential timepoints)
for i, row in condition_df.head(6).iterrows():
    fov_name = row.fov_name
    track_id = ast.literal_eval(row.track_ids)

    zarr_path = (
        output_data_unaligned_dir
        / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}.zarr"
    )

    with open_ome_zarr(zarr_path) as dataset:
        T, C, Z, Y, X = dataset.data.shape

        output_dir_fov = (
            output_plot_unaligned_dir
            / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}"
        )
        output_dir_fov.mkdir(parents=True, exist_ok=True)

        # Create frames for all timepoints
        create_movie_frames(zarr_path, output_dir_fov, range(T), condition)

    # Create unaligned video
    video_path = (
        output_plot_unaligned_dir
        / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}.mp4"
    )

    try:
        ffmpeg.input(
            output_plot_unaligned_dir
            / f"{fov_name[1:].replace('/', '_')}_track_{track_id[0]}"
            / "*.png",
            pattern_type="glob",
            framerate=10,
        ).output(
            str(video_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
        ).overwrite_output().run(
            quiet=True
        )
        print(f"Unaligned video created successfully: {video_path}")
    except ffmpeg.Error as e:
        print(
            f"Error creating unaligned video: {e.stderr.decode() if e.stderr else str(e)}"
        )

# %%
# Get the PC 1 and PC2 values for each aligned timepoint and plot them
_, _, pca_df = compute_pca(embeddings_dataset, n_components=8)

# Plot all tracks in one figure - simplified to only show PC1
plt.figure(figsize=(5.5, 4), facecolor="none")  # Transparent figure background

# Use a different color for each track
colors = plt.cm.tab10(np.linspace(0, 1, len(condition_df.head(6))))

for i, (row, color) in enumerate(zip(condition_df.head(6).iterrows(), colors)):
    idx, row_data = row
    fov_name = row_data.fov_name
    track_ids = ast.literal_eval(row_data.track_ids)

    condition_warp_path = ast.literal_eval(row_data.warp_path)
    condition_start_timepoint = row_data.start_timepoint

    # Get data for this track
    track_pca_df = pca_df[
        (pca_df.fov_name == fov_name) & (pca_df.track_id.isin(track_ids))
    ]

    if not track_pca_df.empty:
        track_label = f"{fov_name.split('/')[-1]}_track_{track_ids[0]}"

        # Sort by timepoint if available
        if "t" in track_pca_df.columns:
            track_pca_df = track_pca_df.sort_values("t")

        # Sample every 5 timepoints to reduce clutter
        track_pca_df = track_pca_df.iloc[::5]

        PC1 = track_pca_df["PCA1"].values

        # Define x-axis (either timepoints or just indices)
        x = np.arange(len(PC1))
        if "t" in track_pca_df.columns:
            x = track_pca_df["t"].values

        # Plot only PC1 with markers to show individual timepoints
        plt.plot(x, PC1, "-o", color=color, markersize=4, label=f"{track_label}")

plt.xlabel("Timepoint")
plt.ylabel("PC1 Value")
plt.title(f"PC1 over time for all tracks")
plt.legend(loc="best", fontsize="small")
plt.tight_layout()

# Set figure background transparent and remove extra whitespace
plt.gca().set_facecolor("none")  # Transparent axes background
plt.savefig(
    output_plot_dir / f"PCA1_all_tracks_{condition}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)
plt.savefig(
    output_plot_dir / f"PCA1_all_tracks_{condition}.pdf",
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)
plt.close()


# %%
# From the PNG make 3x4 grid of images. Each row is specific timepoitn from the same track
# %%
def create_grid_visualization(input_dir_dict, output_plot_dir, is_aligned=True):
    """
    Create a 3x4 grid of images, where each row shows specific timepoints from the same track.
    """
    suffix = "aligned" if is_aligned else "unaligned"
    output_grid_dir = (
        output_plot_dir.parent / f"grid_visualizations_{condition}_{suffix}"
    )
    output_grid_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with extra space at bottom for the arrow
    fig = plt.figure(figsize=(5.5, 4.5), constrained_layout=False, facecolor="none")
    # Adjust subplot positioning to leave room at bottom
    grid = fig.add_gridspec(
        3, 4, left=0.1, right=0.9, bottom=0.15, top=0.95, wspace=0.05, hspace=0.05
    )
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(4)] for i in range(3)])

    row_idx = 0
    for fov_id_dir, timepoints in input_dir_dict.items():
        if row_idx >= 3:  # Limit to 3 rows
            break
        for col_idx, timepoint in enumerate(timepoints):
            if col_idx >= 4:  # Limit to 4 columns
                continue
            img_path = Path(fov_id_dir) / f"t{timepoint}.png"
            if img_path.exists():
                img = mpimg.imread(img_path)
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].axis("off")
                # grid off
                axes[row_idx, col_idx].grid(False)
        row_idx += 1

    # Create a single axis for the arrow that spans the width of the columns
    arrow_ax = fig.add_axes([0.1, 0.12, 0.8, 0.02], facecolor="none")
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.axis("off")

    # Draw the arrow
    arrow_ax.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )

    # Add timepoint labels directly below each column
    for i, t in enumerate(timepoints):
        x_pos = 0.05 + (i * 0.9 / len(timepoints)) + (0.9 / (2 * len(timepoints)))
        arrow_ax.text(x_pos, -0.5, f"t={t}", ha="center", va="top", fontsize=10)

    plt.savefig(
        output_grid_dir / f"grid_visualization_{condition}_{suffix}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()


# %%
# Create grid visualizations for both aligned and unaligned data
# For aligned data
aligned_input_dir_dict = {}
aligned_timepoints = [35, 45, 55, 65]

for i, row in condition_df.head(6).iterrows():
    fov_name = row.fov_name
    track_id = ast.literal_eval(row.track_ids)
    fov_id = fov_name[1:].replace("/", "_") + "_track_" + str(track_id[0])
    aligned_input_dir_dict[str(output_plot_dir / fov_id)] = aligned_timepoints

# For unaligned data
unaligned_input_dir_dict = {}
# Use the same timepoints to better compare aligned vs unaligned
unaligned_timepoints = aligned_timepoints

for i, row in condition_df.head(6).iterrows():
    fov_name = row.fov_name
    track_id = ast.literal_eval(row.track_ids)
    fov_id = fov_name[1:].replace("/", "_") + "_track_" + str(track_id[0])
    unaligned_input_dir_dict[str(output_plot_unaligned_dir / fov_id)] = (
        unaligned_timepoints
    )

# %%
# Create grid visualizations
output_grid_plot_dir = Path(
    gdrive_path
    / f"SEC61B/aligned_response_{condition}/alignment_grid_plots_{condition}"
)
create_grid_visualization(aligned_input_dir_dict, output_grid_plot_dir, is_aligned=True)

output_grid_unaligned_plot_dir = Path(
    gdrive_path
    / f"SEC61B/unaligned_response_{condition}/unaligned_grid_plots_{condition}"
)
create_grid_visualization(
    unaligned_input_dir_dict, output_grid_unaligned_plot_dir, is_aligned=False
)
# %%
