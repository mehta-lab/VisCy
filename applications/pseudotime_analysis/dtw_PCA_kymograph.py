# %%
import ast
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from iohub import open_ome_zarr
from plotting_utils import (
    align_image_stacks,
    create_consensus_embedding,
    find_pattern_matches,
    identify_lineages,
    plot_reference_vs_full_lineages,
)

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_pca

# Plotting configuration
DARK_MODE = False  # Set to True for dark background, False for light background
CMAP = "magma"  # Default colormap
GDRIVE = True

# Create a custom logger for just this script
logger = logging.getLogger("viscy")
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add a console handler specifically for this logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")  # Simplified format
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Make sure the logger isn't affected by parent loggers
logger.propagate = False
# %%
CONDITION_TO_ALIGN = "infection"  # remodelling_no_sensor, remodelling_w_sensor, cell_division, organelle_only ,infection
CONDITION_EMBEDDINGS = "organelle_only"  # phase_n_organelle, organelle_only, infection
input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr"
)

if CONDITION_EMBEDDINGS == "phase_n_organelle":
    feature_path = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
    )
elif CONDITION_EMBEDDINGS == "organelle_only":
    feature_path = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_organelle_only_ntxent_192patch_ckpt52_rev8_GT.zarr"
    )
elif CONDITION_EMBEDDINGS == "infection":
    feature_path = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions_infection/2chan_192patch_100ckpt_timeAware_ntxent_GT.zarr"
    )
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)
embeddings_dataset = read_embedding_dataset(feature_path)
feature_df = embeddings_dataset["sample"].to_dataframe().reset_index(drop=True)

lineages = identify_lineages(feature_df)

logger.info(f"Found {len(lineages)} distinct lineages")

filtered_lineages = []
min_timepoints = 20
for fov_id, track_ids in lineages:
    # Get all rows for this lineage
    lineage_rows = feature_df[
        (feature_df["fov_name"] == fov_id) & (feature_df["track_id"].isin(track_ids))
    ]

    # Count the total number of timepoints
    total_timepoints = len(lineage_rows)

    # Only keep lineages with at least min_timepoints
    if total_timepoints >= min_timepoints:
        filtered_lineages.append((fov_id, track_ids))
logger.info(
    f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints"
)
# %%

if CONDITION_TO_ALIGN == "remodelling_no_sensor":
    # Get the reference lineage
    # From the B/2 SEC61B-DV
    reference_lineage_fov = "/B/2/001000"
    reference_lineage_track_id = 104
    reference_timepoints = [55, 70]  # organelle remodeling

# From the C/2 SEC61B-DV-pl40
elif (
    CONDITION_TO_ALIGN == "remodelling_w_sensor"
    or CONDITION_TO_ALIGN == "organelle_only"
):
    # reference_lineage_fov = "/C/2/000001"
    # reference_lineage_track_id = 115
    # reference_timepoints = [47, 70] #sensor rellocalization and partial remodelling

    # reference_lineage_fov = "/C/2/000001"
    # reference_lineage_track_id = 158
    # reference_timepoints = [44, 74]  # sensor rellocalization and partial remodelling

    reference_lineage_fov = "/C/2/001000"
    reference_lineage_track_id = [129]
    reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling

    # From nuclear rellocalization
    # reference_lineage_fov = "/C/2/001001"
    # reference_lineage_track_id = [130, 131, 132]
    # reference_timepoints = [10, 80]  # sensor rellocalization and partial remodelling

    # reference_lineage_fov = "/C/2/001001"
    # reference_lineage_track_id = [160, 161]
    # reference_timepoints = [10, 80]  # sensor rellocalization and partial remodelling

    # reference_lineage_fov = "/C/2/001001"
    # reference_lineage_track_id = [126, 127]
    # reference_timepoints = [20, 70]  # sensor rellocalization and partial remodelling

elif CONDITION_TO_ALIGN == "infection":
    reference_lineage_fov = "/C/2/001000"
    reference_lineage_track_id = [129]
    reference_timepoints = [8, 70]  # sensor rellocalization and partial remodelling
    reference_annotation_timepoints = {
        "uinfected": list(range(8, 22)),
        "infected": list(range(22, 70)),
    }


# Cell division
elif CONDITION_TO_ALIGN == "cell_division":
    reference_lineage_fov = "/C/2/000001"
    reference_lineage_track_id = [107, 108, 109]
    reference_timepoints = [25, 70]

# Get the reference pattern
reference_pattern = None
reference_lineage = []
for fov_id, track_ids in filtered_lineages:
    if fov_id == reference_lineage_fov and all(
        track_id in track_ids for track_id in reference_lineage_track_id
    ):
        logger.info(
            f"Found reference pattern for {fov_id} {reference_lineage_track_id}"
        )
        reference_pattern = embeddings_dataset.sel(
            sample=(fov_id, reference_lineage_track_id)
        ).features.values
        reference_lineage.append(reference_pattern)
        break
if reference_pattern is None:
    raise ValueError(
        f"Reference pattern not found for {reference_lineage_fov} {reference_lineage_track_id}"
    )
reference_pattern = np.concatenate(reference_lineage)
reference_pattern = reference_pattern[reference_timepoints[0] : reference_timepoints[1]]

# %%
output_gdrive_path = Path(
    "/home/eduardo.hirata/mydata/gdrive/publications/dynaCLR/2025_dynaCLR_paper/fig_manuscript_svg/figure_alignment_embeddings/figure_parts"
)
output_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/figure"
)
if GDRIVE:
    save_path = (
        output_gdrive_path
        / f"SEC61B/PCA_kymograph_feature_{CONDITION_EMBEDDINGS}_aligning_{CONDITION_TO_ALIGN}"
    )
else:
    save_path = (
        output_root
        / f"SEC61B/PCA_kymograph_feature_{CONDITION_EMBEDDINGS}_aligning_{CONDITION_TO_ALIGN}"
    )
save_path.mkdir(parents=True, exist_ok=True)

# Load the matches
condition_df_path = (
    output_root / f"SEC61B/20241107_SEC61B_{CONDITION_TO_ALIGN}_matching_lineages.csv"
)
condition_df = pd.read_csv(condition_df_path)

n_cells = 5
top_n_aligned_cells = condition_df.head(n_cells)


_, _, pca_df = compute_pca(embeddings_dataset, n_components=8)

# Create arrays to store aligned PCA data for each cell
all_aligned_pca_data = []
num_pcs = 8  # Number of PCs from compute_pca
reference_length = reference_timepoints[1] - reference_timepoints[0]

# Process each of the top aligned cells
for i, row in top_n_aligned_cells.iterrows():
    fov_name = row["fov_name"]
    track_ids = ast.literal_eval(row["track_ids"])
    warp_path = ast.literal_eval(row["warp_path"])
    start_time = int(row["start_timepoint"])

    # Get PCA data for this track
    track_pca_df = pca_df[
        (pca_df.fov_name == fov_name) & (pca_df.track_id.isin(track_ids))
    ]

    # Get the PCA column names
    pca_cols = [f"PCA{j+1}" for j in range(num_pcs)]

    # Create an aligned array to store PCA values for each timepoint
    aligned_pca = np.zeros((reference_length, num_pcs))

    # Map each reference timepoint to the corresponding lineage timepoint
    for ref_idx, query_idx in warp_path:
        lineage_idx = int(start_time + query_idx)
        if 0 <= lineage_idx < len(track_pca_df):
            # Extract all PCA columns at once
            pca_values = track_pca_df.iloc[lineage_idx][pca_cols].values
            aligned_pca[ref_idx] = pca_values

    # Fill any missing values for the aligned PCs
    ref_indices_in_path = set(i for i, _ in warp_path)
    for ref_idx in range(reference_length):
        if ref_idx not in ref_indices_in_path:
            closest_ref_idx = min(ref_indices_in_path, key=lambda x: abs(x - ref_idx))
            closest_matches = [(i, q) for i, q in warp_path if i == closest_ref_idx]
            if closest_matches:
                closest_query_idx = closest_matches[0][1]
                lineage_idx = int(start_time + closest_query_idx)
                if 0 <= lineage_idx < len(track_pca_df):
                    # Extract all PCA columns at once
                    pca_values = track_pca_df.iloc[lineage_idx][pca_cols].values
                    aligned_pca[ref_idx] = pca_values
    all_aligned_pca_data.append(aligned_pca)

# Stack all aligned PCA data for visualization
stacked_pca_data = np.stack(all_aligned_pca_data, axis=0)

# Calculate the average across all cells
avg_pca_data = np.mean(stacked_pca_data, axis=0)

# %%
# Create heatmaps for each individual cell

# Plot individual cell heatmaps
for i in range(len(all_aligned_pca_data)):
    cell_data = all_aligned_pca_data[i]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set style based on global dark mode setting
    plt.style.use("dark_background" if DARK_MODE else "default")

    # Create the heatmap
    im = ax.imshow(
        cell_data.T,  # Transpose to have PC components on y-axis
        aspect="auto",
        cmap=CMAP,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("PC Value")

    # Set labels
    ax.set_xlabel("Time")
    ax.set_ylabel("PC Component")
    ax.set_title(f"Aligned PC Components for Cell {i+1}")

    # Set y-ticks to PC numbers
    ax.set_yticks(np.arange(num_pcs))
    ax.set_yticklabels([f"PCA{j+1}" for j in range(num_pcs)])

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path / f"cell_{i+1}_pca_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

# %%
# Create heatmap for the average across all cells
# Set style based on global dark mode setting
plt.style.use("dark_background" if DARK_MODE else "default")
fig, ax = plt.subplots(figsize=(12, 8))

# Create the heatmap for average data
im = ax.imshow(
    avg_pca_data.T,  # Transpose to have PC components on y-axis
    aspect="auto",
    cmap=CMAP,
    interpolation="nearest",
)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Average PC Value")

# Set labels
ax.set_xlabel("Time")
ax.set_ylabel("PC Component")
ax.set_title(f"Average Aligned PC Components (n={n_cells})")

# Set y-ticks to PC numbers
ax.set_yticks(np.arange(num_pcs))
ax.set_yticklabels([f"PCA{j+1}" for j in range(num_pcs)])

# Save the figure
plt.tight_layout()
plt.savefig(save_path / "average_pca_heatmap.pdf", dpi=300, bbox_inches="tight")
# plt.close()

# %%
# Create a figure showing both individual cells and the average
plt.style.use("dark_background" if DARK_MODE else "default")
fig = plt.figure(figsize=(15, 10))

# Create a grid of subplots - top row for individual cells, bottom for average
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Plot individual cells in the top row
for i in range(min(n_cells, 3)):  # Show up to 3 individual cells
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(
        all_aligned_pca_data[i].T,
        aspect="auto",
        cmap=CMAP,
        interpolation="nearest",
    )
    ax.set_title(f"Cell {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("PC Component")
    ax.set_yticks(np.arange(num_pcs))
    ax.set_yticklabels([f"PCA{j+1}" for j in range(num_pcs)])

# Plot the average in the bottom row, spanning all columns
ax_avg = fig.add_subplot(gs[1, :])
im_avg = ax_avg.imshow(
    avg_pca_data.T, aspect="auto", cmap=CMAP, interpolation="nearest"
)
ax_avg.set_title(f"Average (n={n_cells})")
ax_avg.set_xlabel("Time")
ax_avg.set_ylabel("PC Component")
ax_avg.set_yticks(np.arange(num_pcs))
ax_avg.set_yticklabels([f"PCA{j+1}" for j in range(num_pcs)])

# Add a colorbar for the average plot
cbar = plt.colorbar(im_avg, ax=ax_avg)
cbar.set_label("PC Value")

# Add overall title
plt.suptitle(f"PCA Components Over Aligned Time ({CONDITION_EMBEDDINGS})", fontsize=16)

# Save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.savefig(save_path / "combined_pca_heatmap.pdf", dpi=300, bbox_inches="tight")
# plt.close()

# %%
# Create a figure to plot PC1 over time with annotations
plt.style.use("dark_background" if DARK_MODE else "default")
fig, ax = plt.subplots(figsize=(2, 2))

# Plot PC1 for each cell
# TODO change for PC2
PC_COMPONENT = 1

time_points = np.arange(reference_length)
for i in range(min(n_cells, 3)):
    pc_values = all_aligned_pca_data[i][:, PC_COMPONENT - 1]  # Selected PC component
    ax.plot(time_points, pc_values, label=f"Cell {i+1}", alpha=0.7, linewidth=2)

# Plot average PC1
avg_pc = avg_pca_data[:, PC_COMPONENT - 1]
# Use white line for dark mode, black line for light mode
line_color = "w" if DARK_MODE else "k"
# ax.plot(time_points, avg_pc1, f"{line_color}-", label="Average", linewidth=3)

# Add annotation regions based on reference_annotation_timepoints
if CONDITION_TO_ALIGN == "infection" and "reference_annotation_timepoints" in locals():
    y_min, y_max = ax.get_ylim()

    # Define different colors based on dark mode
    if DARK_MODE:
        annotation_colors = {"uinfected": "blue", "infected": "red"}
        text_color = "white"
    else:
        annotation_colors = {"uinfected": "blue", "infected": "orange"}
        text_color = "black"

    for annotation, timepoints in reference_annotation_timepoints.items():
        # Create a span for each annotation period
        ax.axvspan(
            min(timepoints),
            max(timepoints),
            alpha=0.2,
            color=annotation_colors.get(annotation, "gray"),
            label=annotation,
        )

        # Add text annotation in the middle of the span
        mid_point = (min(timepoints) + max(timepoints)) / 2
        # ax.text(
        #     mid_point,
        #     y_max - (y_max - y_min) * 0.05,
        #     annotation,
        #     horizontalalignment="center",
        #     color=text_color,
        #     fontsize=12,
        # )
# Add labels and legend
ax.set_xlabel("Pseudotime")
ax.set_ylabel(f"PC{PC_COMPONENT}")
# ax.set_title("PC1 Over Time After Alignment")
# ax.legend(loc="best")
# ax.grid(True, alpha=0.1)

# Set the x-axis limit to end at 60
ax.set_xlim(right=60)

# Save the figure without legend
plt.tight_layout()
pc_name = f"pc{PC_COMPONENT}"
plt.savefig(
    save_path / f"{pc_name}_over_time_annotated.pdf", dpi=300, bbox_inches="tight"
)

# Create a separate figure for just the legend
figlegend = plt.figure(figsize=(3, 2))

# Get the legend handles and labels from the original plot
handles, labels = ax.get_legend_handles_labels()
# Create the legend on the new figure
figlegend.legend(handles, labels, loc="center")
figlegend.tight_layout()
# Save the legend figure
figlegend.savefig(save_path / f"{pc_name}_legend.pdf", dpi=300, bbox_inches="tight")

# plt.close()

# %%
from skimage.exposure import rescale_intensity, adjust_gamma

from viscy.data.triplet import TripletDataModule

channels_to_display = [
    "Phase3D",
    "raw GFP EX488 EM525-45",
    "raw mCherry EX561 EM600-37",
]
z_range = (15, 30)
yx_patch_size = (192, 192)

# Cach
image_cache = {}
for i, row in top_n_aligned_cells.iterrows():
    fov_name = row["fov_name"]
    track_ids = ast.literal_eval(row["track_ids"])

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
    img_tczyx = []
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
            else:
                image_out[c_idx] = np.max(images[c_idx], axis=0)
        img_tczyx.append(image_out)
    img_tczyx = np.array(img_tczyx)
    image_cache[f"{fov_name[1:].replace('/', '_')}_track_{track_ids[0]}"] = img_tczyx
# %%
t_indices = [15, 35, 55]  # Reference timepoints we want to visualize

fig, axes = plt.subplots(3, len(t_indices), figsize=(len(t_indices) * 3, 3 * 3))

# Get the first 3 cells from the image cache
image_items = list(image_cache.items())[:3]

# Process each cell
for row in range(min(3, len(top_n_aligned_cells))):
    # Get the image data for this cell
    key, img = image_items[row]

    # Get alignment information from the DataFrame
    aligned_cell_data = top_n_aligned_cells.iloc[row]
    warp_path = ast.literal_eval(aligned_cell_data["warp_path"])
    start_time = int(aligned_cell_data["start_timepoint"])

    # Find the corresponding query timepoints for our reference t_indices
    query_t_indices = []
    for ref_t in t_indices:
        # Find the closest reference timepoint in the warp path
        closest_ref_idx = min(warp_path, key=lambda x: abs(x[0] - ref_t))
        # Get corresponding query timepoint
        query_idx = closest_ref_idx[1]
        actual_query_t = start_time + query_idx
        query_t_indices.append(actual_query_t)

    for col, (ref_t, query_t) in enumerate(zip(t_indices, query_t_indices)):
        mcherry_clim = (117, 220)
        gfp_clim = (130, 200)
        phase_clim = (-0.2, 0.25)
        gamma_phase = 1
        gamma_gfp = 0.8
        gamma_mcherry = 1
        T, C, Z, Y, X = img.shape

        # Make sure query_t is within bounds
        query_t = min(max(0, query_t), T - 1)

        img_phase = img[query_t, 0].copy()
        img_gfp = img[query_t, 1].copy()
        img_mcherry = img[query_t, 2].copy()

        img_rgb = np.zeros((Y, X, 3), dtype=np.float32)
        # Rescale the channels
        img_phase = adjust_gamma(
            rescale_intensity(img_phase, in_range=phase_clim, out_range=(0, 1)),
            gamma=gamma_phase,
        )
        img_gfp = adjust_gamma(
            rescale_intensity(img_gfp, in_range=gfp_clim, out_range=(0, 1)),
            gamma=gamma_gfp,
        )
        img_mcherry = adjust_gamma(
            rescale_intensity(img_mcherry, in_range=mcherry_clim, out_range=(0, 1)),
            gamma=gamma_mcherry,
        )

        # image to plot and blending
        if CONDITION_EMBEDDINGS == "infection":
            img_rgb[:, :, 0] = img_phase + img_mcherry
            img_rgb[:, :, 1] = img_phase
            img_rgb[:, :, 2] = img_phase + img_mcherry
        elif CONDITION_EMBEDDINGS == "organelle_only":
            # img_rgb[:, :, 0] = 0
            img_rgb[:, :, 1] = img_gfp
            # img_rgb[:, :, 2] = 0
        else:
            img_rgb[:, :, 0] = img_phase
            img_rgb[:, :, 1] = img_phase + img_gfp
            img_rgb[:, :, 2] = img_phase

        axes[row, col].imshow(img_rgb)
        axes[row, col].axis("off")

        # Add reference and query timepoint labels
        # axes[row, col].text(
        #     10,
        #     10,
        #     f"Ref: {ref_t}\nReal: {query_t}",
        #     color="white",
        #     fontsize=8,
        #     backgroundcolor=(0, 0, 0, 0.5),
        #     ha="left",
        #     va="top",
        # )

# Apply tight layout first
fig.tight_layout()

fig.savefig(
    save_path
    / f"pc_over_time_annotated_time_{t_indices[0]}_{t_indices[1]}_{t_indices[2]}.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    save_path
    / f"pc_over_time_annotated_time_{t_indices[0]}_{t_indices[1]}_{t_indices[2]}.pdf",
    dpi=300,
    bbox_inches="tight",
)

# %%
