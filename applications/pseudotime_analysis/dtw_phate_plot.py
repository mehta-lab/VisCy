# %%
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize

from viscy.representation.embedding_writer import read_embedding_dataset

# %%
input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr"
)
feature_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)

embeddings_dataset = read_embedding_dataset(feature_path)
feature_df = embeddings_dataset["sample"].to_dataframe().reset_index(drop=True)

cell_division_matching_lineages_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/SEC61B/20241107_SEC61B_cell_division_matching_lineages.csv"
infection_matching_lineages_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/SEC61B/20241107_SEC61B_infection_matching_lineages.csv"

organelle_features_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/SEC61B/20241107_SEC61B_organelle_features.zarr"

cell_division_df = pd.read_csv(cell_division_matching_lineages_path)
infection_df = pd.read_csv(infection_matching_lineages_path)

# remove the cells with beads data in fov_name /C/1/*
# Check if any rows are removed
n_before = len(feature_df)
feature_df = feature_df[~feature_df["fov_name"].str.contains("/C/1/")]
n_after = len(feature_df)

print(f"Removed {n_before - n_after} rows containing '/C/1/' in fov_name")

# Remove it for cell_division_df and infection_df
cell_division_df = cell_division_df[~cell_division_df["fov_name"].str.contains("/C/1/")]
infection_df = infection_df[~infection_df["fov_name"].str.contains("/C/1/")]

# Plot phate map
PHATE1 = embeddings_dataset["PHATE1"].values
PHATE2 = embeddings_dataset["PHATE2"].values


# %%
# Create a function to visualize synchronized trajectories
def plot_synchronized_trajectories(
    alignment_df,
    feature_df,
    alignment_type,
    cmap_name="viridis",
    alpha_bg=0.1,
    alpha_trajectories=0.8,
    marker_size=30,
    line_width=2,
):
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create background of all embeddings
    ax.scatter(
        PHATE1, PHATE2, s=5, alpha=alpha_bg, color="lightgray", label="All embeddings"
    )

    # Create color map for trajectory progression
    cmap = plt.get_cmap(cmap_name)

    # Sort by distance to get the best alignments first
    alignment_df = alignment_df.sort_values(by="distance", ascending=True)

    # Use reference cell as the first trajectory
    reference_cell = alignment_df.iloc[0]
    ref_fov_name = reference_cell["fov_name"]
    ref_track_ids = ast.literal_eval(reference_cell["track_ids"])

    # Extract reference trajectory data
    ref_lineage = feature_df[
        (feature_df["fov_name"] == ref_fov_name)
        & (feature_df["track_id"].isin(ref_track_ids))
    ].sort_values(by="t")

    # Extract reference PHATE coordinates
    ref_phate1 = ref_lineage["PHATE1"].values
    ref_phate2 = ref_lineage["PHATE2"].values
    ref_times = ref_lineage["t"].values

    # Plot reference trajectory
    # ax.plot(
    #     ref_phate1,
    #     ref_phate2,
    #     color="red",
    #     linewidth=line_width,
    #     label="Reference trajectory",
    # )

    # Normalize the time points for coloring
    norm_times = (ref_times - ref_times.min()) / (ref_times.max() - ref_times.min())
    print(f"ref_times: {ref_times.min()} /{ref_times.max()}")
    print(ref_times)
    for i, (phate1, phate2, norm_t) in enumerate(
        zip(ref_phate1, ref_phate2, norm_times)
    ):
        ax.scatter(
            phate1,
            phate2,
            color=cmap(norm_t),
            s=marker_size,
            edgecolor="black",
            linewidth=0.5,
        )

    # Process all other aligned trajectories
    # Limit to top 5 alignments for clarity
    for idx, cell_data in alignment_df.iloc[1:5].iterrows():
        fov_name = cell_data["fov_name"]
        track_ids = ast.literal_eval(cell_data["track_ids"])
        warp_path = ast.literal_eval(cell_data["warp_path"])
        start_timepoint = cell_data["start_timepoint"]

        # Extract lineage data
        lineage = feature_df[
            (feature_df["fov_name"] == fov_name)
            & (feature_df["track_id"].isin(track_ids))
        ].sort_values(by="t")

        # Get PHATE coordinates for this trajectory
        lineage_phate1 = lineage["PHATE1"].values
        lineage_phate2 = lineage["PHATE2"].values

        # Use DTW to align and extract warped coordinates
        aligned_coords = []
        for ref_idx, query_idx in warp_path:
            # ref_idx corresponds to reference trajectory time
            # query_idx corresponds to this trajectory's time, adjusted by start_timepoint
            actual_idx = int(query_idx + start_timepoint)
            if actual_idx < len(lineage_phate1):
                aligned_coords.append(
                    (
                        lineage_phate1[actual_idx],
                        lineage_phate2[actual_idx],
                        norm_times[ref_idx],  # Use reference time for color mapping
                    )
                )

        if aligned_coords:
            aligned_x, aligned_y, aligned_t = zip(*aligned_coords)

            # Plot this aligned trajectory
            # ax.plot(
            #     aligned_x,
            #     aligned_y,
            #     linewidth=line_width,
            #     alpha=alpha_trajectories,
            #     color=f"C{idx%10}",
            # )

            # Plot points with color based on aligned time
            for x, y, t in aligned_coords:
                ax.scatter(
                    x,
                    y,
                    color=cmap(t),
                    s=marker_size,
                    alpha=alpha_trajectories,
                    edgecolor="black",
                    linewidth=0.5,
                )

    # Plot styling
    ax.set_title(f"Synchronized {alignment_type} Trajectories in PHATE Space")
    ax.set_xlabel("PHATE1")
    ax.set_ylabel("PHATE2")

    # Create a proper colorbar with normalization
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Normalized Time")

    ax.legend()
    plt.tight_layout()

    return fig


# %%
# Plot cell division alignments
division_plot = plot_synchronized_trajectories(
    cell_division_df, feature_df, "Cell Division", cmap_name="viridis"
)
division_plot.savefig("./SEC61B/cell_division_trajectories.png", dpi=300)
division_plot.show()

# Plot infection alignments
infection_plot = plot_synchronized_trajectories(
    infection_df, feature_df, "Infection", cmap_name="plasma"
)
infection_plot.savefig("./SEC61B/infection_trajectories.png", dpi=300)
infection_plot.show()


# %%
# Optional: Comparative visualization showing both alignment types
def plot_comparative_alignments(cell_division_df, infection_df):
    fig, ax = plt.subplots(figsize=(15, 12))

    # Background of all embeddings
    ax.scatter(
        PHATE1, PHATE2, s=5, alpha=0.1, color="lightgray", label="All embeddings"
    )

    # Sort by distance
    cell_division_df = cell_division_df.sort_values(by="distance", ascending=True)
    infection_df = infection_df.sort_values(by="distance", ascending=True)

    # Get top 5 from each alignment type
    for i, df in enumerate([cell_division_df.iloc[:5], infection_df.iloc[:5]]):
        label_prefix = "Cell Division" if i == 0 else "Infection"
        color_base = "blue" if i == 0 else "red"

        for j, cell_data in df.iterrows():
            fov_name = cell_data["fov_name"]
            track_ids = ast.literal_eval(cell_data["track_ids"])

            # Extract lineage data
            lineage = feature_df[
                (feature_df["fov_name"] == fov_name)
                & (feature_df["track_id"].isin(track_ids))
            ].sort_values(by="t")

            # Get PHATE coordinates
            lineage_phate1 = lineage["PHATE1"].values
            lineage_phate2 = lineage["PHATE2"].values

            label = f"{label_prefix} #{j}" if j < 5 else None
            ax.scatter(
                lineage_phate1,
                lineage_phate2,
                color=color_base,
                alpha=0.7,
                linewidth=2,
                linestyle="-" if i == 0 else "--",
                label=label,
            )

    ax.set_title("Comparison of Cell Division vs Infection Trajectories")
    ax.set_xlabel("PHATE1")
    ax.set_ylabel("PHATE2")
    ax.legend()
    plt.tight_layout()
    fig.savefig("comparative_trajectories.png", dpi=300)
    plt.show()


# %%
plot_comparative_alignments(cell_division_df, infection_df)


# %%
def find_top_matching_tracks(cell_division_df, infection_df, n_top=10) -> pd.DataFrame:
    # Find common tracks between datasets
    intersection_df = pd.merge(
        cell_division_df,
        infection_df,
        on=["fov_name", "track_ids"],
        how="inner",
        suffixes=("_df1", "_df2"),
    )

    # Add column with sum of the values
    intersection_df["distance_sum"] = (
        intersection_df["distance_df1"] + intersection_df["distance_df2"]
    )

    # Find rows with the smallest sum
    intersection_df.sort_values(by="distance_sum", ascending=True, inplace=True)
    return intersection_df.head(n_top)


# Side-by-side comparison with alignment-based coloring
def plot_aligned_tracks_side_by_side(
    intersection_df: pd.DataFrame, feature_df: pd.DataFrame, n_top=5
):
    # Limit to top N
    intersection_df = intersection_df.head(n_top)

    # Now create the figure with subplots - 2 columns for each track
    n_rows = len(intersection_df)

    # Adjust figure size to accommodate colorbars on the right side
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4 * n_rows))

    # Adjust for single row case
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    # Define colormaps
    div_cmap = plt.get_cmap("viridis")
    inf_cmap = plt.get_cmap("plasma")

    # Get reference track info for both cases
    # Division reference
    div_ref_fov = cell_division_df.iloc[0]["fov_name"]
    div_ref_track_ids = ast.literal_eval(cell_division_df.iloc[0]["track_ids"])
    div_ref_lineage = feature_df[
        (feature_df["fov_name"] == div_ref_fov)
        & (feature_df["track_id"].isin(div_ref_track_ids))
    ].sort_values(by="t")
    div_ref_times = div_ref_lineage["t"].values

    # Infection reference
    inf_ref_fov = infection_df.iloc[0]["fov_name"]
    inf_ref_track_ids = ast.literal_eval(infection_df.iloc[0]["track_ids"])
    inf_ref_lineage = feature_df[
        (feature_df["fov_name"] == inf_ref_fov)
        & (feature_df["track_id"].isin(inf_ref_track_ids))
    ].sort_values(by="t")
    inf_ref_times = inf_ref_lineage["t"].values

    # Normalize reference times
    div_norm_ref_times = (
        (div_ref_times - div_ref_times.min())
        / (div_ref_times.max() - div_ref_times.min())
        if len(div_ref_times) > 1
        else np.array([0.5])
    )
    inf_norm_ref_times = (
        (inf_ref_times - inf_ref_times.min())
        / (inf_ref_times.max() - inf_ref_times.min())
        if len(inf_ref_times) > 1
        else np.array([0.5])
    )

    for i, row in enumerate(intersection_df.itertuples()):
        # Get the track information
        fov_name = row.fov_name
        track_ids = ast.literal_eval(row.track_ids)

        # Get DTW alignment info
        div_warp_path = ast.literal_eval(row.warp_path_df1)
        div_start_timepoint = row.start_timepoint_df1

        inf_warp_path = ast.literal_eval(row.warp_path_df2)
        inf_start_timepoint = row.start_timepoint_df2

        # Get all cells in this lineage
        lineage_data = feature_df[
            (feature_df["fov_name"] == fov_name)
            & (feature_df["track_id"].isin(track_ids))
        ].sort_values(by="t")

        if len(lineage_data) == 0:
            print(f"No data found for tracks {track_ids} in FOV {fov_name}")
            continue

        # Get timepoints and coordinates
        phate1 = lineage_data["PHATE1"].values
        phate2 = lineage_data["PHATE2"].values
        times = lineage_data["t"].values

        # Plot background in both subplots
        for j in range(2):
            axes[i, j].scatter(PHATE1, PHATE2, s=5, alpha=0.05, color="lightgray")

        # Left subplot - Cell Division
        axes[i, 0].set_title(
            f"Division: {fov_name}, Tracks {track_ids} (Score: {row.distance_df1:.3f})"
        )

        # Plot trajectory line
        # axes[i, 0].plot(phate1, phate2, color="blue", linewidth=2, alpha=0.7)

        # Apply alignment-based coloring for division
        for ref_idx, query_idx in div_warp_path:
            actual_idx = int(query_idx + div_start_timepoint)
            if 0 <= actual_idx < len(phate1) and ref_idx < len(div_norm_ref_times):
                color = div_cmap(div_norm_ref_times[ref_idx])
                axes[i, 0].scatter(
                    phate1[actual_idx],
                    phate2[actual_idx],
                    color=color,
                    s=60,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Right subplot - Infection
        axes[i, 1].set_title(
            f"Infection: {fov_name}, Tracks {track_ids} (Score: {row.distance_df2:.3f})"
        )

        # Plot trajectory line
        # axes[i, 1].plot(phate1, phate2, color="red", linewidth=2, alpha=0.7)

        # Apply alignment-based coloring for infection
        for ref_idx, query_idx in inf_warp_path:
            actual_idx = int(query_idx + inf_start_timepoint)
            if 0 <= actual_idx < len(phate1) and ref_idx < len(inf_norm_ref_times):
                color = inf_cmap(inf_norm_ref_times[ref_idx])
                axes[i, 1].scatter(
                    phate1[actual_idx],
                    phate2[actual_idx],
                    color=color,
                    s=60,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Add labels
        for j in range(2):
            axes[i, j].set_xlabel("PHATE1")
            axes[i, j].set_ylabel("PHATE2")

    # First adjust the layout to get proper spacing
    plt.tight_layout()

    # Add a small amount of space on the right for colorbars
    fig.subplots_adjust(right=0.85)

    # Create a colorbar axes for each colormap
    cbar_ax1 = fig.add_axes([0.88, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    cbar_ax2 = fig.add_axes([0.88, 0.15, 0.02, 0.3])

    # Add colorbars
    norm = Normalize(vmin=0, vmax=1)

    # Division colorbar
    sm_div = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm)
    sm_div.set_array([])
    cbar_div = fig.colorbar(sm_div, cax=cbar_ax1)
    cbar_div.set_label("Aligned Time (Division Ref.)")

    # Infection colorbar
    sm_inf = plt.cm.ScalarMappable(cmap=inf_cmap, norm=norm)
    sm_inf.set_array([])
    cbar_inf = fig.colorbar(sm_inf, cax=cbar_ax2)
    cbar_inf.set_label("Aligned Time (Infection Ref.)")

    fig.savefig(
        "./SEC61B/aligned_tracks_side_by_side.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return fig


# %%
# Find top matching tracks
matching_df = find_top_matching_tracks(cell_division_df, infection_df, n_top=10)

# %%
# Run the aligned side-by-side comparison
aligned_tracks_plot = plot_aligned_tracks_side_by_side(matching_df, feature_df, n_top=5)

# %%
# Add interactive plotly visualization with hover information
import plotly.express as px
import plotly.graph_objects as go


def create_interactive_phate_plot(feature_df):
    # Create a plotly figure using the same PHATE data as other plots
    fig = go.Figure()

    # Add all points with hover information
    fig.add_trace(
        go.Scatter(
            x=PHATE1,
            y=PHATE2,
            mode="markers",
            marker=dict(
                color=feature_df["t"],  # Color by time
                colorscale="viridis",
                size=5,
                opacity=0.7,
                colorbar=dict(title="Time"),
            ),
            text=feature_df.apply(
                lambda row: f"FOV: {row['fov_name']}<br>Track ID: {row['track_id']}<br>Time: {row['t']}",
                axis=1,
            ),
            hoverinfo="text+x+y",
            name="PHATE Embedding",
        )
    )

    # Update layout
    fig.update_layout(
        title="Interactive PHATE Visualization",
        xaxis_title="PHATE1",
        yaxis_title="PHATE2",
        width=1000,
        height=800,
        template="plotly_white",
        hovermode="closest",
    )

    return fig


# %%
interactive_plot = create_interactive_phate_plot(feature_df)
interactive_plot.write_html("./SEC61B/interactive_phate_visualization.html")
interactive_plot.show()

# %%
