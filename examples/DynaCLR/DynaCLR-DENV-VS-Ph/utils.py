"""Utility functions for visualization and analysis."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from skimage.exposure import rescale_intensity


def add_arrows(df, color, df_coordinates=["PHATE1", "PHATE2"]):
    """
    Add arrows to a plot to show direction of trajectory.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing custom coordinates (like PHATE coordinates (PHATE1, PHATE2))
    color : str
        Color for the arrows
    """
    from matplotlib.patches import FancyArrowPatch

    for i in range(df.shape[0] - 1):
        start = df.iloc[i]
        end = df.iloc[i + 1]
        arrow = FancyArrowPatch(
            (start[df_coordinates[0]], start[df_coordinates[1]]),
            (end[df_coordinates[0]], end[df_coordinates[1]]),
            color=color,
            arrowstyle="-",
            mutation_scale=10,
            lw=1,
            shrinkA=0,
            shrinkB=0,
        )
        plt.gca().add_patch(arrow)


def plot_phate_time_trajectories(
    df,
    output_dir="./phate_timeseries",
    highlight_tracks=None,
):
    """
    Generate a series of PHATE embedding plots for each timepoint, showing trajectories.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the PHATE embeddings
    output_dir : str, optional
        Directory to save the PNG files, by default "./phate_timeseries"
    highlight_tracks : dict, optional
        Dictionary specifying tracks to highlight, by default None
    """
    import os

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if highlight_tracks is None:
        # Default tracks to highlight
        highlight_tracks = {
            "infected": [("/B/4/9", 42)],
            "uninfected": [("/A/3/9", 19)],
        }

    os.makedirs(output_dir, exist_ok=True)

    # Get unique time points
    all_times = sorted(df["t"].unique())

    # Calculate global axis limits to keep them consistent
    padding = 0.1  # Add padding to the limits for better visualization
    x_min = df["PHATE1"].min() - padding * (df["PHATE1"].max() - df["PHATE1"].min())
    x_max = df["PHATE1"].max() + padding * (df["PHATE1"].max() - df["PHATE1"].min())
    y_min = df["PHATE2"].min() - padding * (df["PHATE2"].max() - df["PHATE2"].min())
    y_max = df["PHATE2"].max() + padding * (df["PHATE2"].max() - df["PHATE2"].min())

    # Make sure the aspect ratio is 1:1 by using the same range for both axes
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        # Expand y-limits to match x-range
        center = (y_max + y_min) / 2
        y_min = center - x_range / 2
        y_max = center + x_range / 2
    else:
        # Expand x-limits to match y-range
        center = (x_max + x_min) / 2
        x_min = center - y_range / 2
        x_max = center + y_range / 2

    # Generate plots for each time step
    for t_idx, t in enumerate(all_times):
        plt.close("all")
        _fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111)

        # Plot historical points in gray (all points from previous time steps)
        if t_idx > 0:
            historical_df = df[df["t"] < t]
            ax.scatter(
                historical_df["PHATE1"],
                historical_df["PHATE2"],
                c="lightgray",
                s=10,
                alpha=0.15,
            )

        # Plot current time points
        current_df = df[df["t"] == t]

        # Plot infected vs uninfected points for current time
        for infection_state, color in [(1, "cornflowerblue"), (2, "salmon")]:
            points = current_df[current_df["infection"] == infection_state]
            ax.scatter(points["PHATE1"], points["PHATE2"], c=color, s=30, alpha=0.7)

        # Add track trajectories for highlighted cells
        for label, track_list in highlight_tracks.items():
            for fov_name, track_id in track_list:
                # Get all timepoints up to current time for this track
                track_data = df[
                    (df["fov_name"] == fov_name)
                    & (df["track_id"] == track_id)
                    & (df["t"] <= t)
                ].sort_values("t")

                if len(track_data) > 0:
                    # Draw trajectory using arrows
                    color = "red" if label == "infected" else "blue"

                    if len(track_data) > 1:
                        # Use the arrow function that works with PHATE1/PHATE2 columns
                        add_arrows(
                            track_data, color, df_coordinates=["PHATE1", "PHATE2"]
                        )

                    # Mark current position with a larger point
                    current_pos = track_data[track_data["t"] == t]
                    if len(current_pos) > 0:
                        ax.scatter(
                            current_pos["PHATE1"],
                            current_pos["PHATE2"],
                            s=150,
                            color=color,
                            edgecolor="black",
                            linewidth=1.5,
                            zorder=10,
                        )

        # Set the same axis limits for all frames
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Add legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Uninfected",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Infected",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=12,
                markeredgecolor="black",
                label="Highlighted Uninfected Track",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=12,
                markeredgecolor="black",
                label="Highlighted Infected Track",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Add labels and title with time info
        ax.set_title(f"ImageNet PHATE Embedding - Time: {t}")
        ax.set_xlabel("PHATE1")
        ax.set_ylabel("PHATE2")

        # Set equal aspect ratio for better visualization
        ax.set_aspect("equal")

        # Save figure
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/phate_embedding_t{t:03d}.png", dpi=300, bbox_inches="tight"
        )

        # Only show the first frame in the notebook
        if t == all_times[0]:
            plt.show()


def create_plotly_visualization(
    df,
    highlight_tracks=None,
    df_coordinates=["PHATE1", "PHATE2"],
    time_column="t",
    category_column="infection",
    category_labels={1: "Uninfected", 2: "Infected"},
    category_colors={1: "cornflowerblue", 2: "salmon"},
    highlight_colors={1: "blue", 2: "red"},
    title_prefix="PHATE Embedding",
    plot_size_xy=(1000, 800),
):
    """
    Create an interactive visualization using Plotly with a time slider.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the embedding coordinates
    highlight_tracks : dict, optional
        Dictionary specifying tracks to highlight, by default None
        Format: {category_name: [(fov_name, track_id), ...]}
        e.g., {"infected": [("/B/4/9", 42)], "uninfected": [("/A/3/9", 19)]}
        or {1: [("/A/3/9", 19)], 2: [("/B/4/9", 42)]} where 1=uninfected, 2=infected
    df_coordinates : list, optional
        Column names for the x and y coordinates, by default ["PHATE1", "PHATE2"]
    time_column : str, optional
        Column name for the time points, by default "t"
    category_column : str, optional
        Column name for the category to color by, by default "infection"
    category_labels : dict, optional
        Mapping from category values to display labels, by default {1: "Uninfected", 2: "Infected"}
    category_colors : dict, optional
        Mapping from category values to colors for markers, by default {1: "cornflowerblue", 2: "salmon"}
    highlight_colors : dict, optional
        Mapping from category values to colors for highlighted tracks, by default {1: "blue", 2: "red"}
    title_prefix : str, optional
        Prefix for the plot title, by default "PHATE Embedding"
    plot_size_xy : tuple, optional
        Width and height of the plot in pixels, by default (1000, 800)

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive Plotly figure
    """
    # Check if plotly is available
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Please install it using: pip install plotly")
        return None

    highlight_track_map = {}
    category_value_map = {"uninfected": 1, "infected": 2}
    for key, tracks in highlight_tracks.items():
        # If the key is a string like "infected" or "uninfected", convert to category value
        if isinstance(key, str) and key.lower() in category_value_map:
            category = category_value_map[key.lower()]
        else:
            # Otherwise use the key directly (assumed to be a category value)
            category = key
        highlight_track_map[category] = tracks

    # Get unique time points and categories
    all_times = sorted(df[time_column].unique())
    categories = sorted(df[category_column].unique())

    # Calculate global axis limits
    padding = 0.1
    x_min = df[df_coordinates[0]].min() - padding * (
        df[df_coordinates[0]].max() - df[df_coordinates[0]].min()
    )
    x_max = df[df_coordinates[0]].max() + padding * (
        df[df_coordinates[0]].max() - df[df_coordinates[0]].min()
    )
    y_min = df[df_coordinates[1]].min() - padding * (
        df[df_coordinates[1]].max() - df[df_coordinates[1]].min()
    )
    y_max = df[df_coordinates[1]].max() + padding * (
        df[df_coordinates[1]].max() - df[df_coordinates[1]].min()
    )

    # Make sure the aspect ratio is 1:1
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        center = (y_max + y_min) / 2
        y_min = center - x_range / 2
        y_max = center + x_range / 2
    else:
        center = (x_max + x_min) / 2
        x_min = center - y_range / 2
        x_max = center + y_range / 2

    # Pre-compute all track data to ensure consistency across frames
    track_data_cache = {}
    for category, track_list in highlight_track_map.items():
        for idx, (fov_name, track_id) in enumerate(track_list):
            track_key = f"{category}_{fov_name}_{track_id}"
            print(f"Processing track: {track_key}")
            # Get all data for this track
            full_track_data = df[
                (df["fov_name"] == fov_name) & (df["track_id"] == track_id)
            ].sort_values(time_column)

            print(f"Found {len(full_track_data)} points for track {track_key}")
            if len(full_track_data) > 0:
                track_data_cache[track_key] = full_track_data
                print(
                    f"Time points for {track_key}: {sorted(full_track_data[time_column].unique())}"
                )
            else:
                print(f"WARNING: No data found for track {track_key}")

    print(f"Track data cache keys: {list(track_data_cache.keys())}")

    # Prepare data for all frames of the animation
    frames = []

    # Create traces for each time point
    for t_idx, t in enumerate(all_times):
        frame_data = []

        # Historical data trace (all points from previous timepoints)
        if t_idx > 0:
            historical_df = df[df[time_column] < t]
            frame_data.append(
                go.Scatter(
                    x=historical_df[df_coordinates[0]],
                    y=historical_df[df_coordinates[1]],
                    mode="markers",
                    marker=dict(color="lightgray", size=5, opacity=0.2),
                    name="Historical",
                    hoverinfo="none",
                    showlegend=False,
                )
            )
        else:
            # Empty trace as placeholder
            frame_data.append(
                go.Scatter(
                    x=[], y=[], mode="markers", name="Historical", showlegend=False
                )
            )

        # Current time data
        current_df = df[df[time_column] == t]

        # Plot each category
        for category in categories:
            category_points = current_df[current_df[category_column] == category]
            if len(category_points) > 0:
                frame_data.append(
                    go.Scatter(
                        x=category_points[df_coordinates[0]],
                        y=category_points[df_coordinates[1]],
                        mode="markers",
                        marker=dict(
                            color=category_colors.get(category, "gray"),
                            size=8,
                            opacity=0.7,
                        ),
                        name=category_labels.get(category, f"Category {category}"),
                        hovertext=[
                            f"FOV: {row['fov_name']}, Track: {row['track_id']}, {category_labels.get(category, f'Category {category}')}"
                            for _, row in category_points.iterrows()
                        ],
                        hoverinfo="text",
                        showlegend=False,  # Never show legend
                    )
                )
            else:
                frame_data.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="markers",
                        name=category_labels.get(category, f"Category {category}"),
                        showlegend=False,  # Never show legend
                    )
                )

        # Add highlighted tracks
        for category, track_list in highlight_track_map.items():
            for idx, (fov_name, track_id) in enumerate(track_list):
                track_key = f"{category}_{fov_name}_{track_id}"

                if track_key in track_data_cache:
                    # Get the full track data from cache
                    full_track_data = track_data_cache[track_key]

                    # Filter for data up to current time for trajectory
                    track_data = full_track_data[full_track_data[time_column] <= t]

                    if len(track_data) > 0:
                        color = highlight_colors.get(category, "gray")
                        label = category_labels.get(category, f"Category {category}")

                        # Create single line trace for the entire trajectory
                        frame_data.append(
                            go.Scatter(
                                x=track_data[df_coordinates[0]],
                                y=track_data[df_coordinates[1]],
                                mode="lines",
                                line=dict(color=color, width=2),
                                name=f"Track {track_id} ({label})",
                                showlegend=False,  # Never show legend
                            )
                        )

                        # Add current position marker
                        current_pos = track_data[track_data[time_column] == t]

                        # If no data at current time but we have track data, show the last known position
                        if len(current_pos) == 0:
                            # Get the most recent position before current timepoint
                            latest_pos = track_data.iloc[-1:]

                            if t_idx == 0:
                                print(
                                    f"No current position for {track_key} at time {t}, using last known at {latest_pos[time_column].iloc[0]}"
                                )

                            # Add a semi-transparent marker at the last known position
                            frame_data.append(
                                go.Scatter(
                                    x=latest_pos[df_coordinates[0]],
                                    y=latest_pos[df_coordinates[1]],
                                    mode="markers",
                                    marker=dict(
                                        color=color,
                                        size=15,
                                        line=dict(color="black", width=1),
                                        opacity=0.5,  # Lower opacity for non-current positions
                                    ),
                                    name=f"Last Known Position - {label}",
                                    hovertext=[
                                        f"FOV: {row['fov_name']}, Track: {row['track_id']}, Last Seen at t={row[time_column]}, {label}"
                                        for _, row in latest_pos.iterrows()
                                    ],
                                    hoverinfo="text",
                                    showlegend=False,
                                )
                            )
                        else:
                            # Normal case - we have data at current timepoint
                            if t_idx == 0:
                                print(
                                    f"Found current position for {track_key} at time {t}"
                                )

                            frame_data.append(
                                go.Scatter(
                                    x=current_pos[df_coordinates[0]],
                                    y=current_pos[df_coordinates[1]],
                                    mode="markers",
                                    marker=dict(
                                        color=color,
                                        size=15,
                                        line=dict(color="black", width=1),
                                    ),
                                    name=f"Highlighted {label}",
                                    hovertext=[
                                        f"FOV: {row['fov_name']}, Track: {row['track_id']}, Highlighted {label}"
                                        for _, row in current_pos.iterrows()
                                    ],
                                    hoverinfo="text",
                                    showlegend=False,  # Never show legend
                                )
                            )

        # Create a frame for this time point
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Create the base figure with the first frame data
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=f"{title_prefix} - Time: {all_times[0]}",
            xaxis=dict(title=df_coordinates[0], range=[x_min, x_max]),
            yaxis=dict(
                title=df_coordinates[1],
                range=[y_min, y_max],
                scaleanchor="x",  # Make it 1:1 aspect ratio
                scaleratio=1,
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "x": 0.15,
                    "y": 0,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 16},
                        "prefix": "Time: ",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 0},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [str(t)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": str(t),
                            "method": "animate",
                        }
                        for t in all_times
                    ],
                }
            ],
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        ),
    )

    # Update figure layout
    fig.update_layout(
        width=plot_size_xy[0],
        height=plot_size_xy[1],
        margin=dict(l=50, r=50, t=100, b=100),
        template="plotly_white",
    )
    return fig


def create_image_visualization(
    image_cache,
    subplot_titles=["Mock Phase", "Mock RFP", "Infected Phase", "Infected RFP"],
    condition_keys=["uinfected_cache", "infected_cache"],
    channel_colormaps=["gray", "magma"],
    plot_size_xy=(1000, 800),
    horizontal_spacing=0.05,
    vertical_spacing=0.1,
):
    """
    Create an interactive visualization of images from image cache using Plotly with a time slider.

    Parameters
    ----------
    image_cache : dict
        Dictionary containing cached images by condition and timepoint
        Format: {"condition_key": {"images_by_timepoint": {t: image_array}}}
    subplot_titles : list, optional
        Titles for the subplots, by default ["Mock Phase", "Mock RFP", "Infected Phase", "Infected RFP"]
    condition_keys : list, optional
        Keys for the conditions in the image_cache, by default ["uinfected_cache", "infected_cache"]
    channel_colormaps : list, optional
        Colormaps for each channel, by default ["gray", "magma"]
    plot_size_xy : tuple, optional
        Width and height of the plot in pixels, by default (1000, 800)
    horizontal_spacing : float, optional
        Horizontal spacing between subplots, by default 0.05
    vertical_spacing : float, optional
        Vertical spacing between subplots, by default 0.1

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive Plotly figure
    """
    # Check if plotly is available
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly is not installed. Please install it using: pip install plotly")
        return None

    # Get all available timepoints from all conditions
    all_timepoints = []
    for condition_key in condition_keys:
        if (
            condition_key in image_cache
            and "images_by_timepoint" in image_cache[condition_key]
        ):
            all_timepoints.extend(
                list(image_cache[condition_key]["images_by_timepoint"].keys())
            )

    all_timepoints = sorted(list(set(all_timepoints)))
    print(f"All timepoints: {all_timepoints}")

    if not all_timepoints:
        print("No timepoints found in the image cache")
        return None

    # Create the figure with subplots
    fig = make_subplots(
        rows=len(condition_keys),
        cols=len(channel_colormaps),
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    # Create initial frame
    t_initial = all_timepoints[0]

    # Add each condition as a row
    for row_idx, condition_key in enumerate(condition_keys, 1):
        if (
            condition_key in image_cache
            and t_initial in image_cache[condition_key]["images_by_timepoint"]
        ):
            img = image_cache[condition_key]["images_by_timepoint"][t_initial]

            # Add each channel as a column
            for col_idx, colormap in enumerate(channel_colormaps, 1):
                cmap = cm.get_cmap(colormap)
                img = img[col_idx, 0]
                colored_img = cmap(img)

                # Convert to RGB format (remove alpha channel)
                colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

                if col_idx <= img.shape[0]:  # Make sure we have this channel
                    fig.add_trace(
                        go.Image(
                            z=colored_img,
                            x0=0,
                            y0=0,
                            dx=1,
                            dy=1,
                            colormodel="rgb",
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                else:
                    # Empty placeholder if channel doesn't exist
                    fig.add_trace(
                        go.Image(
                            z=np.zeros((10, 10, 3)),
                            colormodel="rgb",
                            x0=0,
                            y0=0,
                            dx=1,
                            dy=1,
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
        else:
            # Empty placeholders if condition or timepoint not found
            for col_idx, colormap in enumerate(channel_colormaps, 1):
                fig.add_trace(
                    go.Image(
                        z=np.zeros((10, 10, 3)),
                        colormodel="rgb",
                        x0=0,
                        y0=0,
                        dx=1,
                        dy=1,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    # Function to create a frame for a specific timepoint
    def create_frame_for_timepoint(t):
        frame_data = []

        for condition_key in condition_keys:
            if (
                condition_key in image_cache
                and t in image_cache[condition_key]["images_by_timepoint"]
            ):
                img = image_cache[condition_key]["images_by_timepoint"][t]

                for colormap in channel_colormaps:
                    col_idx = channel_colormaps.index(colormap)
                    cmap = cm.get_cmap(colormap)
                    img = img[col_idx, 0]
                    print(f"img shape: {img.shape}")
                    colored_img = cmap(img)

                    # Convert to RGB format (remove alpha channel)
                    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

                    if col_idx < img.shape[0]:  # Make sure we have this channel
                        frame_data.append(
                            go.Image(
                                z=colored_img,
                                colormodel="rgb",
                                x0=0,
                                y0=0,
                                dx=1,
                                dy=1,
                            )
                        )
                    else:
                        # Empty placeholder
                        frame_data.append(
                            go.Image(
                                z=np.zeros((10, 10, 3)),
                                colormodel="rgb",
                                x0=0,
                                y0=0,
                                dx=1,
                                dy=1,
                            )
                        )
            else:
                # Empty placeholders if condition not found
                for _ in channel_colormaps:
                    frame_data.append(
                        go.Image(
                            z=np.zeros((10, 10, 3)),
                            colormodel="rgb",
                            x0=0,
                            y0=0,
                            dx=1,
                            dy=1,
                        )
                    )

        # Create trace indices for updating the correct traces in each frame
        trace_indices = list(range(len(condition_keys) * len(channel_colormaps)))
        return go.Frame(data=frame_data, name=str(t), traces=trace_indices)

    # Create frames for the slider
    frames = [create_frame_for_timepoint(t) for t in all_timepoints]
    fig.frames = frames

    # Update layout
    fig.update_layout(
        title=f"Cell Images - Time: {t_initial}",
        height=plot_size_xy[1],
        width=plot_size_xy[0],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(t),
                        "method": "animate",
                    }
                    for t in all_timepoints
                ],
            }
        ],
    )

    # Update axes to hide ticks and labels
    for row in range(1, len(condition_keys) + 1):
        for col in range(1, len(channel_colormaps) + 1):
            fig.update_xaxes(
                showticklabels=False, showgrid=False, zeroline=False, row=row, col=col
            )
            fig.update_yaxes(
                showticklabels=False, showgrid=False, zeroline=False, row=row, col=col
            )

    return fig


def create_combined_visualization(
    image_cache,
    imagenet_df: pd.DataFrame,
    dynaclr_df: pd.DataFrame,
    highlight_tracks: dict,
    subplot_titles=[
        "Uninfected Phase",
        "Uninfected Viral Sensor",
        "Infected Phase",
        "Infected Viral Sensor",
    ],
    condition_keys=["uninfected_cache", "infected_cache"],
    channel_colormaps=["gray", "magma"],
    category_colors={1: "cornflowerblue", 2: "salmon"},
    highlight_colors={1: "blue", 2: "red"},
    category_labels={1: "Uninfected", 2: "Infected"},
    plot_size_xy=(1800, 600),
    title_location="inside",
):
    """
    Creates a combined visualization with cell images and PHATE embeddings with a shared time slider.
    All plots are arranged side by side in one row.

    Parameters
    ----------
    image_cache : dict
        Image cache dictionary with cell images
    imagenet_df : pandas.DataFrame
        DataFrame with ImageNet PHATE embeddings
    dynaclr_df : pandas.DataFrame
        DataFrame with DynaCLR PHATE embeddings
    highlight_tracks : dict
        Dictionary of tracks to highlight in PHATE plots
    subplot_titles : list
        Titles for the image subplots
    condition_keys : list
        Keys for conditions in image cache
    channel_colormaps : list
        Colormaps for image channels
    category_colors, highlight_colors, category_labels : dict
        Visual configuration for PHATE plots
    plot_size_xy : tuple
        Width and height of the plot
    title_location : str
        Location of subplot titles. Either "inside" (default) or "top"

    Returns
    -------
    plotly.graph_objects.Figure
        Combined interactive figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_timepoints_images = set()
    for condition_key in condition_keys:
        if (
            condition_key in image_cache
            and "images_by_timepoint" in image_cache[condition_key]
        ):
            all_timepoints_images.update(
                image_cache[condition_key]["images_by_timepoint"].keys()
            )

    all_timepoints_imagenet = set(imagenet_df["t"].unique())
    all_timepoints_dynaclr = set(dynaclr_df["t"].unique())

    all_timepoints = sorted(
        list(
            all_timepoints_images.intersection(
                all_timepoints_imagenet, all_timepoints_dynaclr
            )
        )
    )

    if not all_timepoints:
        print("No common timepoints found across all datasets")
        all_timepoints = sorted(
            list(
                all_timepoints_images.union(
                    all_timepoints_imagenet, all_timepoints_dynaclr
                )
            )
        )

    def create_phate_traces(
        df: pd.DataFrame, t: int, df_coordinates: list[str] = ["PHATE1", "PHATE2"]
    ):
        """Creates PHATE plot traces for a specific timepoint"""
        traces = []

        historical_df = df[df["t"] < t]
        if len(historical_df) > 0:
            traces.append(
                go.Scatter(
                    x=historical_df[df_coordinates[0]],
                    y=historical_df[df_coordinates[1]],
                    mode="markers",
                    marker=dict(color="lightgray", size=5, opacity=0.2),
                    name="Historical",
                    hoverinfo="none",
                    showlegend=False,
                )
            )
        else:
            traces.append(go.Scatter(x=[], y=[], mode="markers", showlegend=False))

        current_df = df[df["t"] == t]
        categories = sorted(df["infection"].unique())

        for category in categories:
            category_points = current_df[current_df["infection"] == category]
            if len(category_points) > 0:
                traces.append(
                    go.Scatter(
                        x=category_points[df_coordinates[0]],
                        y=category_points[df_coordinates[1]],
                        mode="markers",
                        marker=dict(
                            color=category_colors.get(category, "gray"),
                            size=8,
                            opacity=0.7,
                        ),
                        name=category_labels.get(category, f"Category {category}"),
                        hovertext=[
                            f"FOV: {row['fov_name']}, Track: {row['track_id']}, {category_labels.get(category, f'Category {category}')}"
                            for _, row in category_points.iterrows()
                        ],
                        hoverinfo="text",
                        showlegend=False,
                    )
                )
            else:
                traces.append(go.Scatter(x=[], y=[], mode="markers", showlegend=False))

        for category, track_list in highlight_tracks.items():
            for fov_name, track_id in track_list:
                track_data = df[
                    (df["fov_name"] == fov_name)
                    & (df["track_id"] == track_id)
                    & (df["t"] <= t)
                ].sort_values("t")

                if len(track_data) > 0:
                    color = highlight_colors.get(category, "gray")

                    traces.append(
                        go.Scatter(
                            x=track_data[df_coordinates[0]],
                            y=track_data[df_coordinates[1]],
                            mode="lines",
                            line=dict(color=color, width=2),
                            showlegend=False,
                        )
                    )

                    current_pos = track_data[track_data["t"] == t]
                    if len(current_pos) == 0:
                        latest_pos = track_data.iloc[-1:]
                        opacity = 0.5
                    else:
                        latest_pos = current_pos
                        opacity = 1.0

                    traces.append(
                        go.Scatter(
                            x=latest_pos[df_coordinates[0]],
                            y=latest_pos[df_coordinates[1]],
                            mode="markers",
                            marker=dict(
                                color=color,
                                size=15,
                                line=dict(color="black", width=1),
                                opacity=opacity,
                            ),
                            hovertext=[
                                f"FOV: {row['fov_name']}, Track: {row['track_id']}, t={row['t']}"
                                for _, row in latest_pos.iterrows()
                            ],
                            hoverinfo="text",
                            showlegend=False,
                        )
                    )

        return traces

    def get_phate_limits(df, df_coordinates=["PHATE1", "PHATE2"]):
        padding = 0.1
        x_min = df[df_coordinates[0]].min() - padding * (
            df[df_coordinates[0]].max() - df[df_coordinates[0]].min()
        )
        x_max = df[df_coordinates[0]].max() + padding * (
            df[df_coordinates[0]].max() - df[df_coordinates[0]].min()
        )
        y_min = df[df_coordinates[1]].min() - padding * (
            df[df_coordinates[1]].max() - df[df_coordinates[1]].min()
        )
        y_max = df[df_coordinates[1]].max() + padding * (
            df[df_coordinates[1]].max() - df[df_coordinates[1]].min()
        )

        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range > y_range:
            center = (y_max + y_min) / 2
            y_min = center - x_range / 2
            y_max = center + x_range / 2
        else:
            center = (x_max + x_min) / 2
            x_min = center - y_range / 2
            x_max = center + y_range / 2

        return x_min, x_max, y_min, y_max

    imagenet_limits = get_phate_limits(imagenet_df)
    dynaclr_limits = get_phate_limits(dynaclr_df)

    t_initial = all_timepoints[0]

    main_fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.33, 0.33, 0.33],
        subplot_titles=["", "ImageNet PHATE", "DynaCLR PHATE"],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
    )

    def create_cell_image_traces(t):
        traces = []
        from matplotlib import cm

        for row_idx, condition_key in enumerate(condition_keys):
            if (
                condition_key in image_cache
                and t in image_cache[condition_key]["images_by_timepoint"]
            ):
                img = image_cache[condition_key]["images_by_timepoint"][t]

                for col_idx, colormap in enumerate(channel_colormaps):
                    if col_idx < img.shape[0]:  # Check if channel exists
                        img_data = img[col_idx, 0]
                        img_data = rescale_intensity(img_data, out_range=(0, 1))

                        if colormap == "gray":
                            rgb_img = np.stack([img_data] * 3, axis=-1)
                            rgb_img = (rgb_img * 255).astype(np.uint8)
                        else:
                            cmap = cm.get_cmap(colormap)
                            colored_img = cmap(img_data)
                            rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

                        x_pos = col_idx * 0.5
                        y_pos = 1.0 - row_idx * 0.5

                        x_coords = np.linspace(x_pos, x_pos + 0.45, rgb_img.shape[1])
                        y_coords = np.linspace(y_pos - 0.45, y_pos, rgb_img.shape[0])

                        traces.append(
                            go.Image(
                                z=rgb_img,
                                x0=x_coords[0],
                                y0=y_coords[0],
                                dx=(x_coords[-1] - x_coords[0]) / rgb_img.shape[1],
                                dy=(y_coords[-1] - y_coords[0]) / rgb_img.shape[0],
                                colormodel="rgb",
                                name=subplot_titles[
                                    row_idx * len(channel_colormaps) + col_idx
                                ],
                            )
                        )
                    else:
                        warnings.warn(
                            f"Channel {col_idx} does not exist in image cache for timepoint {t}"
                        )

        return traces

    for trace in create_cell_image_traces(t_initial):
        main_fig.add_trace(trace, row=1, col=1)

    for trace in create_phate_traces(imagenet_df, t_initial, ["PHATE1", "PHATE2"]):
        main_fig.add_trace(trace, row=1, col=2)

    for trace in create_phate_traces(dynaclr_df, t_initial, ["PHATE1", "PHATE2"]):
        main_fig.add_trace(trace, row=1, col=3)

    for i, title in enumerate(subplot_titles):
        row = i // 2
        col = i % 2

        if title_location == "top":
            x_pos = col * 0.5 + 0.22
            y_pos = 1 - row * 0.5
            yanchor = "bottom"
            font_color = "black"
        else:
            x_pos = col * 0.5 + 0.22
            y_pos = 1 - row * 0.5 - 0.05
            yanchor = "top"
            font_color = "white"

        main_fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=title,
            showarrow=False,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor=yanchor,
            font=dict(size=10, color=font_color),
            row=1,
            col=1,
        )

    main_fig.update_xaxes(
        range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=1, col=1
    )
    main_fig.update_yaxes(
        range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=1, col=1
    )

    main_fig.update_xaxes(title="PHATE1", range=imagenet_limits[:2], row=1, col=2)
    main_fig.update_yaxes(
        title="PHATE2",
        range=imagenet_limits[2:],
        scaleanchor="x2",
        scaleratio=1,
        row=1,
        col=2,
    )
    main_fig.update_xaxes(title="PHATE1", range=dynaclr_limits[:2], row=1, col=3)
    main_fig.update_yaxes(
        title="PHATE2",
        range=dynaclr_limits[2:],
        scaleanchor="x3",
        scaleratio=1,
        row=1,
        col=3,
    )

    main_fig.update_layout(
        title="Cell Images and PHATE Embeddings",
        width=plot_size_xy[0],
        height=plot_size_xy[1],
        sliders=[
            {
                "active": 1,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "fromcurrent": False,
                            },
                        ],
                        "label": str(t),
                        "method": "animate",
                    }
                    for t in all_timepoints
                ],
            }
        ],
    )

    frames = []
    for t in all_timepoints:
        frame_data = []

        frame_data.extend(create_cell_image_traces(t))

        frame_data.extend(create_phate_traces(imagenet_df, t, ["PHATE1", "PHATE2"]))
        frame_data.extend(create_phate_traces(dynaclr_df, t, ["PHATE1", "PHATE2"]))

        frames.append(go.Frame(data=frame_data, name=str(t)))

    main_fig.frames = frames

    main_fig.update_layout(
        transition={"duration": 0},
        updatemenus=[],  # Remove any animation buttons
    )

    return main_fig
