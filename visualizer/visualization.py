"""
PHATE embedding visualization with flexible coloring modes.

This module provides functions for creating interactive PHATE scatter plots
with support for multiple coloring modes, trajectory visualization, and
timepoint highlighting.

Functions
---------
create_phate_figure : Create interactive PHATE scatter plot with flexible coloring
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import INFECTION_COLORS


def create_phate_figure(
    df: pd.DataFrame,
    color_by: str = "annotation",
    selected_values: Optional[list] = None,
    selected_tracks: Optional[list[str]] = None,
    show_trajectories: bool = False,
    highlight_timepoint: Optional[int] = None,
) -> go.Figure:
    """
    Create interactive PHATE scatter plot with flexible coloring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with PHATE coordinates and metadata.
    color_by : str
        Coloring mode: "annotation", "time", "track_id", or "dataset".
    selected_values : list, optional
        Values to display (for categorical coloring).
    selected_tracks : list[str], optional
        Track keys to highlight.
    show_trajectories : bool, optional
        If True, draw lines connecting points in temporal order for selected tracks.
    highlight_timepoint : int, optional
        Specific timepoint to highlight for selected tracks.

    Returns
    -------
    fig : go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    DATASET_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#f1c40f", "#e74c3c"]

    if color_by == "annotation" and "annotation" in df.columns:
        if selected_values is None:
            selected_values = df["annotation"].unique().tolist()

        filtered_df = df[df["annotation"].isin(selected_values)]

        for status in selected_values:
            status_df = filtered_df[filtered_df["annotation"] == status]

            if len(status_df) == 0:
                continue

            if selected_tracks:
                highlighted = status_df[status_df["track_key"].isin(selected_tracks)]
                background = status_df[~status_df["track_key"].isin(selected_tracks)]

                if len(background) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=background["PHATE1"],
                            y=background["PHATE2"],
                            mode="markers",
                            name=f"{status} (background)",
                            marker=dict(
                                color=INFECTION_COLORS.get(status, "#95a5a6"),
                                size=4,
                                opacity=0.3,
                            ),
                            customdata=background[
                                ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                            ].values,
                            hovertemplate=(
                                "<b>Dataset:</b> %{customdata[4]}<br>"
                                "<b>Track:</b> %{customdata[0]}<br>"
                                "<b>Time:</b> %{customdata[1]}<br>"
                                "<b>FOV:</b> %{customdata[2]}<br>"
                                "<b>Status:</b> " + status + "<br>"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                if len(highlighted) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=highlighted["PHATE1"],
                            y=highlighted["PHATE2"],
                            mode="markers",
                            name=f"{status} (selected)",
                            marker=dict(
                                color=INFECTION_COLORS.get(status, "#95a5a6"),
                                size=8,
                                opacity=0.9,
                                line=dict(width=1, color="white"),
                            ),
                            customdata=highlighted[
                                ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                            ].values,
                            hovertemplate=(
                                "<b>Dataset:</b> %{customdata[4]}<br>"
                                "<b>Track:</b> %{customdata[0]}<br>"
                                "<b>Time:</b> %{customdata[1]}<br>"
                                "<b>FOV:</b> %{customdata[2]}<br>"
                                "<b>Status:</b> " + status + "<br>"
                                "<extra></extra>"
                            ),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=status_df["PHATE1"],
                        y=status_df["PHATE2"],
                        mode="markers",
                        name=status.replace("_", " ").title(),
                        marker=dict(
                            color=INFECTION_COLORS.get(status, "#95a5a6"),
                            size=5,
                            opacity=0.6,
                        ),
                        customdata=status_df[
                            ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Dataset:</b> %{customdata[4]}<br>"
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<b>Status:</b> " + status + "<br>"
                            "<extra></extra>"
                        ),
                    )
                )

    elif color_by == "time":
        filtered_df = df.copy()

        if selected_tracks:
            highlighted = filtered_df[filtered_df["track_key"].isin(selected_tracks)]
            background = filtered_df[~filtered_df["track_key"].isin(selected_tracks)]

            if len(background) > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=background["PHATE1"],
                        y=background["PHATE2"],
                        mode="markers",
                        name="background",
                        marker=dict(
                            color=background["t"],
                            colorscale="Viridis",
                            size=4,
                            opacity=0.3,
                            showscale=False,
                        ),
                        customdata=background[
                            ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Dataset:</b> %{customdata[4]}<br>"
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

            if len(highlighted) > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=highlighted["PHATE1"],
                        y=highlighted["PHATE2"],
                        mode="markers",
                        name="time (selected tracks)",
                        marker=dict(
                            color=highlighted["t"],
                            colorscale="Viridis",
                            size=8,
                            opacity=0.9,
                            showscale=True,
                            colorbar=dict(title="Time"),
                            line=dict(width=1, color="white"),
                        ),
                        customdata=highlighted[
                            ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Dataset:</b> %{customdata[4]}<br>"
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<extra></extra>"
                        ),
                    )
                )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=filtered_df["PHATE1"],
                    y=filtered_df["PHATE2"],
                    mode="markers",
                    name="time",
                    marker=dict(
                        color=filtered_df["t"],
                        colorscale="Viridis",
                        size=5,
                        opacity=0.6,
                        showscale=True,
                        colorbar=dict(title="Time"),
                    ),
                    customdata=filtered_df[
                        ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                    ].values,
                    hovertemplate=(
                        "<b>Dataset:</b> %{customdata[4]}<br>"
                        "<b>Track:</b> %{customdata[0]}<br>"
                        "<b>Time:</b> %{customdata[1]}<br>"
                        "<b>FOV:</b> %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    elif color_by == "track_id":
        filtered_df = df.copy()

        if selected_tracks:
            filtered_df = filtered_df[filtered_df["track_key"].isin(selected_tracks)]

            colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
            track_colors = {
                track: colors[i % len(colors)]
                for i, track in enumerate(selected_tracks)
            }

            for track_key in selected_tracks:
                track_df = filtered_df[filtered_df["track_key"] == track_key]

                if len(track_df) == 0:
                    continue

                fig.add_trace(
                    go.Scattergl(
                        x=track_df["PHATE1"],
                        y=track_df["PHATE2"],
                        mode="markers",
                        name=track_key,
                        marker=dict(
                            color=track_colors[track_key],
                            size=8,
                            opacity=0.7,
                            line=dict(width=1, color="white"),
                        ),
                        customdata=track_df[
                            ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Dataset:</b> %{customdata[4]}<br>"
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<extra></extra>"
                        ),
                    )
                )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=filtered_df["PHATE1"],
                    y=filtered_df["PHATE2"],
                    mode="markers",
                    name="all tracks",
                    marker=dict(
                        color="#95a5a6",
                        size=5,
                        opacity=0.4,
                    ),
                    customdata=filtered_df[
                        ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                    ].values,
                    hovertemplate=(
                        "<b>Dataset:</b> %{customdata[4]}<br>"
                        "<b>Track:</b> %{customdata[0]}<br>"
                        "<b>Time:</b> %{customdata[1]}<br>"
                        "<b>FOV:</b> %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    elif color_by == "dataset" and "dataset_id" in df.columns:
        filtered_df = df.copy()
        unique_datasets = sorted(filtered_df["dataset_id"].unique())

        dataset_color_map = {
            dataset: DATASET_COLORS[i % len(DATASET_COLORS)]
            for i, dataset in enumerate(unique_datasets)
        }

        if selected_tracks:
            highlighted = filtered_df[filtered_df["track_key"].isin(selected_tracks)]
            background = filtered_df[~filtered_df["track_key"].isin(selected_tracks)]

            for dataset_id in unique_datasets:
                dataset_bg = background[background["dataset_id"] == dataset_id]
                if len(dataset_bg) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=dataset_bg["PHATE1"],
                            y=dataset_bg["PHATE2"],
                            mode="markers",
                            name=f"{dataset_id} (background)",
                            marker=dict(
                                color=dataset_color_map[dataset_id],
                                size=4,
                                opacity=0.3,
                            ),
                            customdata=dataset_bg[
                                ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                            ].values,
                            hovertemplate=(
                                "<b>Dataset:</b> %{customdata[4]}<br>"
                                "<b>Track:</b> %{customdata[0]}<br>"
                                "<b>Time:</b> %{customdata[1]}<br>"
                                "<b>FOV:</b> %{customdata[2]}<br>"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

            for dataset_id in unique_datasets:
                dataset_hl = highlighted[highlighted["dataset_id"] == dataset_id]
                if len(dataset_hl) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=dataset_hl["PHATE1"],
                            y=dataset_hl["PHATE2"],
                            mode="markers",
                            name=f"{dataset_id} (selected)",
                            marker=dict(
                                color=dataset_color_map[dataset_id],
                                size=8,
                                opacity=0.9,
                                line=dict(width=1, color="white"),
                            ),
                            customdata=dataset_hl[
                                ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                            ].values,
                            hovertemplate=(
                                "<b>Dataset:</b> %{customdata[4]}<br>"
                                "<b>Track:</b> %{customdata[0]}<br>"
                                "<b>Time:</b> %{customdata[1]}<br>"
                                "<b>FOV:</b> %{customdata[2]}<br>"
                                "<extra></extra>"
                            ),
                        )
                    )
        else:
            for dataset_id in unique_datasets:
                dataset_df = filtered_df[filtered_df["dataset_id"] == dataset_id]
                if len(dataset_df) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=dataset_df["PHATE1"],
                            y=dataset_df["PHATE2"],
                            mode="markers",
                            name=dataset_id,
                            marker=dict(
                                color=dataset_color_map[dataset_id],
                                size=5,
                                opacity=0.6,
                            ),
                            customdata=dataset_df[
                                ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                            ].values,
                            hovertemplate=(
                                "<b>Dataset:</b> %{customdata[4]}<br>"
                                "<b>Track:</b> %{customdata[0]}<br>"
                                "<b>Time:</b> %{customdata[1]}<br>"
                                "<b>FOV:</b> %{customdata[2]}<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

    if show_trajectories and selected_tracks:
        for track_key in selected_tracks:
            track_df = df[df["track_key"] == track_key].sort_values("t")

            if len(track_df) < 2:
                continue

            if color_by == "annotation" and "annotation" in track_df.columns:
                annotation_value = track_df["annotation"].iloc[0]
                color = INFECTION_COLORS.get(annotation_value, "#95a5a6")
            else:
                color = "#666666"

            fig.add_trace(
                go.Scattergl(
                    x=track_df["PHATE1"],
                    y=track_df["PHATE2"],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=2,
                        dash="solid",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            for i in range(0, len(track_df) - 1, 3):
                row_current = track_df.iloc[i]
                row_next = track_df.iloc[i + 1]

                row_next["PHATE1"] - row_current["PHATE1"]
                row_next["PHATE2"] - row_current["PHATE2"]

                fig.add_annotation(
                    x=row_next["PHATE1"],
                    y=row_next["PHATE2"],
                    ax=row_current["PHATE1"],
                    ay=row_current["PHATE2"],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    opacity=0.6,
                )

    if highlight_timepoint is not None and selected_tracks:
        highlight_points = []
        for track_key in selected_tracks:
            track_df = df[df["track_key"] == track_key]
            timepoint_data = track_df[track_df["t"] == highlight_timepoint]

            if len(timepoint_data) > 0:
                highlight_points.append(timepoint_data)

        if highlight_points:
            highlight_df = pd.concat(highlight_points, ignore_index=True)

            fig.add_trace(
                go.Scattergl(
                    x=highlight_df["PHATE1"],
                    y=highlight_df["PHATE2"],
                    mode="markers",
                    name=f"t={highlight_timepoint}",
                    marker=dict(
                        size=15,
                        color="yellow",
                        symbol="star",
                        line=dict(width=2, color="black"),
                    ),
                    customdata=highlight_df[
                        ["track_key", "t", "fov_name", "track_id", "dataset_id"]
                    ].values,
                    hovertemplate=(
                        "<b>⭐ HIGHLIGHTED ⭐</b><br>"
                        "<b>Dataset:</b> %{customdata[4]}<br>"
                        "<b>Track:</b> %{customdata[0]}<br>"
                        "<b>Time:</b> %{customdata[1]}<br>"
                        "<b>FOV:</b> %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="PHATE Embedding - Click points to select tracks",
        xaxis_title="PHATE1",
        yaxis_title="PHATE2",
        hovermode="closest",
        template="plotly_white",
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig
