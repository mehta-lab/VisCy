"""Figure creation utilities."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd


class FigureCreator:
    """Utility class for creating plotly figures."""

    @staticmethod
    def create_track_colored_figure(
        features_df: pd.DataFrame,
        filtered_features_df: pd.DataFrame,
        clusters: List[Dict],
        cluster_points: set,
        show_arrows: bool = False,
        x_axis: str = "PCA1",
        y_axis: str = "PCA2",
        trajectory_mode: str = "x",
        selection_mode: str = "region",
        highlight_point: Optional[Tuple] = None,
    ) -> go.Figure:
        """Create a scatter plot with track-based coloring.

        Parameters
        ----------
        features_df : pd.DataFrame
            The full features dataframe.
        filtered_features_df : pd.DataFrame
            The filtered features dataframe.
        clusters : List[Dict]
            List of clusters.
        cluster_points : set
            Set of points in clusters.
        show_arrows : bool, optional
            Whether to show arrows, by default False
        x_axis : str, optional
            X-axis label, by default "PCA1"
        y_axis : str, optional
            Y-axis label, by default "PCA2"
        trajectory_mode : str, optional
            Trajectory mode, by default "x"
        selection_mode : str, optional
            Selection mode, by default "region"
        highlight_point : Optional[Tuple], optional
            Point to highlight, by default None

        Returns
        -------
        go.Figure
            The scatter plot figure.
        """
        fig = go.Figure()

        # Create background scatter plot with all points
        background_df = features_df[~features_df.index.isin(filtered_features_df.index)]
        if not background_df.empty:
            fig.add_trace(
                FigureCreator._create_scatter_trace(
                    background_df,
                    x_axis,
                    y_axis,
                    "lightgrey",
                    "Background",
                    opacity=0.1,
                    show_in_legend=False,
                )
            )

        # Create cluster color mapping
        cluster_colors = [
            f"rgb{tuple(int(x*255) for x in plt.cm.Set2(i % 8)[:3])}"
            for i in range(len(clusters))
        ]
        point_to_cluster = {}
        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster:
                point_key = (point["fov_name"], point["track_id"], point["t"])
                point_to_cluster[point_key] = cluster_idx

        # Process each track
        track_groups = filtered_features_df.groupby(["fov_name", "track_id"])
        for (fov, track_id), track_data in track_groups:
            # Determine base opacity for this track
            base_opacity = (
                0.3
                if highlight_point
                and (fov, track_id) != (highlight_point[0], highlight_point[1])
                else 1.0
            )

            # Determine if any points in this track are in clusters
            track_points = [(fov, track_id, t) for t in track_data["t"]]
            clustered_points = [p for p in track_points if p in cluster_points]

            # Split track data into clustered and non-clustered points
            if clustered_points:
                # Add clustered points with their respective colors
                for cluster_idx in set(
                    point_to_cluster[p]
                    for p in clustered_points
                    if p in point_to_cluster
                ):
                    cluster_mask = track_data.apply(
                        lambda row: (fov, track_id, row["t"]) in point_to_cluster
                        and point_to_cluster[(fov, track_id, row["t"])] == cluster_idx,
                        axis=1,
                    )
                    cluster_data = track_data[cluster_mask]
                    if not cluster_data.empty:
                        opacity = base_opacity
                        if highlight_point and (fov, track_id) == (
                            highlight_point[0],
                            highlight_point[1],
                        ):
                            opacity = (
                                1.0
                                if cluster_data["t"].iloc[0] == highlight_point[2]
                                else 0.3
                            )
                        fig.add_trace(
                            FigureCreator._create_scatter_trace(
                                cluster_data,
                                x_axis,
                                y_axis,
                                cluster_colors[cluster_idx],
                                f"Cluster {cluster_idx + 1}",
                                opacity=opacity,
                            )
                        )

                # Add non-clustered points
                non_cluster_mask = ~track_data.apply(
                    lambda row: (fov, track_id, row["t"]) in point_to_cluster, axis=1
                )
                non_cluster_data = track_data[non_cluster_mask]
            else:
                non_cluster_data = track_data

            if not non_cluster_data.empty:
                # Generate a random color for this track
                track_color = (
                    f"rgb{tuple(int(x*255) for x in plt.cm.tab20(track_id % 20)[:3])}"
                )

                if highlight_point and (fov, track_id) == (
                    highlight_point[0],
                    highlight_point[1],
                ):
                    # Split the data into highlighted and non-highlighted points
                    highlight_mask = non_cluster_data["t"] == highlight_point[2]
                    highlighted_data = non_cluster_data[highlight_mask]
                    other_data = non_cluster_data[~highlight_mask]

                    if not other_data.empty:
                        fig.add_trace(
                            FigureCreator._create_scatter_trace(
                                other_data,
                                x_axis,
                                y_axis,
                                track_color,
                                f"Track {track_id}",
                                opacity=0.3,
                                show_in_legend=True,
                            )
                        )

                    if not highlighted_data.empty:
                        fig.add_trace(
                            FigureCreator._create_scatter_trace(
                                highlighted_data,
                                x_axis,
                                y_axis,
                                track_color,
                                f"Track {track_id} (Selected)",
                                opacity=1.0,
                                show_in_legend=False,
                            )
                        )
                else:
                    fig.add_trace(
                        FigureCreator._create_scatter_trace(
                            non_cluster_data,
                            x_axis,
                            y_axis,
                            track_color,
                            f"Track {track_id}",
                            opacity=base_opacity,
                            show_in_legend=True,
                        )
                    )

            # Add arrows if requested
            if show_arrows:
                FigureCreator._add_arrows(
                    fig, track_data, x_axis, y_axis, trajectory_mode
                )

        # Update layout
        FigureCreator._update_figure_layout(fig, x_axis, y_axis)

        return fig

    @staticmethod
    def create_time_colored_figure(
        features_df: pd.DataFrame,
        filtered_features_df: pd.DataFrame,
        show_arrows: bool = False,
        x_axis: str = "PCA1",
        y_axis: str = "PCA2",
        trajectory_mode: str = "x",
        selection_mode: str = "region",
        highlight_point: Optional[Tuple] = None,
    ) -> go.Figure:
        """Create scatter plot with time-based coloring.

        Parameters
        ----------
        features_df : pd.DataFrame
            The full features dataframe.
        filtered_features_df : pd.DataFrame
            The filtered features dataframe.
        show_arrows : bool, optional
            Whether to show arrows, by default False
        x_axis : str, optional
            X-axis label, by default "PCA1"
        y_axis : str, optional
            Y-axis label, by default "PCA2"
        trajectory_mode : str, optional
            Trajectory mode, by default "x"
        selection_mode : str, optional
            Selection mode, by default "region"
        highlight_point : Optional[Tuple], optional
            Point to highlight, by default None

        Returns
        -------
        go.Figure
            The scatter plot figure.
        """
        fig = go.Figure()

        # Create background scatter plot with all points
        background_df = features_df[~features_df.index.isin(filtered_features_df.index)]
        if not background_df.empty:
            fig.add_trace(
                FigureCreator._create_scatter_trace(
                    background_df,
                    x_axis,
                    y_axis,
                    "lightgrey",
                    "Background",
                    opacity=0.1,
                    show_in_legend=False,
                )
            )

        # Create color scale for time points
        min_time = filtered_features_df["t"].min()
        max_time = filtered_features_df["t"].max()
        norm = plt.Normalize(min_time, max_time)
        cmap = plt.cm.viridis
        colors = [
            f"rgb{tuple(int(x*255) for x in cmap(norm(t))[:3])}"
            for t in filtered_features_df["t"]
        ]

        # Process each track
        track_groups = filtered_features_df.groupby(["fov_name", "track_id"])
        for (fov, track_id), track_data in track_groups:
            # Determine base opacity for this track
            base_opacity = 0.3 if highlight_point else 1.0

            # Split the data into highlighted and non-highlighted points
            if highlight_point:
                if (fov, track_id) == (highlight_point[0], highlight_point[1]):
                    highlight_mask = track_data["t"] == highlight_point[2]
                    highlighted_data = track_data[highlight_mask]
                    other_data = track_data[~highlight_mask]

                    if not other_data.empty:
                        track_colors = [
                            f"rgb{tuple(int(x*255) for x in cmap(norm(t))[:3])}"
                            for t in other_data["t"]
                        ]
                        fig.add_trace(
                            FigureCreator._create_scatter_trace(
                                other_data,
                                x_axis,
                                y_axis,
                                track_colors,
                                f"Track {track_id}",
                                opacity=0.3,
                                show_in_legend=True,
                            )
                        )

                    if not highlighted_data.empty:
                        track_colors = [
                            f"rgb{tuple(int(x*255) for x in cmap(norm(t))[:3])}"
                            for t in highlighted_data["t"]
                        ]
                        fig.add_trace(
                            FigureCreator._create_scatter_trace(
                                highlighted_data,
                                x_axis,
                                y_axis,
                                track_colors,
                                f"Track {track_id} (Selected)",
                                opacity=1.0,
                                show_in_legend=False,
                            )
                        )
                else:
                    # This is a different track, show all points with reduced opacity
                    track_colors = [
                        f"rgb{tuple(int(x*255) for x in cmap(norm(t))[:3])}"
                        for t in track_data["t"]
                    ]
                    fig.add_trace(
                        FigureCreator._create_scatter_trace(
                            track_data,
                            x_axis,
                            y_axis,
                            track_colors,
                            f"Track {track_id}",
                            opacity=0.3,
                            show_in_legend=True,
                        )
                    )
            else:
                # No point is highlighted, show all points with full opacity
                track_colors = [
                    f"rgb{tuple(int(x*255) for x in cmap(norm(t))[:3])}"
                    for t in track_data["t"]
                ]
                fig.add_trace(
                    FigureCreator._create_scatter_trace(
                        track_data,
                        x_axis,
                        y_axis,
                        track_colors,
                        f"Track {track_id}",
                        opacity=1.0,
                        show_in_legend=True,
                    )
                )

            # Add arrows if requested
            if show_arrows:
                FigureCreator._add_arrows(
                    fig, track_data, x_axis, y_axis, trajectory_mode
                )

        # Update layout
        FigureCreator._update_figure_layout(fig, x_axis, y_axis)

        return fig

    @staticmethod
    def _update_figure_layout(
        fig: go.Figure, x_axis: str = "PCA1", y_axis: str = "PCA2"
    ):
        """Update the layout for a figure.

        Parameters
        ----------
        fig : go.Figure
            The figure to update.
        x_axis : str, optional
            X-axis label, by default "PCA1"
        y_axis : str, optional
            Y-axis label, by default "PCA2"
        """
        fig.update_layout(
            plot_bgcolor="white",
            title="PCA visualization of Selected Tracks",
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            uirevision=True,
            hovermode="closest",
            showlegend=True,
            dragmode="lasso",
            clickmode="event+select",
            selectionrevision=True,
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                title="Tracks",
                bordercolor="Black",
                borderwidth=1,
            ),
            margin=dict(l=50, r=150, t=50, b=50),
            autosize=True,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    @staticmethod
    def _create_scatter_trace(
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        color: Union[str, List[str]],
        name: str,
        opacity: float = 1.0,
        show_in_legend: bool = True,
    ) -> go.Scattergl:
        """Create a scatter plot trace using WebGL for better performance.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data.
        x_axis : str
            The column name for x-axis data.
        y_axis : str
            The column name for y-axis data.
        color : Union[str, List[str]]
            The color(s) for the points.
        name : str
            The name of the trace.
        opacity : float, optional
            The opacity of the points, by default 1.0
        show_in_legend : bool, optional
            Whether to show the trace in the legend, by default True

        Returns
        -------
        go.Scattergl
            The WebGL-accelerated scatter plot trace.
        """
        return go.Scattergl(
            x=df[x_axis],
            y=df[y_axis],
            mode="markers",
            marker=dict(
                color=color,
                size=8,
                opacity=opacity,
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            name=name,
            text=[
                f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                for track_id, t, fov in zip(df["track_id"], df["t"], df["fov_name"])
            ],
            hoverinfo="text",
            showlegend=show_in_legend,
            selectedpoints=None,
            selected=dict(marker=dict(size=12, color="rgba(0, 123, 255, 1.0)")),
            unselected=dict(marker=dict(opacity=0.3)),
        )

    @staticmethod
    def _add_arrows(
        fig: go.Figure,
        track_data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        trajectory_mode: str,
    ):
        """Add arrows to the figure based on trajectory mode."""
        if trajectory_mode == "x":
            x_coords = track_data[x_axis].values
            y_coords = track_data[y_axis].values

            # Add dashed lines for the trajectory with reduced opacity
            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(
                        color="rgba(128, 128, 128, 0.3)",
                        width=1,
                        dash="dot",
                        opacity=0.3,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add arrows at regular intervals with reduced opacity
            arrow_interval = max(1, len(track_data) // 3)
            for i in range(0, len(track_data) - 1, arrow_interval):
                dx = x_coords[i + 1] - x_coords[i]
                dy = y_coords[i + 1] - y_coords[i]

                if dx * dx + dy * dy > 1e-6:
                    fig.add_annotation(
                        x=x_coords[i + 1],
                        y=y_coords[i + 1],
                        ax=x_coords[i],
                        ay=y_coords[i],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor="rgba(128, 128, 128, 0.3)",
                        opacity=0.3,
                    )
        else:
            x_coords = track_data[x_axis].values
            y_coords = track_data[y_axis].values

            # Add dashed lines for the trajectory with reduced opacity
            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(
                        color="rgba(128, 128, 128, 0.3)",
                        width=1,
                        dash="dot",
                        opacity=0.3,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add arrows at regular intervals with reduced opacity
            arrow_interval = max(1, len(track_data) // 3)
            for i in range(0, len(track_data) - 1, arrow_interval):
                dx = x_coords[i + 1] - x_coords[i]
                dy = y_coords[i + 1] - y_coords[i]

                if dx * dx + dy * dy > 1e-6:
                    fig.add_annotation(
                        x=x_coords[i + 1],
                        y=y_coords[i + 1],
                        ax=x_coords[i],
                        ay=y_coords[i],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor="rgba(128, 128, 128, 0.3)",
                        opacity=0.3,
                    )
