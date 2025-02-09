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
                    selectable=False,  # Make background points non-selectable
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
                                selectable=True,  # Make cluster points selectable
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
                                selectable=True,  # Make track points selectable
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
                                selectable=True,  # Make track points selectable
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
                            selectable=True,  # Make track points selectable
                        )
                    )

            # Add arrows if requested
            if show_arrows:
                FigureCreator._add_arrows(
                    fig, track_data, x_axis, y_axis, "x", track_color
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
        highlight_point: Optional[Tuple] = None,
    ) -> go.Figure:
        """Create scatter plot with time-based coloring."""
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
                    selectable=False,
                )
            )

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

            if highlight_point and (fov, track_id) == (
                highlight_point[0],
                highlight_point[1],
            ):
                # Split the data into highlighted and non-highlighted points
                highlight_mask = track_data["t"] == highlight_point[2]
                highlighted_data = track_data[highlight_mask]
                other_data = track_data[~highlight_mask]

                if not other_data.empty:
                    fig.add_trace(
                        FigureCreator._create_time_colored_trace(
                            other_data,
                            x_axis,
                            y_axis,
                            f"Track {track_id}",
                            opacity=0.3,
                            show_in_legend=True,
                        )
                    )

                if not highlighted_data.empty:
                    fig.add_trace(
                        FigureCreator._create_time_colored_trace(
                            highlighted_data,
                            x_axis,
                            y_axis,
                            f"Track {track_id} (Selected)",
                            opacity=1.0,
                            show_in_legend=False,
                        )
                    )
            else:
                fig.add_trace(
                    FigureCreator._create_time_colored_trace(
                        track_data,
                        x_axis,
                        y_axis,
                        f"Track {track_id}",
                        opacity=base_opacity,
                        show_in_legend=True,
                    )
                )

            # Add arrows if requested
            if show_arrows:
                # Get the middle time point color for arrows
                mid_t = track_data["t"].iloc[len(track_data) // 2]
                norm_t = (mid_t - filtered_features_df["t"].min()) / (
                    filtered_features_df["t"].max() - filtered_features_df["t"].min()
                )
                track_color = (
                    f"rgb{tuple(int(x*255) for x in plt.cm.viridis(norm_t)[:3])}"
                )
                FigureCreator._add_arrows(
                    fig, track_data, x_axis, y_axis, "x", track_color
                )

        # Update layout
        FigureCreator._update_figure_layout(fig, x_axis, y_axis)
        return fig

    @staticmethod
    def _add_arrows(
        fig: go.Figure,
        track_data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        trajectory_mode: str,
        track_color: Optional[str] = None,
    ):
        """Add arrows to the figure to show trajectory direction.

        Parameters
        ----------
        fig : go.Figure
            The figure to add arrows to.
        track_data : pd.DataFrame
            The track data.
        x_axis : str
            X-axis label.
        y_axis : str
            Y-axis label.
        trajectory_mode : str
            Trajectory mode.
        track_color : Optional[str]
            Color to use for the arrows, matching the track color.
        """
        # Sort data by time
        track_data = track_data.sort_values("t")

        # Get coordinates
        x = track_data[x_axis].values
        y = track_data[y_axis].values

        # Calculate arrow vectors
        dx = np.diff(x)
        dy = np.diff(y)

        # Select evenly spaced indices for max 5 arrows
        n_points = len(dx)
        if n_points <= 5:
            indices = range(n_points)
        else:
            indices = np.linspace(0, n_points - 1, 5, dtype=int)

        # Add arrows as scatter traces
        for i in indices:
            # Calculate arrow head coordinates
            arrow_x = [x[i], x[i] + dx[i]]
            arrow_y = [y[i], y[i] + dy[i]]

            # Use track color with some transparency, or default to gray if no color provided
            arrow_color = track_color if track_color else "rgba(100,100,100,0.8)"
            if track_color and track_color.startswith("rgb"):
                # Convert rgb to rgba with 0.8 opacity
                arrow_color = track_color.replace("rgb", "rgba").replace(")", ",0.8)")

            # Add arrow trace
            fig.add_trace(
                go.Scattergl(
                    x=arrow_x,
                    y=arrow_y,
                    mode="lines",
                    line=dict(
                        color=arrow_color,
                        width=2.5,
                        dash="solid",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    @staticmethod
    def _create_scatter_trace(
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        color: Union[str, List[str]],
        name: str,
        opacity: float = 1.0,
        show_in_legend: bool = True,
        selectable: bool = True,
    ) -> go.Scattergl:
        """Create a scatter trace."""
        hover_text = [
            f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
            for track_id, t, fov in zip(df["track_id"], df["t"], df["fov_name"])
        ]

        # Handle both single color and color list cases
        marker_dict = {
            "size": 12,
            "opacity": opacity,
            "line": dict(width=1, color="rgba(50,50,50,0.2)"),
        }

        # If color is a string, use it directly
        if isinstance(color, str):
            marker_dict["color"] = color
            selected_color = color
            unselected_color = color
        # If color is a list, use it for the base color and use a neutral color for selected/unselected
        else:
            marker_dict["color"] = color
            # Use a neutral color for selection states when using color arrays
            selected_color = "rgba(100,100,100,1.0)"
            unselected_color = "rgba(100,100,100,0.3)"

        scatter_args = {
            "x": df[x_axis],
            "y": df[y_axis],
            "mode": "markers",
            "marker": marker_dict,
            "name": name,
            "text": hover_text,
            "hoverinfo": "text",
            "showlegend": show_in_legend,
            "selectedpoints": [],
        }

        # Only add selection styling if the point is selectable
        if selectable:
            scatter_args.update(
                {
                    "selected": dict(
                        marker=dict(
                            color=selected_color,
                            size=12,
                            opacity=1.0,
                        )
                    ),
                    "unselected": dict(
                        marker=dict(
                            color=unselected_color,
                            opacity=opacity,
                        )
                    ),
                }
            )

        return go.Scattergl(**scatter_args)

    @staticmethod
    def _create_time_colored_trace(
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        name: str,
        opacity: float = 1.0,
        show_in_legend: bool = True,
    ) -> go.Scattergl:
        """Create a time-colored scatter trace."""
        hover_text = [
            f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
            for track_id, t, fov in zip(df["track_id"], df["t"], df["fov_name"])
        ]

        marker_dict = {
            "size": 12,
            "opacity": opacity,
            "line": dict(width=1, color="rgba(50,50,50,0.2)"),
            "color": df["t"],
            "colorscale": "Viridis",
            "showscale": False,
        }

        scatter_args = {
            "x": df[x_axis],
            "y": df[y_axis],
            "mode": "markers",
            "marker": marker_dict,
            "name": name,
            "text": hover_text,
            "hoverinfo": "text",
            "showlegend": show_in_legend,
            "selectedpoints": None,  # Allow only single point selection
            "selected": dict(
                marker=dict(
                    size=14,
                    opacity=1.0,
                    color="#007bff",  # Use blue color for selected points
                )
            ),
            "unselected": dict(
                marker=dict(
                    opacity=opacity,
                )
            ),
        }

        return go.Scattergl(**scatter_args)

    @staticmethod
    def _update_figure_layout(
        fig: go.Figure, x_axis: str = "PCA1", y_axis: str = "PCA2"
    ):
        """Update the figure layout.

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
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            plot_bgcolor="white",
            paper_bgcolor="white",
            dragmode="lasso",
            clickmode="event+select",
            uirevision="same",
            selectdirection="any",
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="lightgray",
                fixedrange=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="lightgray",
                fixedrange=False,
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
            ),
        )
