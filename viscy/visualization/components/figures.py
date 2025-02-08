"""Figure creation utilities."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


class FigureCreator:
    """Utility class for creating plotly figures."""

    @staticmethod
    def create_track_colored_figure(
        features_df,
        filtered_features_df,
        clusters: List[Dict[str, Any]] = None,
        cluster_points: set = None,
        show_arrows: bool = False,
        x_axis: str = "PCA1",
        y_axis: str = "PCA2",
        trajectory_mode: str = "x",
        selection_mode: str = "region",
    ) -> go.Figure:
        """Create scatter plot with track-based coloring.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full features dataframe.
        filtered_features_df : pd.DataFrame
            Filtered features dataframe.
        clusters : List[Dict[str, Any]], optional
            List of cluster information, by default None
        cluster_points : set, optional
            Set of points in clusters, by default None
        show_arrows : bool, optional
            Whether to show trajectory arrows, by default False
        x_axis : str, optional
            X-axis component, by default "PCA1"
        y_axis : str, optional
            Y-axis component, by default "PCA2"
        trajectory_mode : str, optional
            Trajectory mode ("x" or "y"), by default "x"
        selection_mode : str, optional
            Selection mode ("region" or "lasso"), by default "region"

        Returns
        -------
        go.Figure
            The plotly figure.
        """
        fig = go.Figure()
        unique_tracks = filtered_features_df["track_id"].unique()
        cmap = plt.cm.tab20
        track_colors = {
            track_id: f"rgb{tuple(int(x*255) for x in cmap(i % 20)[:3])}"
            for i, track_id in enumerate(unique_tracks)
        }

        # Add background points
        background_df = features_df[
            (features_df["fov_name"].isin(filtered_features_df["fov_name"].unique()))
            & (~features_df["track_id"].isin(unique_tracks))
        ]

        if not background_df.empty:
            fig.add_trace(
                go.Scattergl(
                    x=background_df[x_axis],
                    y=background_df[y_axis],
                    mode="markers",
                    marker=dict(size=12, color="lightgray", opacity=0.3),
                    name="Other tracks",
                    text=[
                        f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for track_id, t, fov in zip(
                            background_df["track_id"],
                            background_df["t"],
                            background_df["fov_name"],
                        )
                    ],
                    hoverinfo="text",
                    showlegend=True,
                    hoverlabel=dict(namelength=-1),
                )
            )

        # Add points for each selected track
        for track_id in unique_tracks:
            track_data = filtered_features_df[
                filtered_features_df["track_id"] == track_id
            ].sort_values("t")

            # Get points for this track that are in clusters
            track_points = list(
                zip(
                    [fov for fov in track_data["fov_name"]],
                    [track_id] * len(track_data),
                    [t for t in track_data["t"]],
                )
            )

            # Determine colors based on cluster membership
            colors = []
            opacities = []
            if clusters and cluster_points:
                cluster_colors = [
                    f"rgb{tuple(int(x*255) for x in plt.cm.Set2(i % 8)[:3])}"
                    for i in range(len(clusters))
                ]
                point_to_cluster = {}
                for cluster_idx, cluster in enumerate(clusters):
                    for point in cluster:
                        point_key = (point["fov_name"], point["track_id"], point["t"])
                        point_to_cluster[point_key] = cluster_idx

                for point in track_points:
                    if point in point_to_cluster:
                        colors.append(cluster_colors[point_to_cluster[point]])
                        opacities.append(1.0)
                    else:
                        colors.append("lightgray")
                        opacities.append(0.3)
            else:
                colors = [track_colors[track_id]] * len(track_data)
                opacities = [1.0] * len(track_data)

            # Add points
            fig.add_trace(
                go.Scattergl(
                    x=track_data[x_axis],
                    y=track_data[y_axis],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors,
                        line=dict(width=1, color="black"),
                        opacity=opacities,
                    ),
                    name=f"Track {track_id}",
                    text=[
                        f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for t, fov in zip(track_data["t"], track_data["fov_name"])
                    ],
                    hoverinfo="text",
                    unselected=dict(marker=dict(opacity=0.3, size=10)),
                    selected=dict(marker=dict(size=12, opacity=1.0)),
                    hoverlabel=dict(namelength=-1),
                )
            )

            # Add trajectory lines and arrows if requested
            if show_arrows and len(track_data) > 1:
                x_coords = track_data[x_axis].values
                y_coords = track_data[y_axis].values

                # Add dashed lines for the trajectory
                fig.add_trace(
                    go.Scattergl(
                        x=x_coords,
                        y=y_coords,
                        mode="lines",
                        line=dict(
                            color=track_colors[track_id],
                            width=1,
                            dash="dot",
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Add arrows at regular intervals
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
                            arrowcolor=track_colors[track_id],
                            opacity=0.8,
                        )

        # Add shaded region for trajectory selection
        if selection_mode == "region":
            x_range = [
                filtered_features_df[x_axis].min(),
                filtered_features_df[x_axis].max(),
            ]
            y_range = [
                filtered_features_df[y_axis].min(),
                filtered_features_df[y_axis].max(),
            ]

            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
            y_range = [y_range[0] - y_padding, y_range[1] + y_padding]

            if trajectory_mode == "x":
                x_mid = (x_range[0] + x_range[1]) / 2
                # Calculate tolerance based on data density
                x_values = filtered_features_df[x_axis].values
                x_std = np.std(x_values)
                tolerance = x_std * 0.5  # Use half standard deviation as tolerance

                fig.add_shape(
                    type="rect",
                    x0=x_mid - tolerance,
                    x1=x_mid + tolerance,
                    y0=y_range[0],
                    y1=y_range[1],
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    line=dict(width=1, color="blue"),
                    layer="below",
                    editable=True,
                )
            else:
                y_mid = (y_range[0] + y_range[1]) / 2
                # Calculate tolerance based on data density
                y_values = filtered_features_df[y_axis].values
                y_std = np.std(y_values)
                tolerance = y_std * 0.5  # Use half standard deviation as tolerance

                fig.add_shape(
                    type="rect",
                    x0=x_range[0],
                    x1=x_range[1],
                    y0=y_mid - tolerance,
                    y1=y_mid + tolerance,
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line=dict(width=1, color="red"),
                    layer="below",
                    editable=True,
                )

            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)

        FigureCreator._update_figure_layout(fig, x_axis, y_axis)
        return fig

    @staticmethod
    def create_time_colored_figure(
        features_df,
        filtered_features_df,
        show_arrows: bool = False,
        x_axis: str = "PCA1",
        y_axis: str = "PCA2",
        trajectory_mode: str = "x",
        selection_mode: str = "region",
    ) -> go.Figure:
        """Create scatter plot with time-based coloring.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full features dataframe.
        filtered_features_df : pd.DataFrame
            Filtered features dataframe.
        show_arrows : bool, optional
            Whether to show trajectory arrows, by default False
        x_axis : str, optional
            X-axis component, by default "PCA1"
        y_axis : str, optional
            Y-axis component, by default "PCA2"
        trajectory_mode : str, optional
            Trajectory mode ("x" or "y"), by default "x"
        selection_mode : str, optional
            Selection mode ("region" or "lasso"), by default "region"

        Returns
        -------
        go.Figure
            The plotly figure.
        """
        fig = go.Figure()

        # Add background points
        all_tracks_df = features_df[
            features_df["fov_name"].isin(filtered_features_df["fov_name"].unique())
        ]
        fig.add_trace(
            go.Scattergl(
                x=all_tracks_df[x_axis],
                y=all_tracks_df[y_axis],
                mode="markers",
                marker=dict(size=12, color="lightgray", opacity=0.3),
                name="Other points",
                text=[
                    f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                    for track_id, t, fov in zip(
                        all_tracks_df["track_id"],
                        all_tracks_df["t"],
                        all_tracks_df["fov_name"],
                    )
                ],
                hoverinfo="text",
                hoverlabel=dict(namelength=-1),
            )
        )

        # Add time-colored points
        fig.add_trace(
            go.Scattergl(
                x=filtered_features_df[x_axis],
                y=filtered_features_df[y_axis],
                mode="markers",
                marker=dict(
                    size=10,
                    color=filtered_features_df["t"],
                    colorscale="Viridis",
                    colorbar=dict(title="Time"),
                ),
                text=[
                    f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                    for track_id, t, fov in zip(
                        filtered_features_df["track_id"],
                        filtered_features_df["t"],
                        filtered_features_df["fov_name"],
                    )
                ],
                hoverinfo="text",
                showlegend=False,
                hoverlabel=dict(namelength=-1),
            )
        )

        # Add arrows if requested
        if show_arrows:
            for track_id in filtered_features_df["track_id"].unique():
                track_data = filtered_features_df[
                    filtered_features_df["track_id"] == track_id
                ].sort_values("t")

                if len(track_data) > 1:
                    x_coords = track_data[x_axis].values
                    y_coords = track_data[y_axis].values
                    distances = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)
                    threshold = np.median(distances) * 0.5

                    arrow_x = []
                    arrow_y = []

                    for i in range(len(track_data) - 1):
                        if distances[i] > threshold:
                            arrow_x.extend([x_coords[i], x_coords[i + 1], None])
                            arrow_y.extend([y_coords[i], y_coords[i + 1], None])

                    if arrow_x:
                        fig.add_trace(
                            go.Scatter(
                                x=arrow_x,
                                y=arrow_y,
                                mode="lines",
                                line=dict(
                                    color="rgba(128, 128, 128, 0.5)",
                                    width=1,
                                    dash="dot",
                                ),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

        # Add shaded region for trajectory
        if selection_mode == "region":
            x_range = [
                filtered_features_df[x_axis].min(),
                filtered_features_df[x_axis].max(),
            ]
            y_range = [
                filtered_features_df[y_axis].min(),
                filtered_features_df[y_axis].max(),
            ]

            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
            y_range = [y_range[0] - y_padding, y_range[1] + y_padding]

            if trajectory_mode == "x":
                x_mid = (x_range[0] + x_range[1]) / 2
                # Calculate tolerance based on data density
                x_values = filtered_features_df[x_axis].values
                x_std = np.std(x_values)
                tolerance = x_std * 0.5  # Use half standard deviation as tolerance

                fig.add_shape(
                    type="rect",
                    x0=x_mid - tolerance,
                    x1=x_mid + tolerance,
                    y0=y_range[0],
                    y1=y_range[1],
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    line=dict(width=1, color="blue"),
                    layer="below",
                    editable=True,
                )
            else:
                y_mid = (y_range[0] + y_range[1]) / 2
                # Calculate tolerance based on data density
                y_values = filtered_features_df[y_axis].values
                y_std = np.std(y_values)
                tolerance = y_std * 0.5  # Use half standard deviation as tolerance

                fig.add_shape(
                    type="rect",
                    x0=x_range[0],
                    x1=x_range[1],
                    y0=y_mid - tolerance,
                    y1=y_mid + tolerance,
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line=dict(width=1, color="red"),
                    layer="below",
                    editable=True,
                )

            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)

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
