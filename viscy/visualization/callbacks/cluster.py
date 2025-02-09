"""Cluster-related callbacks for the visualization app."""

from typing import Dict, Any, List

import dash
from dash.dependencies import Input, Output, State


class ClusterCallbacks:
    """Class to handle cluster-related callbacks."""

    def __init__(self, app):
        """Initialize the cluster callbacks.

        Args:
            app: The main visualization app instance.
        """
        self.app = app

    def register(self):
        """Register all cluster-related callbacks."""

        @self.app.app.callback(
            [
                Output("cluster-container", "children"),
                Output("view-tabs", "value"),
                Output("scatter-plot", "figure", allow_duplicate=True),
                Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                Input("cluster-button", "n_clicks"),
                Input("clear-clusters", "n_clicks"),
            ],
            [
                State("scatter-plot", "selectedData"),
                State("scatter-plot", "figure"),
                State("color-mode", "value"),
                State("show-arrows", "value"),
                State("x-axis", "value"),
                State("y-axis", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_clusters(
            cluster_clicks: int,
            clear_clicks: int,
            selected_data: Dict[str, Any],
            current_figure: Dict[str, Any],
            color_mode: str,
            show_arrows: List[str],
            x_axis: str,
            y_axis: str,
        ):
            """Update cluster display."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if (
                button_id == "cluster-button"
                and selected_data
                and selected_data.get("points")
            ):
                # Create new cluster from selected points
                new_cluster = []
                seen_points = set()  # Track unique points to avoid duplicates

                for point in selected_data["points"]:
                    text = point["text"]
                    lines = text.split("<br>")
                    track_id = int(lines[0].split(": ")[1])
                    t = int(lines[1].split(": ")[1])
                    fov = lines[2].split(": ")[1]

                    point_key = (fov, track_id, t)
                    if (
                        point_key not in seen_points
                        and point_key in self.app.image_cache
                        and point_key
                        not in self.app.cluster_points  # Avoid adding points that are already in clusters
                    ):
                        new_cluster.append(
                            {
                                "track_id": track_id,
                                "t": t,
                                "fov_name": fov,
                            }
                        )
                        seen_points.add(point_key)
                        self.app.cluster_points.add(point_key)

                if new_cluster:
                    self.app.clusters.append(new_cluster)
                    # Create new figure with updated colors
                    fig = self.app._create_track_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                    )
                    # Always keep dragmode as lasso after creating a cluster
                    fig.update_layout(
                        dragmode="lasso",
                        clickmode="none",
                        uirevision=True,  # Keep UI state consistent
                        selectdirection="any",
                    )
                    # Clear the selection state but maintain lasso mode
                    return (
                        self.app._get_cluster_images(),
                        "clusters-tab",
                        fig,
                        None,  # Clear selection after creating cluster
                    )

            elif button_id == "clear-clusters":
                self.app.clusters = []
                self.app.cluster_points.clear()
                # Restore original coloring
                fig = self.app._create_track_colored_figure(
                    len(show_arrows or []) > 0,
                    x_axis,
                    y_axis,
                )
                # Keep dragmode as lasso after clearing clusters
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="none",
                    uirevision=True,  # Keep UI state consistent
                    selectdirection="any",
                )
                return (
                    None,
                    "clusters-tab",
                    fig,
                    None,  # Clear selection completely
                )

            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
