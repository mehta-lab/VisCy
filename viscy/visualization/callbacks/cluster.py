"""Cluster-related callbacks for the visualization app."""

import base64
import json
from typing import Dict, Any, List

import dash
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate


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
        self._register_cluster_creation_callback()
        self._register_cluster_annotation_callback()
        self._register_cluster_deletion_callback()
        self._register_export_import_callbacks()

    def _register_cluster_creation_callback(self):
        """Register callback for creating new clusters."""

        @self.app.app.callback(
            [
                Output("cluster-list", "children"),
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
                        and point_key not in self.app.cluster_points
                    ):
                        new_cluster.append(
                            {
                                "track_id": track_id,
                                "t": t,
                                "fov_name": fov,
                                "x": point["x"],  # Store coordinates for reference
                                "y": point["y"],
                            }
                        )
                        seen_points.add(point_key)
                        self.app.cluster_points.add(point_key)

                if new_cluster:
                    # Add new cluster
                    cluster_idx = len(self.app.clusters)
                    self.app.clusters.append(new_cluster)

                    # Initialize empty annotation for the new cluster
                    self.app.cluster_annotations[cluster_idx] = {
                        "label": f"Cluster {cluster_idx + 1}",
                        "description": "",
                    }

                    # Create new figure with updated colors
                    fig = self.app._create_track_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                    )

                    # Maintain lasso mode and selection state
                    fig.update_layout(
                        dragmode="lasso",
                        clickmode="event+select",
                        uirevision=True,
                        selectdirection="any",
                    )

                    # Create cluster display
                    cluster_display = self.app._get_cluster_images()

                    return cluster_display, "clusters-tab", fig, None

            elif button_id == "clear-clusters":
                # Clear all clusters
                self.app.clusters = []
                self.app.cluster_annotations = {}
                self.app.cluster_points.clear()

                # Update figure
                fig = self.app._create_track_colored_figure(
                    len(show_arrows or []) > 0,
                    x_axis,
                    y_axis,
                )
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="event+select",
                    uirevision=True,
                    selectdirection="any",
                )

                return None, "clusters-tab", fig, None

            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    def _register_cluster_annotation_callback(self):
        """Register callbacks for cluster annotation updates."""

        @self.app.app.callback(
            Output("cluster-store", "data", allow_duplicate=True),
            [
                Input({"type": "cluster-label", "index": ALL}, "value"),
                Input({"type": "cluster-description", "index": ALL}, "value"),
            ],
            [State("cluster-store", "data")],
            prevent_initial_call=True,
        )
        def update_cluster_annotations(labels, descriptions, current_data):
            """Update cluster annotations when labels or descriptions change."""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            trigger = ctx.triggered[0]
            prop_id = json.loads(trigger["prop_id"].split(".")[0])
            cluster_idx = prop_id["index"]

            if "label" in prop_id["type"]:
                self.app.cluster_annotations[cluster_idx]["label"] = trigger["value"]
            else:
                self.app.cluster_annotations[cluster_idx]["description"] = trigger[
                    "value"
                ]

            return self.app.cluster_annotations

    def _register_cluster_deletion_callback(self):
        """Register callback for deleting individual clusters."""

        @self.app.app.callback(
            [
                Output("cluster-list", "children", allow_duplicate=True),
                Output("scatter-plot", "figure", allow_duplicate=True),
            ],
            [Input({"type": "delete-cluster", "index": ALL}, "n_clicks")],
            [
                State("show-arrows", "value"),
                State("x-axis", "value"),
                State("y-axis", "value"),
            ],
            prevent_initial_call=True,
        )
        def delete_cluster(delete_clicks, show_arrows, x_axis, y_axis):
            """Delete a cluster when its delete button is clicked."""
            ctx = dash.callback_context
            if not ctx.triggered or not any(delete_clicks):
                raise PreventUpdate

            trigger = ctx.triggered[0]
            cluster_idx = json.loads(trigger["prop_id"].split(".")[0])["index"]

            # Remove cluster points from tracking set
            for point in self.app.clusters[cluster_idx]:
                point_key = (point["fov_name"], point["track_id"], point["t"])
                if point_key in self.app.cluster_points:
                    self.app.cluster_points.remove(point_key)

            # Remove cluster and its annotation
            self.app.clusters.pop(cluster_idx)
            self.app.cluster_annotations.pop(cluster_idx)

            # Reindex remaining annotations
            self.app.cluster_annotations = {
                i: ann for i, ann in enumerate(self.app.cluster_annotations.values())
            }

            # Update figure
            fig = self.app._create_track_colored_figure(
                len(show_arrows or []) > 0,
                x_axis,
                y_axis,
            )
            fig.update_layout(
                dragmode="lasso",
                clickmode="event+select",
                uirevision=True,
                selectdirection="any",
            )

            return self.app._get_cluster_images(), fig

    def _register_export_import_callbacks(self):
        """Register callbacks for exporting and importing clusters."""

        @self.app.app.callback(
            Output("download-clusters", "data"),
            Input("export-clusters", "n_clicks"),
            prevent_initial_call=True,
        )
        def export_clusters(n_clicks):
            """Export clusters to a JSON file."""
            if not n_clicks:
                raise PreventUpdate

            export_data = self.app.export_clusters()
            return dict(
                content=json.dumps(export_data, indent=2),
                filename="clusters.json",
                type="application/json",
            )

        @self.app.app.callback(
            [
                Output("cluster-list", "children", allow_duplicate=True),
                Output("scatter-plot", "figure", allow_duplicate=True),
                Output("upload-clusters", "style"),
            ],
            [
                Input("import-clusters", "n_clicks"),
                Input("upload-clusters", "contents"),
            ],
            [
                State("upload-clusters", "filename"),
                State("show-arrows", "value"),
                State("x-axis", "value"),
                State("y-axis", "value"),
                State("upload-clusters", "style"),
            ],
            prevent_initial_call=True,
        )
        def import_clusters(
            import_clicks, contents, filename, show_arrows, x_axis, y_axis, upload_style
        ):
            """Import clusters from a JSON file."""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id == "import-clusters":
                # Show upload component
                upload_style["display"] = "block"
                return dash.no_update, dash.no_update, upload_style

            elif trigger_id == "upload-clusters" and contents:
                # Process uploaded file
                content_type, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                import_data = json.loads(decoded)

                # Import clusters
                self.app.clusters = []
                self.app.cluster_annotations = {}
                self.app.cluster_points.clear()

                for i, cluster_data in enumerate(import_data["clusters"]):
                    self.app.clusters.append(cluster_data["points"])
                    self.app.cluster_annotations[i] = cluster_data["annotation"]
                    for point in cluster_data["points"]:
                        self.app.cluster_points.add(
                            (point["fov_name"], point["track_id"], point["t"])
                        )

                # Update figure
                fig = self.app._create_track_colored_figure(
                    len(show_arrows or []) > 0,
                    x_axis,
                    y_axis,
                )
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="none",
                    uirevision=True,
                    selectdirection="any",
                )

                # Hide upload component
                upload_style["display"] = "none"

                return self.app._get_cluster_images(), fig, upload_style

            return dash.no_update, dash.no_update, upload_style
