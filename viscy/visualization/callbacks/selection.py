"""Selection callbacks for the visualization app."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback_context
from sklearn.cluster import DBSCAN


class SelectionCallbacks:
    """Callbacks for handling track selection and clustering."""

    def __init__(self, app_instance):
        """Initialize the callbacks.

        Parameters
        ----------
        app_instance : EmbeddingVisualizationApp
            The main application instance.
        """
        self.app_instance = app_instance

    def register(self):
        """Register all selection callbacks."""

        @self.app_instance.app.callback(
            [
                Output("selected-tracks-store", "data"),
                Output("cluster-store", "data"),
            ],
            [
                Input("scatter-plot", "selectedData"),
                Input("cluster-button", "n_clicks"),
            ],
            [
                State("selected-tracks-store", "data"),
                State("cluster-store", "data"),
                State("eps-slider", "value"),
                State("min-samples-slider", "value"),
            ],
        )
        def update_selection(
            selected_data: Dict[str, Any],
            cluster_clicks: int,
            selected_tracks: Dict[str, Any],
            cluster_data: Dict[str, Any],
            eps: float,
            min_samples: int,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """Update the selection based on user interaction.

            Parameters
            ----------
            selected_data : Dict[str, Any]
                Data for selected points.
            cluster_clicks : int
                Number of times cluster button was clicked.
            selected_tracks : Dict[str, Any]
                Currently selected tracks.
            cluster_data : Dict[str, Any]
                Current cluster data.
            eps : float
                DBSCAN epsilon parameter.
            min_samples : int
                DBSCAN min_samples parameter.

            Returns
            -------
            Tuple[Dict[str, Any], Dict[str, Any]]
                Updated selection and cluster data.
            """
            ctx = callback_context
            if not ctx.triggered:
                return selected_tracks or {}, cluster_data or {}

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id == "scatter-plot":
                if not selected_data:
                    return {}, {}

                points = selected_data["points"]
                selected = {
                    "points": [
                        {
                            "fov_name": point["text"].split("<br>")[2].split(": ")[1],
                            "track_id": int(
                                point["text"].split("<br>")[0].split(": ")[1]
                            ),
                            "t": int(point["text"].split("<br>")[1].split(": ")[1]),
                        }
                        for point in points
                    ]
                }
                return selected, cluster_data or {}

            elif trigger_id == "cluster-button" and selected_tracks:
                points = selected_tracks["points"]
                if len(points) < min_samples:
                    return selected_tracks, {}

                # Extract features for clustering
                features = []
                for point in points:
                    track_data = self.app_instance.filtered_features_df[
                        (
                            self.app_instance.filtered_features_df["fov_name"]
                            == point["fov_name"]
                        )
                        & (
                            self.app_instance.filtered_features_df["track_id"]
                            == point["track_id"]
                        )
                        & (self.app_instance.filtered_features_df["t"] == point["t"])
                    ]
                    if not track_data.empty:
                        pca_cols = [
                            col for col in track_data.columns if col.startswith("PCA")
                        ]
                        features.append(track_data[pca_cols].values[0])

                if not features:
                    return selected_tracks, {}

                # Perform clustering
                features = np.array(features)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(features)

                # Group points by cluster
                clusters = []
                for label in np.unique(labels):
                    if label == -1:
                        continue
                    cluster_points = [
                        point for i, point in enumerate(points) if labels[i] == label
                    ]
                    if cluster_points:
                        clusters.append({"points": cluster_points})

                return selected_tracks, {"clusters": clusters}

            return selected_tracks or {}, cluster_data or {}
