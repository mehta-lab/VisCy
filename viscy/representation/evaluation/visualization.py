import atexit
import base64
import json
import logging
from io import BytesIO
from pathlib import Path

import dash
import dash.dependencies as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingVisualizationApp:
    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        features_path: str,
        channels_to_display: list[str] | str,
        fov_tracks: dict[str, list[int] | str],
        z_range: tuple[int, int] = (0, 1),
        yx_patch_size: tuple[int, int] = (128, 128),
        num_PC_components: int = 3,
        cache_path: str | None = None,
        num_loading_workers: int = 16,
        output_dir: str | None = None,
    ) -> None:
        """
        Initialize a Dash application for visualizing the DynaCLR embeddings.

        This class provides a visualization tool for visualizing the DynaCLR embeddings into a 2D space (e.g. PCA, UMAP, PHATE).
        It allows users to interactively explore and analyze trajectories, visualize clusters, and explore the embedding space.

        Parameters
        ----------
        data_path: str
            Path to the data directory.
        tracks_path: str
            Path to the tracks directory.
        features_path: str
            Path to the features directory.
        channels_to_display: list[str] | str
            List of channels to display.
        fov_tracks: dict[str, list[int] | str]
            Dictionary of FOV names and track IDs.
        z_range: tuple[int, int] | list[int,int]
            Range of z-slices to display.
        yx_patch_size: tuple[int, int] | list[int,int]
            Size of the yx-patch to display.
        num_PC_components: int
            Number of PCA components to use.
        cache_path: str | None
            Path to the cache directory.
        num_loading_workers: int
            Number of workers to use for loading data.
        output_dir: str | None, optional
            Directory to save CSV files and other outputs. If None, uses current working directory.
        Returns
        -------
        None
            Initializes the visualization app.
        """
        self.data_path = Path(data_path)
        self.tracks_path = Path(tracks_path)
        self.features_path = Path(features_path)
        self.fov_tracks = fov_tracks
        self.image_cache = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.app = None
        self.features_df = None
        self.fig = None
        self.channels_to_display = channels_to_display
        self.z_range = z_range
        self.yx_patch_size = yx_patch_size
        self.filtered_tracks_by_fov = {}
        self._z_idx = (self.z_range[1] - self.z_range[0]) // 2
        self.num_PC_components = num_PC_components
        self.num_loading_workers = num_loading_workers
        # Initialize cluster storage before preparing data and creating figure
        self.clusters = []  # List to store all clusters
        self.cluster_points = set()  # Set to track all points in clusters
        self.cluster_names = {}  # Dictionary to store cluster names
        self.next_cluster_id = 1  # Counter for cluster IDs
        # Initialize data
        self._prepare_data()
        self._create_figure()
        self._init_app()
        atexit.register(self._cleanup_cache)

    def _prepare_data(self):
        """Prepare the feature data and PCA transformation"""
        embedding_dataset = read_embedding_dataset(self.features_path)
        features = embedding_dataset["features"]
        self.features_df = features["sample"].to_dataframe().reset_index(drop=True)

        # Check if UMAP or PHATE columns already exist
        existing_dims = []
        dim_options = []

        # Check for PCA and compute if needed
        if not any(col.startswith("PC") for col in self.features_df.columns):
            # PCA transformation
            scaled_features = StandardScaler().fit_transform(features.values)
            pca = PCA(n_components=self.num_PC_components)
            pca_coords = pca.fit_transform(scaled_features)

            # Add PCA coordinates to the features dataframe
            for i in range(self.num_PC_components):
                self.features_df[f"PC{i + 1}"] = pca_coords[:, i]

            # Store explained variance for PCA
            self.pca_explained_variance = [
                f"PC{i + 1} ({var:.1f}%)"
                for i, var in enumerate(pca.explained_variance_ratio_ * 100)
            ]

            # Add PCA options
            for i, pc_label in enumerate(self.pca_explained_variance):
                dim_options.append({"label": pc_label, "value": f"PC{i + 1}"})
                existing_dims.append(f"PC{i + 1}")

        # Check for UMAP coordinates
        umap_dims = [col for col in self.features_df.columns if col.startswith("UMAP")]
        if umap_dims:
            for dim in umap_dims:
                dim_options.append({"label": dim, "value": dim})
                existing_dims.append(dim)

        # Check for PHATE coordinates
        phate_dims = [
            col for col in self.features_df.columns if col.startswith("PHATE")
        ]
        if phate_dims:
            for dim in phate_dims:
                dim_options.append({"label": dim, "value": dim})
                existing_dims.append(dim)

        # Store dimension options for dropdowns
        self.dim_options = dim_options

        # Set default x and y axes based on available dimensions
        self.default_x = existing_dims[0] if existing_dims else "PC1"
        self.default_y = existing_dims[1] if len(existing_dims) > 1 else "PC2"

        # Process each FOV and its track IDs
        all_filtered_features = []
        for fov_name, track_ids in self.fov_tracks.items():
            if track_ids == "all":
                fov_tracks = (
                    self.features_df[self.features_df["fov_name"] == fov_name][
                        "track_id"
                    ]
                    .unique()
                    .tolist()
                )
            else:
                fov_tracks = track_ids

            self.filtered_tracks_by_fov[fov_name] = fov_tracks

            # Filter features for this FOV and its track IDs
            fov_features = self.features_df[
                (self.features_df["fov_name"] == fov_name)
                & (self.features_df["track_id"].isin(fov_tracks))
            ]
            all_filtered_features.append(fov_features)

        # Combine all filtered features
        self.filtered_features_df = pd.concat(all_filtered_features, axis=0)

    def _create_figure(self):
        """Create the initial scatter plot figure"""
        self.fig = self._create_track_colored_figure()

    def _init_app(self):
        """Initialize the Dash application"""
        self.app = dash.Dash(__name__)

        # Add cluster assignment button next to clear selection
        cluster_controls = html.Div(
            [
                html.Button(
                    "Assign to New Cluster",
                    id="assign-cluster",
                    style={
                        "backgroundColor": "#28a745",
                        "color": "white",
                        "border": "none",
                        "padding": "5px 10px",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "marginRight": "10px",
                    },
                ),
                html.Button(
                    "Clear All Clusters",
                    id="clear-clusters",
                    style={
                        "backgroundColor": "#dc3545",
                        "color": "white",
                        "border": "none",
                        "padding": "5px 10px",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "marginRight": "10px",
                    },
                ),
                html.Button(
                    "Save Clusters to CSV",
                    id="save-clusters-csv",
                    style={
                        "backgroundColor": "#17a2b8",
                        "color": "white",
                        "border": "none",
                        "padding": "5px 10px",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "marginRight": "10px",
                    },
                ),
                html.Button(
                    "Clear Selection",
                    id="clear-selection",
                    style={
                        "backgroundColor": "#6c757d",
                        "color": "white",
                        "border": "none",
                        "padding": "5px 10px",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={"marginLeft": "10px", "display": "inline-block"},
        )
        # Create tabs for different views
        tabs = dcc.Tabs(
            id="view-tabs",
            value="timeline-tab",
            children=[
                dcc.Tab(
                    label="Track Timeline",
                    value="timeline-tab",
                    children=[
                        html.Div(
                            id="track-timeline",
                            style={
                                "height": "auto",
                                "overflowY": "auto",
                                "maxHeight": "80vh",
                                "padding": "10px",
                                "marginTop": "10px",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Clusters",
                    value="clusters-tab",
                    id="clusters-tab",
                    children=[
                        html.Div(
                            id="cluster-container",
                            style={
                                "padding": "10px",
                                "marginTop": "10px",
                            },
                        ),
                    ],
                    style={"display": "none"},  # Initially hidden
                ),
            ],
            style={"marginTop": "20px"},
        )

        # Add modal for cluster naming
        cluster_name_modal = html.Div(
            id="cluster-name-modal",
            children=[
                html.Div(
                    [
                        html.H3("Name Your Cluster", style={"marginBottom": "20px"}),
                        html.Label("Cluster Name:"),
                        dcc.Input(
                            id="cluster-name-input",
                            type="text",
                            placeholder="Enter cluster name...",
                            style={"width": "100%", "marginBottom": "20px"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Save",
                                    id="save-cluster-name",
                                    style={
                                        "backgroundColor": "#28a745",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px 16px",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                        "marginRight": "10px",
                                    },
                                ),
                                html.Button(
                                    "Cancel",
                                    id="cancel-cluster-name",
                                    style={
                                        "backgroundColor": "#6c757d",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px 16px",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                            style={"textAlign": "right"},
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "30px",
                        "borderRadius": "8px",
                        "maxWidth": "400px",
                        "margin": "auto",
                        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                        "border": "1px solid #ddd",
                    },
                )
            ],
            style={
                "display": "none",
                "position": "fixed",
                "top": "0",
                "left": "0",
                "width": "100%",
                "height": "100%",
                "backgroundColor": "rgba(0, 0, 0, 0.5)",
                "zIndex": "1000",
                "justifyContent": "center",
                "alignItems": "center",
            },
        )

        # Update layout to use tabs
        self.app.layout = html.Div(
            style={
                "maxWidth": "95vw",
                "margin": "auto",
                "padding": "20px",
            },
            children=[
                html.H1(
                    "Track Visualization",
                    style={"textAlign": "center", "marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Div(
                            style={
                                "width": "100%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "marginBottom": "20px",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "20px",
                                        "flexWrap": "wrap",
                                    },
                                    children=[
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Color by:",
                                                    style={"marginRight": "10px"},
                                                ),
                                                dcc.Dropdown(
                                                    id="color-mode",
                                                    options=[
                                                        {
                                                            "label": "Track ID",
                                                            "value": "track",
                                                        },
                                                        {
                                                            "label": "Time",
                                                            "value": "time",
                                                        },
                                                    ],
                                                    value="track",
                                                    style={"width": "200px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show-arrows",
                                                    options=[
                                                        {
                                                            "label": "Show arrows",
                                                            "value": "show",
                                                        }
                                                    ],
                                                    value=[],
                                                    style={"marginLeft": "20px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "X-axis:",
                                                    style={"marginRight": "10px"},
                                                ),
                                                dcc.Dropdown(
                                                    id="x-axis",
                                                    options=self.dim_options,
                                                    value=self.default_x,
                                                    style={"width": "200px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Y-axis:",
                                                    style={"marginRight": "10px"},
                                                ),
                                                dcc.Dropdown(
                                                    id="y-axis",
                                                    options=self.dim_options,
                                                    value=self.default_y,
                                                    style={"width": "200px"},
                                                ),
                                            ]
                                        ),
                                        cluster_controls,
                                    ],
                                ),
                            ],
                        ),
                    ]
                ),
                dcc.Loading(
                    id="loading",
                    children=[
                        dcc.Graph(
                            id="scatter-plot",
                            figure=self.fig,
                            config={
                                "displayModeBar": True,
                                "editable": False,
                                "showEditInChartStudio": False,
                                "modeBarButtonsToRemove": [
                                    "select2d",
                                    "resetScale2d",
                                ],
                                "edits": {
                                    "annotationPosition": False,
                                    "annotationTail": False,
                                    "annotationText": False,
                                    "shapePosition": True,
                                },
                                "scrollZoom": True,
                            },
                            style={"height": "50vh"},
                        ),
                    ],
                    type="default",
                ),
                tabs,
                cluster_name_modal,
            ],
        )

        @self.app.callback(
            [
                dd.Output("scatter-plot", "figure", allow_duplicate=True),
                dd.Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                dd.Input("color-mode", "value"),
                dd.Input("show-arrows", "value"),
                dd.Input("x-axis", "value"),
                dd.Input("y-axis", "value"),
                dd.Input("scatter-plot", "relayoutData"),
                dd.Input("scatter-plot", "selectedData"),
            ],
            [dd.State("scatter-plot", "figure")],
            prevent_initial_call=True,
        )
        def update_figure(
            color_mode,
            show_arrows,
            x_axis,
            y_axis,
            relayout_data,
            selected_data,
            current_figure,
        ):
            show_arrows = len(show_arrows or []) > 0

            ctx = dash.callback_context
            if not ctx.triggered:
                triggered_id = "No clicks yet"
            else:
                triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Create new figure when necessary
            if triggered_id in [
                "color-mode",
                "show-arrows",
                "x-axis",
                "y-axis",
            ]:
                if color_mode == "track":
                    fig = self._create_track_colored_figure(show_arrows, x_axis, y_axis)
                else:
                    fig = self._create_time_colored_figure(show_arrows, x_axis, y_axis)

                # Update dragmode and selection settings
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="event+select",
                    uirevision="true",
                    selectdirection="any",
                )
            else:
                fig = dash.no_update

            return fig, selected_data

        @self.app.callback(
            dd.Output("track-timeline", "children"),
            [dd.Input("scatter-plot", "clickData")],
            prevent_initial_call=True,
        )
        def update_track_timeline(clickData):
            """Update the track timeline based on the clicked point"""
            if clickData is None:
                return html.Div("Click on a point to see the track timeline")

            # Parse the hover text to get track_id, time and fov_name
            hover_text = clickData["points"][0]["text"]
            track_id = int(hover_text.split("<br>")[0].split(": ")[1])
            clicked_time = int(hover_text.split("<br>")[1].split(": ")[1])
            fov_name = hover_text.split("<br>")[2].split(": ")[1]

            # Get all timepoints for this track
            track_data = self.features_df[
                (self.features_df["fov_name"] == fov_name)
                & (self.features_df["track_id"] == track_id)
            ].sort_values("t")

            if track_data.empty:
                return html.Div(f"No data found for track {track_id}")

            # Get unique timepoints
            timepoints = track_data["t"].unique()

            # Create a list to store all timepoint columns
            timepoint_columns = []

            # First create the time labels row
            time_labels = []
            for t in timepoints:
                is_clicked = t == clicked_time
                time_style = {
                    "width": "150px",
                    "textAlign": "center",
                    "padding": "5px",
                    "fontWeight": "bold" if is_clicked else "normal",
                    "color": "#007bff" if is_clicked else "black",
                }
                time_labels.append(html.Div(f"t={t}", style=time_style))

            timepoint_columns.append(
                html.Div(
                    time_labels,
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "minWidth": "fit-content",
                        "borderBottom": "2px solid #ddd",
                        "marginBottom": "10px",
                        "paddingBottom": "5px",
                    },
                )
            )

            # Then create image rows for each channel
            for channel in self.channels_to_display:
                channel_images = []
                for t in timepoints:
                    cache_key = (fov_name, track_id, t)
                    if (
                        cache_key in self.image_cache
                        and channel in self.image_cache[cache_key]
                    ):
                        is_clicked = t == clicked_time
                        image_style = {
                            "width": "150px",
                            "height": "150px",
                            "border": (
                                "3px solid #007bff" if is_clicked else "1px solid #ddd"
                            ),
                            "borderRadius": "4px",
                        }
                        channel_images.append(
                            html.Div(
                                html.Img(
                                    src=self.image_cache[cache_key][channel],
                                    style=image_style,
                                ),
                                style={
                                    "width": "150px",
                                    "padding": "5px",
                                },
                            )
                        )

                if channel_images:
                    # Add channel label
                    timepoint_columns.append(
                        html.Div(
                            [
                                html.Div(
                                    channel,
                                    style={
                                        "width": "100px",
                                        "fontWeight": "bold",
                                        "fontSize": "14px",
                                        "padding": "5px",
                                        "backgroundColor": "#f8f9fa",
                                        "borderRadius": "4px",
                                        "marginBottom": "5px",
                                        "textAlign": "center",
                                    },
                                ),
                                html.Div(
                                    channel_images,
                                    style={
                                        "display": "flex",
                                        "flexDirection": "row",
                                        "minWidth": "fit-content",
                                        "marginBottom": "15px",
                                    },
                                ),
                            ]
                        )
                    )

            # Create the main container with synchronized scrolling
            return html.Div(
                [
                    html.H4(
                        f"Track {track_id} (FOV: {fov_name})",
                        style={
                            "marginBottom": "20px",
                            "fontSize": "20px",
                            "fontWeight": "bold",
                            "color": "#2c3e50",
                        },
                    ),
                    html.Div(
                        timepoint_columns,
                        style={
                            "overflowX": "auto",
                            "overflowY": "hidden",
                            "whiteSpace": "nowrap",
                            "backgroundColor": "white",
                            "padding": "20px",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                            "marginBottom": "20px",
                        },
                    ),
                ]
            )

        # Add callback to show/hide clusters tab and handle modal
        @self.app.callback(
            [
                dd.Output("clusters-tab", "style"),
                dd.Output("cluster-container", "children"),
                dd.Output("view-tabs", "value"),
                dd.Output("scatter-plot", "figure", allow_duplicate=True),
                dd.Output("cluster-name-modal", "style"),
                dd.Output("cluster-name-input", "value"),
                dd.Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                dd.Input("assign-cluster", "n_clicks"),
                dd.Input("clear-clusters", "n_clicks"),
                dd.Input("save-cluster-name", "n_clicks"),
                dd.Input("cancel-cluster-name", "n_clicks"),
                dd.Input({"type": "edit-cluster-name", "index": dash.ALL}, "n_clicks"),
            ],
            [
                dd.State("scatter-plot", "selectedData"),
                dd.State("scatter-plot", "figure"),
                dd.State("color-mode", "value"),
                dd.State("show-arrows", "value"),
                dd.State("x-axis", "value"),
                dd.State("y-axis", "value"),
                dd.State("cluster-name-input", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_clusters_tab(
            assign_clicks,
            clear_clicks,
            save_name_clicks,
            cancel_name_clicks,
            edit_name_clicks,
            selected_data,
            current_figure,
            color_mode,
            show_arrows,
            x_axis,
            y_axis,
            cluster_name,
        ):
            ctx = dash.callback_context
            if not ctx.triggered:
                return (
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Handle edit cluster name button clicks
            if button_id.startswith('{"type":"edit-cluster-name"'):
                try:
                    id_dict = json.loads(button_id)
                    cluster_idx = id_dict["index"]

                    # Get current cluster name
                    current_name = self.cluster_names.get(
                        cluster_idx, f"Cluster {cluster_idx + 1}"
                    )

                    # Show modal
                    modal_style = {
                        "display": "flex",
                        "position": "fixed",
                        "top": "0",
                        "left": "0",
                        "width": "100%",
                        "height": "100%",
                        "backgroundColor": "rgba(0, 0, 0, 0.5)",
                        "zIndex": "1000",
                        "justifyContent": "center",
                        "alignItems": "center",
                    }

                    return (
                        {"display": "block"},
                        self._get_cluster_images(),
                        "clusters-tab",
                        dash.no_update,
                        modal_style,
                        current_name,
                        dash.no_update,  # Don't change selection
                    )
                except Exception:
                    return (
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                    )

            if (
                button_id == "assign-cluster"
                and selected_data
                and selected_data.get("points")
            ):
                # Create new cluster from selected points
                new_cluster = []
                for point in selected_data["points"]:
                    text = point["text"]
                    lines = text.split("<br>")
                    track_id = int(lines[0].split(": ")[1])
                    t = int(lines[1].split(": ")[1])
                    fov = lines[2].split(": ")[1]

                    cache_key = (fov, track_id, t)
                    if cache_key in self.image_cache:
                        new_cluster.append(
                            {
                                "track_id": track_id,
                                "t": t,
                                "fov_name": fov,
                            }
                        )
                        self.cluster_points.add(cache_key)

                if new_cluster:
                    # Add cluster to list but don't assign name yet
                    self.clusters.append(new_cluster)
                    # Open modal for naming
                    modal_style = {
                        "display": "flex",
                        "position": "fixed",
                        "top": "0",
                        "left": "0",
                        "width": "100%",
                        "height": "100%",
                        "backgroundColor": "rgba(0, 0, 0, 0.5)",
                        "zIndex": "1000",
                        "justifyContent": "center",
                        "alignItems": "center",
                    }
                    return (
                        {"display": "block"},
                        self._get_cluster_images(),
                        "clusters-tab",
                        dash.no_update,  # Don't update figure yet
                        modal_style,  # Show modal
                        "",  # Clear input
                        None,  # Clear selection
                    )

            elif button_id == "save-cluster-name" and cluster_name:
                # Assign name to the most recently created cluster
                if self.clusters:
                    cluster_id = len(self.clusters) - 1
                    self.cluster_names[cluster_id] = cluster_name.strip()

                    # Create new figure with updated colors
                    fig = self._create_track_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                    )
                    # Ensure the dragmode is set based on selection_mode
                    fig.update_layout(
                        dragmode="lasso",
                        clickmode="event+select",
                        uirevision="true",  # Keep the UI state
                        selectdirection="any",
                    )
                    modal_style = {"display": "none"}
                    return (
                        {"display": "block"},
                        self._get_cluster_images(),
                        "clusters-tab",
                        fig,
                        modal_style,  # Hide modal
                        "",  # Clear input
                        None,  # Clear selection
                    )

            elif button_id == "cancel-cluster-name":
                # Remove the cluster that was just created
                if self.clusters:
                    # Remove points from cluster_points set
                    for point in self.clusters[-1]:
                        cache_key = (point["fov_name"], point["track_id"], point["t"])
                        self.cluster_points.discard(cache_key)
                    # Remove the cluster
                    self.clusters.pop()

                    # Create new figure with updated colors
                    fig = self._create_track_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                    )
                    # Ensure the dragmode is set based on selection_mode
                    fig.update_layout(
                        dragmode="lasso",
                        clickmode="event+select",
                        uirevision="true",  # Keep the UI state
                        selectdirection="any",
                    )
                    modal_style = {"display": "none"}
                    return (
                        (
                            {"display": "none"}
                            if not self.clusters
                            else {"display": "block"}
                        ),
                        self._get_cluster_images() if self.clusters else None,
                        "timeline-tab" if not self.clusters else "clusters-tab",
                        fig,
                        modal_style,  # Hide modal
                        "",  # Clear input
                        None,  # Clear selection
                    )

            elif button_id == "clear-clusters":
                self.clusters = []
                self.cluster_points.clear()
                self.cluster_names.clear()
                # Restore original coloring
                fig = self._create_track_colored_figure(
                    len(show_arrows or []) > 0,
                    x_axis,
                    y_axis,
                )
                # Reset UI state completely to ensure clean slate
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="event+select",
                    uirevision=None,  # Reset UI state completely
                    selectdirection="any",
                )
                modal_style = {"display": "none"}
                return (
                    {"display": "none"},
                    None,
                    "timeline-tab",
                    fig,
                    modal_style,
                    "",
                    None,
                )  # Clear selection

            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        # Add callback for saving clusters to CSV
        @self.app.callback(
            dd.Output("cluster-container", "children", allow_duplicate=True),
            [dd.Input("save-clusters-csv", "n_clicks")],
            prevent_initial_call=True,
        )
        def save_clusters_csv(n_clicks):
            """Callback to save clusters to CSV file"""
            if n_clicks and self.clusters:
                try:
                    output_path = self.save_clusters_to_csv()
                    return html.Div(
                        [
                            html.H3("Clusters", style={"marginBottom": "20px"}),
                            html.Div(
                                f"✅ Successfully saved {len(self.clusters)} clusters to: {output_path}",
                                style={
                                    "backgroundColor": "#d4edda",
                                    "color": "#155724",
                                    "padding": "10px",
                                    "borderRadius": "4px",
                                    "marginBottom": "20px",
                                    "border": "1px solid #c3e6cb",
                                },
                            ),
                            self._get_cluster_images(),
                        ]
                    )
                except Exception as e:
                    return html.Div(
                        [
                            html.H3("Clusters", style={"marginBottom": "20px"}),
                            html.Div(
                                f"❌ Error saving clusters: {str(e)}",
                                style={
                                    "backgroundColor": "#f8d7da",
                                    "color": "#721c24",
                                    "padding": "10px",
                                    "borderRadius": "4px",
                                    "marginBottom": "20px",
                                    "border": "1px solid #f5c6cb",
                                },
                            ),
                            self._get_cluster_images(),
                        ]
                    )
            elif n_clicks and not self.clusters:
                return html.Div(
                    [
                        html.H3("Clusters", style={"marginBottom": "20px"}),
                        html.Div(
                            "⚠️ No clusters to save. Create clusters first by selecting points and clicking 'Assign to New Cluster'.",
                            style={
                                "backgroundColor": "#fff3cd",
                                "color": "#856404",
                                "padding": "10px",
                                "borderRadius": "4px",
                                "marginBottom": "20px",
                                "border": "1px solid #ffeaa7",
                            },
                        ),
                    ]
                )
            return dash.no_update

        @self.app.callback(
            [
                dd.Output("scatter-plot", "figure", allow_duplicate=True),
                dd.Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [dd.Input("clear-selection", "n_clicks")],
            [
                dd.State("color-mode", "value"),
                dd.State("show-arrows", "value"),
                dd.State("x-axis", "value"),
                dd.State("y-axis", "value"),
            ],
            prevent_initial_call=True,
        )
        def clear_selection(n_clicks, color_mode, show_arrows, x_axis, y_axis):
            """Callback to clear the selection and restore original opacity"""
            if n_clicks:
                # Create a new figure with no selections
                if color_mode == "track":
                    fig = self._create_track_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                    )
                else:
                    fig = self._create_time_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                    )

                # Update layout to maintain lasso mode but clear selections
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="event+select",
                    uirevision=None,  # Reset UI state
                    selectdirection="any",
                )

                return fig, None  # Return new figure and clear selectedData
            return dash.no_update, dash.no_update

    def _calculate_equal_aspect_ranges(self, x_data, y_data):
        """Calculate ranges for x and y axes to ensure equal aspect ratio.

        Parameters
        ----------
        x_data : array-like
            Data for x-axis
        y_data : array-like
            Data for y-axis

        Returns
        -------
        tuple
            (x_range, y_range) as tuples of (min, max) with equal scaling
        """
        # Get data ranges
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

        # Add padding (5% on each side)
        x_padding = 0.05 * (x_max - x_min)
        y_padding = 0.05 * (y_max - y_min)

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Ensure equal scaling by using the larger range
        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range > y_range:
            # Expand y-range to match x-range aspect ratio
            y_center = (y_max + y_min) / 2
            y_min = y_center - x_range / 2
            y_max = y_center + x_range / 2
        else:
            # Expand x-range to match y-range aspect ratio
            x_center = (x_max + x_min) / 2
            x_min = x_center - y_range / 2
            x_max = x_center + y_range / 2

        return (x_min, x_max), (y_min, y_max)

    def _create_track_colored_figure(
        self,
        show_arrows=False,
        x_axis=None,
        y_axis=None,
    ):
        """Create scatter plot with track-based coloring"""
        x_axis = x_axis or self.default_x
        y_axis = y_axis or self.default_y

        unique_tracks = self.filtered_features_df["track_id"].unique()
        cmap = plt.cm.tab20
        track_colors = {
            track_id: f"rgb{tuple(int(x * 255) for x in cmap(i % 20)[:3])}"
            for i, track_id in enumerate(unique_tracks)
        }

        fig = go.Figure()

        # Set initial layout with lasso mode
        fig.update_layout(
            dragmode="lasso",
            clickmode="event+select",
            selectdirection="any",
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

        # Add background points with hover info (excluding the colored tracks)
        background_df = self.features_df[
            (self.features_df["fov_name"].isin(self.fov_tracks.keys()))
            & (~self.features_df["track_id"].isin(unique_tracks))
        ]

        if not background_df.empty:
            # Subsample background points if there are too many
            if len(background_df) > 5000:  # Adjust this threshold as needed
                background_df = background_df.sample(n=5000, random_state=42)

            fig.add_trace(
                go.Scattergl(
                    x=background_df[x_axis],
                    y=background_df[y_axis],
                    mode="markers",
                    marker=dict(size=12, color="lightgray", opacity=0.3),
                    name=f"Other tracks (showing {len(background_df)} of {len(self.features_df)} points)",
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
            track_data = self.filtered_features_df[
                self.filtered_features_df["track_id"] == track_id
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
            if self.clusters:
                cluster_colors = [
                    f"rgb{tuple(int(x * 255) for x in plt.cm.Set2(i % 8)[:3])}"
                    for i in range(len(self.clusters))
                ]
                point_to_cluster = {}
                for cluster_idx, cluster in enumerate(self.clusters):
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

            # Add points using Scattergl for better performance
            scatter_kwargs = {
                "x": track_data[x_axis],
                "y": track_data[y_axis],
                "mode": "markers",
                "marker": dict(
                    size=10,  # Reduced size
                    color=colors,
                    line=dict(width=1, color="black"),
                    opacity=opacities,
                ),
                "name": f"Track {track_id}",
                "text": [
                    f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                    for t, fov in zip(track_data["t"], track_data["fov_name"])
                ],
                "hoverinfo": "text",
                "hoverlabel": dict(namelength=-1),  # Show full text in hover
            }

            # Only apply selection properties if there are clusters
            # This prevents opacity conflicts when no clusters exist
            if self.clusters:
                scatter_kwargs.update(
                    {
                        "unselected": dict(marker=dict(opacity=0.3, size=10)),
                        "selected": dict(marker=dict(size=12, opacity=1.0)),
                    }
                )

            fig.add_trace(go.Scattergl(**scatter_kwargs))

            # Add trajectory lines and arrows if requested
            if show_arrows and len(track_data) > 1:
                x_coords = track_data[x_axis].values
                y_coords = track_data[y_axis].values

                # Add dashed lines for the trajectory using Scattergl
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

                # Add arrows at regular intervals (reduced frequency)
                arrow_interval = max(
                    1, len(track_data) // 3
                )  # Reduced number of arrows
                for i in range(0, len(track_data) - 1, arrow_interval):
                    # Calculate arrow angle
                    dx = x_coords[i + 1] - x_coords[i]
                    dy = y_coords[i + 1] - y_coords[i]

                    # Only add arrow if there's significant movement
                    if dx * dx + dy * dy > 1e-6:  # Minimum distance threshold
                        # Add arrow annotation
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
                            arrowsize=1,  # Reduced size
                            arrowwidth=1,  # Reduced width
                            arrowcolor=track_colors[track_id],
                            opacity=0.8,
                        )

        # Compute axis ranges to ensure equal aspect ratio
        all_x_data = self.filtered_features_df[x_axis]
        all_y_data = self.filtered_features_df[y_axis]

        if not all_x_data.empty and not all_y_data.empty:
            x_range, y_range = self._calculate_equal_aspect_ranges(
                all_x_data, all_y_data
            )

            # Set equal aspect ratio and range
            fig.update_layout(
                xaxis=dict(
                    range=x_range, scaleanchor="y", scaleratio=1, constrain="domain"
                ),
                yaxis=dict(range=y_range, constrain="domain"),
            )

        return fig

    def _create_time_colored_figure(
        self,
        show_arrows=False,
        x_axis=None,
        y_axis=None,
    ):
        """Create scatter plot with time-based coloring"""
        x_axis = x_axis or self.default_x
        y_axis = y_axis or self.default_y

        fig = go.Figure()

        # Set initial layout with lasso mode
        fig.update_layout(
            dragmode="lasso",
            clickmode="event+select",
            selectdirection="any",
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

        # Add background points with hover info
        all_tracks_df = self.features_df[
            self.features_df["fov_name"].isin(self.fov_tracks.keys())
        ]

        # Subsample background points if there are too many
        if len(all_tracks_df) > 5000:  # Adjust this threshold as needed
            all_tracks_df = all_tracks_df.sample(n=5000, random_state=42)

        fig.add_trace(
            go.Scattergl(
                x=all_tracks_df[x_axis],
                y=all_tracks_df[y_axis],
                mode="markers",
                marker=dict(size=12, color="lightgray", opacity=0.3),
                name=f"Other points (showing {len(all_tracks_df)} of {len(self.features_df)} points)",
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

        # Add time-colored points using Scattergl
        fig.add_trace(
            go.Scattergl(
                x=self.filtered_features_df[x_axis],
                y=self.filtered_features_df[y_axis],
                mode="markers",
                marker=dict(
                    size=10,  # Reduced size
                    color=self.filtered_features_df["t"],
                    colorscale="Viridis",
                    colorbar=dict(title="Time"),
                ),
                text=[
                    f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                    for track_id, t, fov in zip(
                        self.filtered_features_df["track_id"],
                        self.filtered_features_df["t"],
                        self.filtered_features_df["fov_name"],
                    )
                ],
                hoverinfo="text",
                showlegend=False,
                hoverlabel=dict(namelength=-1),  # Show full text in hover
            )
        )

        # Add arrows if requested, but more efficiently
        if show_arrows:
            for track_id in self.filtered_features_df["track_id"].unique():
                track_data = self.filtered_features_df[
                    self.filtered_features_df["track_id"] == track_id
                ].sort_values("t")

                if len(track_data) > 1:
                    # Calculate distances between consecutive points
                    x_coords = track_data[x_axis].values
                    y_coords = track_data[y_axis].values
                    distances = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)

                    # Only show arrows for movements larger than the median distance
                    threshold = np.median(distances) * 0.5

                    # Add arrows as a single trace
                    arrow_x = []
                    arrow_y = []

                    for i in range(len(track_data) - 1):
                        if distances[i] > threshold:
                            arrow_x.extend([x_coords[i], x_coords[i + 1], None])
                            arrow_y.extend([y_coords[i], y_coords[i + 1], None])

                    if arrow_x:  # Only add if there are arrows to show
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

        # Compute axis ranges to ensure equal aspect ratio
        all_x_data = self.filtered_features_df[x_axis]
        all_y_data = self.filtered_features_df[y_axis]
        if not all_x_data.empty and not all_y_data.empty:
            x_range, y_range = self._calculate_equal_aspect_ranges(
                all_x_data, all_y_data
            )

            # Set equal aspect ratio and range
            fig.update_layout(
                xaxis=dict(
                    range=x_range, scaleanchor="y", scaleratio=1, constrain="domain"
                ),
                yaxis=dict(range=y_range, constrain="domain"),
            )

        return fig

    @staticmethod
    def _normalize_image(img_array):
        """Normalize a single image array to [0, 255] more efficiently"""
        min_val = img_array.min()
        max_val = img_array.max()
        if min_val == max_val:
            return np.zeros_like(img_array, dtype=np.uint8)
        # Normalize in one step
        return ((img_array - min_val) * 255 / (max_val - min_val)).astype(np.uint8)

    @staticmethod
    def _numpy_to_base64(img_array):
        """Convert numpy array to base64 string with compression"""
        if not isinstance(img_array, np.uint8):
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        # Use JPEG format with quality=85 for better compression
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode(
            "utf-8"
        )

    def save_cache(self, cache_path: str | None = None):
        """Save the image cache to disk using pickle.

        Parameters
        ----------
        cache_path : str | None, optional
            Path to save the cache. If None, uses self.cache_path, by default None
        """
        import pickle

        if cache_path is None:
            if self.cache_path is None:
                logger.warning("No cache path specified, skipping cache save")
                return
            cache_path = self.cache_path
        else:
            cache_path = Path(cache_path)

        # Create parent directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cache metadata for validation
        cache_metadata = {
            "data_path": str(self.data_path),
            "tracks_path": str(self.tracks_path),
            "features_path": str(self.features_path),
            "channels": self.channels_to_display,
            "z_range": self.z_range,
            "yx_patch_size": self.yx_patch_size,
            "cache_size": len(self.image_cache),
        }

        try:
            logger.info(f"Saving image cache to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump((cache_metadata, self.image_cache), f)
            logger.info(f"Successfully saved cache with {len(self.image_cache)} images")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_cache(self, cache_path: str | None = None) -> bool:
        """Load the image cache from disk using pickle.

        Parameters
        ----------
        cache_path : str | None, optional
            Path to load the cache from. If None, uses self.cache_path, by default None

        Returns
        -------
        bool
            True if cache was successfully loaded, False otherwise
        """
        import pickle

        if cache_path is None:
            if self.cache_path is None:
                logger.warning("No cache path specified, skipping cache load")
                return False
            cache_path = self.cache_path
        else:
            cache_path = Path(cache_path)

        if not cache_path.exists():
            logger.warning(f"Cache file {cache_path} does not exist")
            return False

        try:
            logger.info(f"Loading image cache from {cache_path}")
            with open(cache_path, "rb") as f:
                cache_metadata, loaded_cache = pickle.load(f)

            # Validate cache metadata
            if (
                cache_metadata["data_path"] != str(self.data_path)
                or cache_metadata["tracks_path"] != str(self.tracks_path)
                or cache_metadata["features_path"] != str(self.features_path)
                or cache_metadata["channels"] != self.channels_to_display
                or cache_metadata["z_range"] != self.z_range
                or cache_metadata["yx_patch_size"] != self.yx_patch_size
            ):
                logger.warning("Cache metadata mismatch, skipping cache load")
                return False

            self.image_cache = loaded_cache
            logger.info(
                f"Successfully loaded cache with {len(self.image_cache)} images"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False

    def preload_images(self):
        """Preload all images into memory"""
        # Try to load from cache first
        if self.cache_path and self.load_cache():
            return

        logger.info("Preloading images into cache...")
        logger.info(f"FOVs to process: {list(self.filtered_tracks_by_fov.keys())}")

        # Process each FOV and its tracks
        for fov_name, track_ids in self.filtered_tracks_by_fov.items():
            if not track_ids:  # Skip FOVs with no tracks
                logger.info(f"Skipping FOV {fov_name} as it has no tracks")
                continue

            logger.info(f"Processing FOV {fov_name} with tracks {track_ids}")

            try:
                data_module = TripletDataModule(
                    data_path=self.data_path,
                    tracks_path=self.tracks_path,
                    include_fov_names=[fov_name] * len(track_ids),
                    include_track_ids=track_ids,
                    source_channel=self.channels_to_display,
                    z_range=self.z_range,
                    initial_yx_patch_size=self.yx_patch_size,
                    final_yx_patch_size=self.yx_patch_size,
                    batch_size=1,
                    num_workers=self.num_loading_workers,
                    normalizations=None,
                    predict_cells=True,
                )
                data_module.setup("predict")

                for batch in data_module.predict_dataloader():
                    try:
                        images = batch["anchor"].numpy()
                        indices = batch["index"]
                        track_id = indices["track_id"].tolist()
                        t = indices["t"].tolist()

                        img = np.stack(images)
                        cache_key = (fov_name, track_id[0], t[0])

                        logger.debug(f"Processing cache key: {cache_key}")

                        # Process each channel based on its type
                        processed_channels = {}
                        for idx, channel in enumerate(self.channels_to_display):
                            try:
                                if channel in ["Phase3D", "DIC", "BF"]:
                                    # For phase contrast, use the middle z-slice
                                    z_idx = (self.z_range[1] - self.z_range[0]) // 2
                                    processed = self._normalize_image(
                                        img[0, idx, z_idx]
                                    )
                                else:
                                    # For fluorescence, use max projection
                                    processed = self._normalize_image(
                                        np.max(img[0, idx], axis=0)
                                    )

                                processed_channels[channel] = self._numpy_to_base64(
                                    processed
                                )
                                logger.debug(
                                    f"Successfully processed channel {channel} for {cache_key}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error processing channel {channel} for {cache_key}: {e}"
                                )
                                continue

                        if (
                            processed_channels
                        ):  # Only store if at least one channel was processed
                            self.image_cache[cache_key] = processed_channels

                    except Exception as e:
                        logger.error(
                            f"Error processing batch for {fov_name}, track {track_id}: {e}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Error setting up data module for FOV {fov_name}: {e}")
                continue

        logger.info(f"Successfully cached {len(self.image_cache)} images")
        # Log some statistics about the cache
        cached_fovs = set(key[0] for key in self.image_cache.keys())
        cached_tracks = set((key[0], key[1]) for key in self.image_cache.keys())
        logger.info(f"Cached FOVs: {cached_fovs}")
        logger.info(f"Number of unique track-FOV combinations: {len(cached_tracks)}")

        # Save cache if path is specified
        if self.cache_path:
            self.save_cache()

    def _cleanup_cache(self):
        """Clear the image cache when the program exits"""
        logging.info("Cleaning up image cache...")
        self.image_cache.clear()

    def _get_trajectory_images_lasso(self, x_axis, y_axis, selected_data):
        """Get images of points selected by lasso"""
        if not selected_data or not selected_data.get("points"):
            return html.Div("Use the lasso tool to select points")

        # Dictionary to store points for each lasso selection
        lasso_clusters = {}

        # Track which points we've seen to avoid duplicates within clusters
        seen_points = set()

        # Process each selected point
        for point in selected_data["points"]:
            text = point["text"]
            lines = text.split("<br>")
            track_id = int(lines[0].split(": ")[1])
            t = int(lines[1].split(": ")[1])
            fov = lines[2].split(": ")[1]

            point_id = (track_id, t, fov)
            cache_key = (fov, track_id, t)

            # Skip if we don't have the image in cache
            if cache_key not in self.image_cache:
                logger.debug(f"Skipping point {point_id} as it's not in the cache")
                continue

            # Determine which curve (lasso selection) this point belongs to
            curve_number = point.get("curveNumber", 0)
            if curve_number not in lasso_clusters:
                lasso_clusters[curve_number] = []

            # Only add if we haven't seen this point in this cluster
            cluster_point_id = (curve_number, point_id)
            if cluster_point_id not in seen_points:
                seen_points.add(cluster_point_id)
                lasso_clusters[curve_number].append(
                    {
                        "track_id": track_id,
                        "t": t,
                        "fov_name": fov,
                        x_axis: point["x"],
                        y_axis: point["y"],
                    }
                )

        if not lasso_clusters:
            return html.Div("No cached images found for the selected points")

        # Create sections for each lasso selection
        cluster_sections = []
        for cluster_idx, points in lasso_clusters.items():
            cluster_df = pd.DataFrame(points)

            # Create channel rows for this cluster
            channel_rows = []
            for channel in self.channels_to_display:
                images = []
                for _, row in cluster_df.iterrows():
                    cache_key = (row["fov_name"], row["track_id"], row["t"])
                    images.append(
                        html.Div(
                            [
                                html.Img(
                                    src=self.image_cache[cache_key][channel],
                                    style={
                                        "width": "150px",
                                        "height": "150px",
                                        "margin": "5px",
                                        "border": "1px solid #ddd",
                                    },
                                ),
                                html.Div(
                                    f"Track {row['track_id']}, t={row['t']}",
                                    style={
                                        "textAlign": "center",
                                        "fontSize": "12px",
                                    },
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin": "5px",
                                "verticalAlign": "top",
                            },
                        )
                    )

                if images:  # Only add row if there are images
                    channel_rows.extend(
                        [
                            html.H5(
                                f"{channel}",
                                style={
                                    "margin": "10px 5px",
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Div(
                                images,
                                style={
                                    "overflowX": "auto",
                                    "whiteSpace": "nowrap",
                                    "padding": "10px",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "5px",
                                    "marginBottom": "20px",
                                    "backgroundColor": "#f8f9fa",
                                },
                            ),
                        ]
                    )

            if channel_rows:  # Only add cluster section if it has images
                cluster_sections.append(
                    html.Div(
                        [
                            html.H3(
                                f"Lasso Selection {cluster_idx + 1}",
                                style={
                                    "marginTop": "30px",
                                    "marginBottom": "15px",
                                    "fontSize": "24px",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #007bff",
                                    "paddingBottom": "5px",
                                },
                            ),
                            html.Div(
                                channel_rows,
                                style={
                                    "backgroundColor": "#ffffff",
                                    "padding": "15px",
                                    "borderRadius": "8px",
                                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                },
                            ),
                        ]
                    )
                )

        return html.Div(
            [
                html.H2(
                    f"Selected Points ({len(cluster_sections)} selections)",
                    style={
                        "marginBottom": "20px",
                        "fontSize": "28px",
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                    },
                ),
                html.Div(cluster_sections),
            ]
        )

    def _get_output_info_display(self) -> html.Div:
        """
        Create a display component showing the output directory information.

        Returns
        -------
        html.Div
            HTML component displaying output directory info
        """
        return html.Div(
            [
                html.H4(
                    "Output Directory",
                    style={"marginBottom": "10px", "fontSize": "16px"},
                ),
                html.Div(
                    [
                        html.Span("📁 ", style={"fontSize": "14px"}),
                        html.Span(
                            str(self.output_dir),
                            style={
                                "fontFamily": "monospace",
                                "backgroundColor": "#f8f9fa",
                                "padding": "4px 8px",
                                "borderRadius": "4px",
                                "border": "1px solid #dee2e6",
                                "fontSize": "12px",
                            },
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    "CSV files will be saved to this directory with timestamped names.",
                    style={
                        "fontSize": "12px",
                        "color": "#6c757d",
                        "fontStyle": "italic",
                    },
                ),
            ],
            style={
                "backgroundColor": "#e9ecef",
                "padding": "10px",
                "borderRadius": "6px",
                "marginBottom": "15px",
                "border": "1px solid #ced4da",
            },
        )

    def _get_cluster_images(self):
        """Display images for all clusters in a grid layout"""
        if not self.clusters:
            return html.Div(
                [self._get_output_info_display(), html.Div("No clusters created yet")]
            )

        # Create cluster colors once
        cluster_colors = [
            f"rgb{tuple(int(x * 255) for x in plt.cm.Set2(i % 8)[:3])}"
            for i in range(len(self.clusters))
        ]

        # Create individual cluster panels
        cluster_panels = []
        for cluster_idx, cluster_points in enumerate(self.clusters):
            # Get cluster name or use default
            cluster_name = self.cluster_names.get(
                cluster_idx, f"Cluster {cluster_idx + 1}"
            )

            # Create a single scrollable container for all channels
            all_channel_images = []
            for channel in self.channels_to_display:
                images = []
                for point in cluster_points:
                    cache_key = (point["fov_name"], point["track_id"], point["t"])

                    images.append(
                        html.Div(
                            [
                                html.Img(
                                    src=self.image_cache[cache_key][channel],
                                    style={
                                        "width": "100px",
                                        "height": "100px",
                                        "margin": "2px",
                                        "border": f"2px solid {cluster_colors[cluster_idx]}",
                                        "borderRadius": "4px",
                                    },
                                ),
                                html.Div(
                                    f"Track {point['track_id']}, t={point['t']}",
                                    style={
                                        "textAlign": "center",
                                        "fontSize": "10px",
                                    },
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin": "2px",
                                "verticalAlign": "top",
                            },
                        )
                    )

                if images:
                    all_channel_images.extend(
                        [
                            html.H6(
                                f"{channel}",
                                style={
                                    "margin": "5px",
                                    "fontSize": "12px",
                                    "fontWeight": "bold",
                                    "position": "sticky",
                                    "left": "0",
                                    "backgroundColor": "#f8f9fa",
                                    "zIndex": "1",
                                    "paddingLeft": "5px",
                                },
                            ),
                            html.Div(
                                images,
                                style={
                                    "whiteSpace": "nowrap",
                                    "marginBottom": "10px",
                                },
                            ),
                        ]
                    )

            if all_channel_images:
                # Create a panel for this cluster with synchronized scrolling
                cluster_panels.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        cluster_name,
                                        style={
                                            "color": cluster_colors[cluster_idx],
                                            "fontWeight": "bold",
                                            "fontSize": "16px",
                                        },
                                    ),
                                    html.Span(
                                        f" ({len(cluster_points)} points)",
                                        style={
                                            "color": "#2c3e50",
                                            "fontSize": "14px",
                                        },
                                    ),
                                    html.Button(
                                        "✏️",
                                        id={
                                            "type": "edit-cluster-name",
                                            "index": cluster_idx,
                                        },
                                        style={
                                            "backgroundColor": "transparent",
                                            "border": "none",
                                            "cursor": "pointer",
                                            "fontSize": "12px",
                                            "marginLeft": "5px",
                                            "color": "#6c757d",
                                        },
                                        title="Edit cluster name",
                                    ),
                                ],
                                style={
                                    "marginBottom": "10px",
                                    "borderBottom": f"2px solid {cluster_colors[cluster_idx]}",
                                    "paddingBottom": "5px",
                                    "position": "sticky",
                                    "top": "0",
                                    "backgroundColor": "white",
                                    "zIndex": "1",
                                },
                            ),
                            html.Div(
                                all_channel_images,
                                style={
                                    "overflowX": "auto",
                                    "overflowY": "auto",
                                    "height": "400px",
                                    "backgroundColor": "#ffffff",
                                    "padding": "10px",
                                    "borderRadius": "8px",
                                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                },
                            ),
                        ],
                        style={
                            "width": "24%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "5px",
                            "boxSizing": "border-box",
                        },
                    )
                )

        # Create rows of 4 panels each
        rows = []
        for i in range(0, len(cluster_panels), 4):
            row = html.Div(
                cluster_panels[i : i + 4],
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "gap": "10px",
                    "marginBottom": "10px",
                },
            )
            rows.append(row)

        return html.Div(
            [
                html.H2(
                    [
                        "Clusters ",
                        html.Span(
                            f"({len(self.clusters)} total)",
                            style={"color": "#666"},
                        ),
                    ],
                    style={
                        "marginBottom": "20px",
                        "fontSize": "28px",
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                    },
                ),
                self._get_output_info_display(),
                html.Div(
                    rows,
                    style={
                        "maxHeight": "calc(100vh - 200px)",
                        "overflowY": "auto",
                        "padding": "10px",
                    },
                ),
            ]
        )

    def get_output_dir(self) -> Path:
        """
        Get the output directory for saving files.

        Returns
        -------
        Path
            The output directory path
        """
        return self.output_dir

    def save_clusters_to_csv(self, output_path: str | None = None) -> str:
        """
        Save cluster information to CSV file.

        This method exports all cluster data including track_id, time, FOV,
        cluster assignment, and cluster names to a CSV file for further analysis.

        Parameters
        ----------
        output_path : str | None, optional
            Path to save the CSV file. If None, generates a timestamped filename
            in the output directory, by default None

        Returns
        -------
        str
            Path to the saved CSV file

        Notes
        -----
        The CSV will contain columns:
        - cluster_id: The cluster number (1-indexed)
        - cluster_name: The custom name assigned to the cluster
        - track_id: The track identifier
        - time: The timepoint
        - fov_name: The field of view name
        - cluster_size: Number of points in the cluster
        """
        if not self.clusters:
            logger.warning("No clusters to save")
            return ""

        # Prepare data for CSV export
        csv_data = []
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_id = cluster_idx + 1  # 1-indexed for user-friendly output
            cluster_size = len(cluster)
            cluster_name = self.cluster_names.get(cluster_idx, f"Cluster {cluster_id}")

            for point in cluster:
                csv_data.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_name,
                        "track_id": point["track_id"],
                        "time": point["t"],
                        "fov_name": point["fov_name"],
                        "cluster_size": cluster_size,
                    }
                )

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)

        if output_path is None:
            # Generate timestamped filename in output directory
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"clusters_{timestamp}.csv"
        else:
            output_path = Path(output_path)
            # If only filename is provided, use output directory
            if not output_path.parent.name:
                output_path = self.output_dir / output_path.name

        try:
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(df)} cluster points to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error saving clusters to CSV: {e}")
            raise

    def run(self, debug=False, port=None):
        """Run the Dash server

        Parameters
        ----------
        debug : bool, optional
            Whether to run in debug mode, by default False
        port : int, optional
            Port to run on. If None, will try ports from 8050-8070, by default None
        """
        import socket

        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return False
                except socket.error:
                    return True

        if port is None:
            # Try ports from 8050 to 8070
            # FIXME: set a range for the ports
            port_range = list(range(8050, 8071))
            for p in port_range:
                if not is_port_in_use(p):
                    port = p
                    break
            if port is None:
                raise RuntimeError(
                    f"Could not find an available port in range {port_range[0]}-{port_range[-1]}"
                )

        try:
            logger.info(f"Starting server on port {port}")
            self.app.run(
                debug=debug,
                port=port,
                use_reloader=False,  # Disable reloader to prevent multiple instances
            )
        except KeyboardInterrupt:
            logger.info("Server shutdown requested...")
        except Exception as e:
            logger.error(f"Error running server: {e}")
        finally:
            self._cleanup_cache()
            logger.info("Server shutdown complete")
