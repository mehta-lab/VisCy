import json
import logging
from pathlib import Path
from typing import Union
import atexit

import dash
import dash.dependencies as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import (
    compute_pca,
)
from viscy.representation.visualization.cluster import (
    ClusterManager,
)
from viscy.representation.visualization.settings import VizConfig

logger = logging.getLogger("lightning.pytorch")


class EmbeddingVisualizationApp:
    def __init__(
        self,
        viz_config: VizConfig,
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
        viz_config: VizConfig
            Configuration object for visualization.
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
        self.viz_config = viz_config
        self.image_cache = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.app = None
        self.features_df: pd.DataFrame | None = None
        self.fig = None
        self.num_loading_workers = num_loading_workers

        # Initialize cluster storage before preparing data and creating figure
        self.cluster_manager = ClusterManager()

        # Store datasets for per-dataset access
        self.datasets = viz_config.get_datasets()
        self._DEFAULT_MARKER_SIZE = 15

        # Initialize data
        self._prepare_data()
        self._create_figure()
        self._init_app()
        atexit.register(self._cleanup_cache)

    def _prepare_data(self):
        """Load and prepare the data for visualization"""
        # Load features from all datasets and extract raw embeddings
        all_features_dfs = []
        all_raw_embeddings = []

        for dataset_name, dataset_config in self.datasets.items():
            logger.info(f"Loading features from dataset: {dataset_name}")
            embedding_dataset = read_embedding_dataset(
                Path(dataset_config.features_path)
            )

            # Extract raw features/embeddings
            raw_features = embedding_dataset["features"].values
            all_raw_embeddings.append(raw_features)

            # Extract metadata
            features = embedding_dataset["features"]
            features_df = features["sample"].to_dataframe().reset_index(drop=True)

            # Add dataset identifier
            features_df["dataset"] = dataset_name

            all_features_dfs.append(features_df)

        # Combine all features dataframes (metadata)
        self.features_df = pd.concat(all_features_dfs, axis=0, ignore_index=True)

        # Concatenate all raw embeddings
        combined_embeddings = np.concatenate(all_raw_embeddings, axis=0)
        logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")

        # Check if dimensionality reduction columns already exist
        existing_dims = []
        dim_options = []

        # Always recompute PCA on combined embeddings for multi-dataset scenario
        if len(self.datasets) > 1 or not any(
            col.startswith("PCA") for col in self.features_df.columns
        ):
            logger.info(
                f"Computing PCA with {self.viz_config.num_PC_components} components on combined embeddings"
            )

            # Use the compute_pca function
            pca_coords, _ = compute_pca(
                combined_embeddings,
                n_components=self.viz_config.num_PC_components,
                normalize_features=True,
            )

            # We need to get the explained variance separately since compute_pca doesn't return the model
            # FIXME: ideally the compute_pca function should return the model
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(combined_embeddings)
            pca_model = PCA(
                n_components=self.viz_config.num_PC_components, random_state=42
            )
            pca_model.fit(scaled_features)

            # Store explained variance for PCA labels
            self.pca_explained_variance = [
                f"PC{i + 1} ({var:.1f}%)"
                for i, var in enumerate(pca_model.explained_variance_ratio_ * 100)
            ]

            # Add PCA coordinates to the features dataframe
            for i in range(self.viz_config.num_PC_components):
                self.features_df[f"PCA{i + 1}"] = pca_coords[:, i]

            # Add PCA options to dropdown
            for i, pc_label in enumerate(self.pca_explained_variance):
                dim_options.append({"label": pc_label, "value": f"PCA{i + 1}"})
                existing_dims.append(f"PCA{i + 1}")

        # Compute PHATE if specified in config
        if self.viz_config.num_phate_components is not None:
            logger.info(
                f"Computing PHATE with {self.viz_config.num_phate_components} components on combined embeddings"
            )

            try:
                from viscy.representation.evaluation.dimensionality_reduction import (
                    compute_phate,
                )

                # Use the compute_phate function
                # TODO: make knn and decay configurable and the gamma parameter
                phate_model, phate_coords = compute_phate(
                    combined_embeddings,
                    n_components=self.viz_config.num_phate_components,
                    knn=5,  # Default parameters - could be made configurable
                    decay=40,
                    random_state=42,
                )

                # Add PHATE coordinates to the features dataframe
                for i in range(self.viz_config.num_phate_components):
                    self.features_df[f"PHATE{i + 1}"] = phate_coords[:, i]
                    dim_options.append(
                        {"label": f"PHATE{i + 1}", "value": f"PHATE{i + 1}"}
                    )
                    existing_dims.append(f"PHATE{i + 1}")

                logger.info(
                    f"Successfully computed PHATE with {self.viz_config.num_phate_components} components"
                )

            except ImportError:
                logger.warning(
                    "PHATE is not available. Install with: pip install viscy[phate]"
                )
            except Exception as e:
                logger.warning(f"PHATE computation failed: {str(e)}")

        # Check for existing UMAP coordinates (if they exist in the original data)
        umap_dims = [col for col in self.features_df.columns if col.startswith("UMAP")]
        if umap_dims:
            for dim in umap_dims:
                dim_options.append({"label": dim, "value": dim})
                existing_dims.append(dim)

        # Store dimension options for dropdowns
        self.dim_options = dim_options

        # Set default x and y axes based on available dimensions
        # TODO: hardcoding to default to PCA1 and PCA2
        self.default_x = existing_dims[0] if existing_dims else "PCA1"
        self.default_y = existing_dims[1] if len(existing_dims) > 1 else "PCA2"

        # Collect all valid (dataset, fov, track) combinations
        self.valid_combinations = []

        for dataset_name, dataset_config in self.datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"  fov_tracks: {dataset_config.fov_tracks}")

            for fov_name, track_ids in dataset_config.fov_tracks.items():
                if track_ids == "all":
                    # Get all tracks for this dataset/FOV combination
                    fov_tracks = self.features_df[
                        (self.features_df["dataset"] == dataset_name)
                        & (self.features_df["fov_name"] == fov_name)
                    ]["track_id"].unique()

                    logger.info(
                        f"  FOV {fov_name}: found {len(fov_tracks)} tracks for 'all'"
                    )

                    for track_id in fov_tracks:
                        self.valid_combinations.append(
                            (dataset_name, fov_name, track_id)
                        )
                else:
                    logger.info(f"  FOV {fov_name}: using specific tracks {track_ids}")
                    for track_id in track_ids:
                        self.valid_combinations.append(
                            (dataset_name, fov_name, track_id)
                        )

        logger.info(f"Total valid combinations: {len(self.valid_combinations)}")

        # Create a MultiIndex for efficient filtering
        if self.valid_combinations:
            # Create temporary column for filtering
            self.features_df["_temp_combo"] = list(
                zip(
                    self.features_df["dataset"],
                    self.features_df["fov_name"],
                    self.features_df["track_id"],
                )
            )

            # Create mask for selected data
            selected_mask = self.features_df["_temp_combo"].isin(
                self.valid_combinations
            )

            # Apply mask FIRST, then drop the temporary column
            filtered_df_with_temp = self.features_df[selected_mask].copy()
            background_df_with_temp = self.features_df[~selected_mask].copy()

            # Now drop the temporary column from the filtered dataframes
            self.filtered_features_df = filtered_df_with_temp.drop(
                "_temp_combo", axis=1
            )
            self.background_features_df = background_df_with_temp.drop(
                "_temp_combo", axis=1
            )

            # Subsample background points if there are too many
            if len(self.background_features_df) > 5000:
                self.background_features_df = self.background_features_df.sample(
                    n=5000, random_state=42
                )

            # Pre-compute track colors
            cmap = plt.cm.get_cmap("tab20")
            self.track_colors = {
                track_key: f"rgb{tuple(int(x*255) for x in cmap(i % 20)[:3])}"
                for i, track_key in enumerate(self.valid_combinations)
            }

            # Drop the temporary column from the original dataframe
            self.features_df = self.features_df.drop("_temp_combo", axis=1)
        else:
            self.filtered_features_df = pd.DataFrame()
            self.background_features_df = pd.DataFrame()
            self.track_colors = {}

        logger.info(
            f"Prepared data with {len(self.features_df)} total samples, "
            f"{len(self.filtered_features_df)} filtered samples, "
            f"and {len(self.background_features_df)} background samples, "
            f"with {len(self.valid_combinations)} unique tracks"
        )

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
            [
                dd.Output("track-timeline", "children"),
                dd.Output("scatter-plot", "figure", allow_duplicate=True),
            ],
            [dd.Input("scatter-plot", "clickData")],
            [
                dd.State("color-mode", "value"),
                dd.State("show-arrows", "value"),
                dd.State("x-axis", "value"),
                dd.State("y-axis", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_track_timeline(clickData, color_mode, show_arrows, x_axis, y_axis):
            """Update the track timeline based on the clicked point"""
            if clickData is None or self.features_df is None:
                return (
                    html.Div("Click on a point to see the track timeline"),
                    dash.no_update,
                )

            # Parse the hover text to get dataset, track_id, time and fov_name
            hover_text = clickData["points"][0]["text"]
            lines = hover_text.split("<br>")
            dataset_name = lines[0].split(": ")[1]
            track_id = int(lines[1].split(": ")[1])
            clicked_time = int(lines[2].split(": ")[1])
            fov_name = lines[3].split(": ")[1]
            # Get channels specific to this dataset
            channels_to_display = self.datasets[dataset_name].channels_to_display

            # Get all timepoints for this track
            track_data = self.features_df[
                (self.features_df["dataset"] == dataset_name)
                & (self.features_df["fov_name"] == fov_name)
                & (self.features_df["track_id"] == int(track_id))
            ]

            if track_data.empty:
                return (
                    html.Div(
                        f"No data found for track {track_id} in dataset {dataset_name}"
                    ),
                    dash.no_update,
                )

            # Sort by time
            track_data = track_data.sort_values("t")
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
                    "fontSize": "20px" if is_clicked else "14px",
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
            for channel in channels_to_display:
                channel_images = []
                for t in timepoints:
                    # Use correct 4-tuple cache key format
                    cache_key = (dataset_name, fov_name, int(track_id), int(t))

                    if (
                        cache_key in self.image_cache
                        and channel in self.image_cache[cache_key]
                    ):
                        is_clicked = t == clicked_time
                        image_style = {
                            "width": "150px",
                            "height": "150px",
                            "border": "1px solid #ddd",
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

            # Create the main container
            timeline_content = html.Div(
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

            # Create updated figure with highlighted clicked point
            show_arrows = len(show_arrows or []) > 0
            if color_mode == "track":
                fig = self._create_track_colored_figure(show_arrows, x_axis, y_axis)
            else:
                fig = self._create_time_colored_figure(show_arrows, x_axis, y_axis)

            return timeline_content, dash.no_update

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

                    # Get current cluster name using the manager
                    cluster = self.cluster_manager.get_cluster_by_index(cluster_idx)
                    current_name = (
                        cluster.name if cluster else f"Cluster {cluster_idx + 1}"
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
                        dash.no_update,
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

            # Handle clear clusters button
            elif button_id == "clear-clusters" and clear_clicks:
                self.cluster_manager.clear_all_clusters()
                logger.info("Cleared all clusters")

                # Update figure to remove cluster coloring
                if color_mode == "track":
                    fig = self._create_track_colored_figure(show_arrows, x_axis, y_axis)
                else:
                    fig = self._create_time_colored_figure(show_arrows, x_axis, y_axis)

                return (
                    {"display": "none"},  # Hide clusters tab
                    html.Div("No clusters created yet"),
                    "timeline-tab",  # Switch back to timeline tab
                    fig,
                    {"display": "none"},  # Hide modal
                    "",
                    None,  # Clear selection
                )

            # Handle save cluster name button
            elif button_id == "save-cluster-name" and save_name_clicks and cluster_name:
                # Get the most recent cluster and update its name
                if self.cluster_manager.clusters:
                    latest_cluster = self.cluster_manager.clusters[-1]
                    latest_cluster.name = cluster_name
                    logger.info(
                        f"Named cluster {latest_cluster.id} as '{cluster_name}'"
                    )

                # Close modal and update clusters display
                modal_style = {"display": "none"}
                return (
                    {"display": "block"},
                    self._get_cluster_images(),
                    "clusters-tab",
                    dash.no_update,
                    modal_style,
                    "",
                    dash.no_update,
                )

            # Handle cancel cluster name button
            elif button_id == "cancel-cluster-name" and cancel_name_clicks:
                # Just close the modal without saving
                modal_style = {"display": "none"}
                return (
                    {"display": "block"},
                    self._get_cluster_images(),
                    "clusters-tab",
                    dash.no_update,
                    modal_style,
                    "",
                    dash.no_update,
                )

            # Handle assign cluster button
            elif button_id == "assign-cluster" and assign_clicks and selected_data:
                if not selected_data or not selected_data.get("points"):
                    logger.warning("No points selected for clustering")
                    return (
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                    )

                # Extract selected point indices
                selected_indices = [
                    point.get("pointIndex") for point in selected_data["points"]
                ]
                selected_indices = [idx for idx in selected_indices if idx is not None]

                if not selected_indices:
                    logger.warning("No valid point indices found in selection")
                    return (
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                    )

                # Get the selected data points from filtered_features_df
                if len(selected_indices) > len(self.filtered_features_df):
                    logger.error("Selected indices exceed dataframe size")
                    return (
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                    )

                # Create cluster from selected points
                selected_points_df = self.filtered_features_df.iloc[
                    selected_indices
                ].copy()

                # Create a new cluster using the cluster manager
                cluster_id = self.cluster_manager.create_cluster(selected_points_df)
                logger.info(
                    f"Created cluster {cluster_id} with {len(selected_points_df)} points"
                )

                # Show modal for naming the cluster
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

                # Update figure to show cluster coloring
                fig = self._create_cluster_colored_figure(show_arrows, x_axis, y_axis)

                return (
                    {"display": "block"},  # Show clusters tab
                    self._get_cluster_images(),
                    "clusters-tab",  # Switch to clusters tab
                    fig,  # Updated figure with cluster colors
                    modal_style,  # Show naming modal
                    f"Cluster {len(self.cluster_manager.clusters)}",  # Default name
                    None,  # Clear selection
                )

            # Default return (no updates)
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        @self.app.callback(
            [
                dd.Output("scatter-plot", "figure", allow_duplicate=True),
                dd.Output("scatter-plot", "selectedData", allow_duplicate=True),
                dd.Output("track-timeline", "children", allow_duplicate=True),
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
                    uirevision=None,  # Reset UI state to clear selections
                    selectdirection="any",
                )

                # Clear the track timeline as well
                empty_timeline = html.Div(
                    "Click on a point to see the track timeline",
                    style={
                        "textAlign": "center",
                        "color": "#666",
                        "fontSize": "16px",
                        "padding": "40px",
                        "fontStyle": "italic",
                    },
                )

                return (
                    fig,
                    None,
                    empty_timeline,
                )  # Clear figure selection, selectedData, and timeline
            return dash.no_update, dash.no_update, dash.no_update

    def _create_track_colored_figure(
        self,
        show_arrows=False,
        x_axis=None,
        y_axis=None,
    ):
        """Create scatter plot with track-based coloring"""
        if self.filtered_features_df is None or self.filtered_features_df.empty:
            return go.Figure()

        x_axis = x_axis or self.default_x
        y_axis = y_axis or self.default_y

        fig = go.Figure()

        # Set initial layout with lasso mode
        fig.update_layout(
            dragmode="lasso",
            clickmode="event+select",
            selectdirection="any",
            plot_bgcolor="white",
            title="Embedding Visualization",
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
        fig.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)

        # Use pre-computed filtered and background data
        filtered_features_df = self.filtered_features_df

        # Add background points using pre-computed background data
        if not self.background_features_df.empty:
            fig.add_trace(
                go.Scattergl(
                    x=self.background_features_df[x_axis],
                    y=self.background_features_df[y_axis],
                    mode="markers",
                    marker=dict(size=8, color="lightgray", opacity=0.4),
                    name=f"Other tracks ({len(self.background_features_df)} points)",
                    text=[
                        f"Dataset: {dataset}<br>Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for dataset, track_id, t, fov in zip(
                            self.background_features_df["dataset"],
                            self.background_features_df["track_id"],
                            self.background_features_df["t"],
                            self.background_features_df["fov_name"],
                        )
                    ],
                    hoverinfo="text",
                    showlegend=True,
                    hoverlabel=dict(namelength=-1),
                )
            )

        # Use pre-computed unique track keys and colors
        # Add points for each selected track with cluster coloring
        for dataset_name, fov_name, track_id in self.valid_combinations:
            track_data = filtered_features_df[
                (filtered_features_df["dataset"] == dataset_name)
                & (filtered_features_df["fov_name"] == fov_name)
                & (filtered_features_df["track_id"] == int(track_id))
            ]

            if track_data.empty:
                logger.warning(
                    f"No data found for track {track_id} in dataset {dataset_name} and fov {fov_name}"
                )
                continue

            # Sort by time
            if hasattr(track_data, "sort_values"):
                track_data = track_data.sort_values("t")
            timepoints = track_data["t"].unique()

            # Add track points
            fig.add_trace(
                go.Scattergl(
                    x=track_data[x_axis],
                    y=track_data[y_axis],
                    mode="markers",
                    marker=dict(
                        size=self._DEFAULT_MARKER_SIZE,  # Use variable sizes
                        color=self.track_colors[(dataset_name, fov_name, track_id)],
                        opacity=1.0,
                        line=dict(width=0.5, color="black"),
                    ),
                    name=f"{dataset_name}:{track_id}",
                    text=[
                        f"Dataset: {dataset_name}<br>Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for t, fov in zip(track_data["t"], track_data["fov_name"])
                    ],
                    hoverinfo="text",
                    unselected=dict(
                        marker=dict(opacity=0.6, size=self._DEFAULT_MARKER_SIZE)
                    ),
                    selected=dict(
                        marker=dict(size=self._DEFAULT_MARKER_SIZE * 1.5, opacity=1.0)
                    ),
                    hoverlabel=dict(namelength=-1),
                )
            )

            # Add trajectory lines and arrows if requested
            if show_arrows and len(track_data) > 1:
                x_coords = track_data[x_axis].values
                y_coords = track_data[y_axis].values
                track_key = (dataset_name, fov_name, track_id)

                # Add dashed lines for the trajectory
                fig.add_trace(
                    go.Scattergl(
                        x=x_coords,
                        y=y_coords,
                        mode="lines",
                        line=dict(
                            color=self.track_colors[track_key],
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

                    # Only add arrow if there's significant movement
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
                            arrowcolor=self.track_colors[track_key],
                            opacity=0.8,
                        )

        return fig

    def _create_time_colored_figure(
        self,
        show_arrows=False,
        x_axis=None,
        y_axis=None,
    ):
        """Create scatter plot with time-based coloring"""
        if self.filtered_features_df is None or self.filtered_features_df.empty:
            return go.Figure()

        x_axis = x_axis or self.default_x
        y_axis = y_axis or self.default_y

        fig = go.Figure()

        # Set initial layout with lasso mode
        fig.update_layout(
            dragmode="lasso",
            clickmode="event+select",
            selectdirection="any",
            plot_bgcolor="white",
            title="Embedding Visualization",
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
                bordercolor="Black",
                borderwidth=1,
            ),
            margin=dict(l=50, r=150, t=50, b=50),
            autosize=True,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)

        # Use pre-computed filtered and background data
        filtered_features_df = self.filtered_features_df

        # Add background points using pre-computed background data
        if not self.background_features_df.empty:
            fig.add_trace(
                go.Scattergl(
                    x=self.background_features_df[x_axis],
                    y=self.background_features_df[y_axis],
                    mode="markers",
                    marker=dict(size=8, color="lightgray", opacity=0.3),
                    name=f"Other points ({len(self.background_features_df)} points)",
                    text=[
                        f"Dataset: {dataset}<br>Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for dataset, track_id, t, fov in zip(
                            self.background_features_df["dataset"],
                            self.background_features_df["track_id"],
                            self.background_features_df["t"],
                            self.background_features_df["fov_name"],
                        )
                    ],
                    hoverinfo="text",
                    hoverlabel=dict(namelength=-1),
                )
            )
        # Add time-colored points
        if not filtered_features_df.empty:
            fig.add_trace(
                go.Scattergl(
                    x=filtered_features_df[x_axis],
                    y=filtered_features_df[y_axis],
                    mode="markers",
                    marker=dict(
                        size=self._DEFAULT_MARKER_SIZE,  # Use variable sizes
                        color=filtered_features_df["t"],
                        colorscale="Viridis",
                        colorbar=dict(title="Time"),
                    ),
                    text=[
                        f"Dataset: {dataset}<br>Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for dataset, track_id, t, fov in zip(
                            filtered_features_df["dataset"],
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
        if show_arrows and not filtered_features_df.empty:
            for dataset_name, fov_name, track_id in filtered_features_df.apply(
                lambda row: (row["dataset"], row["fov_name"], str(row["track_id"])),
                axis=1,
            ).unique():
                track_data = filtered_features_df[
                    (filtered_features_df["dataset"] == dataset_name)
                    & (filtered_features_df["fov_name"] == fov_name)
                    & (filtered_features_df["track_id"] == track_id)
                ]

                if len(track_data) <= 1:
                    continue

                # Sort by time
                if hasattr(track_data, "sort_values"):
                    track_data = track_data.sort_values("t")
                x_coords = track_data[x_axis].values
                y_coords = track_data[y_axis].values
                distances = np.sqrt(
                    np.diff(np.array(x_coords)) ** 2 + np.diff(np.array(y_coords)) ** 2
                )

                # Only show arrows for movements larger than the median distance
                threshold = np.median(distances) * 0.5 if len(distances) > 0 else 0

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

        return fig

    def _create_cluster_colored_figure(
        self,
        show_arrows=False,
        x_axis=None,
        y_axis=None,
    ):
        """Create scatter plot with cluster-based coloring"""
        if self.filtered_features_df is None or self.filtered_features_df.empty:
            return go.Figure()

        x_axis = x_axis or self.default_x
        y_axis = y_axis or self.default_y

        fig = go.Figure()

        # Set initial layout
        fig.update_layout(
            dragmode="lasso",
            showlegend=True,
            height=700,
            xaxis=dict(scaleanchor="y", scaleratio=1),  # Square plot
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Create cluster colors
        cluster_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        # Add unclustered points (background)
        clustered_indices = set()
        for cluster in self.cluster_manager.clusters:
            clustered_indices.update(cluster.point_indices)

        unclustered_mask = ~self.filtered_features_df.index.isin(clustered_indices)
        if unclustered_mask.any():
            unclustered_df = self.filtered_features_df[unclustered_mask]

            fig.add_trace(
                go.Scatter(
                    x=unclustered_df[x_axis],
                    y=unclustered_df[y_axis],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="lightgray",
                        opacity=0.6,
                    ),
                    name="Unclustered",
                    hovertemplate="<b>Unclustered</b><br>"
                    + f"{x_axis}: %{{x}}<br>"
                    + f"{y_axis}: %{{y}}<br>"
                    + "<extra></extra>",
                )
            )

        # Add each cluster as a separate trace
        for i, cluster in enumerate(self.cluster_manager.clusters):
            cluster_df = self.filtered_features_df.loc[cluster.point_indices]
            color = cluster_colors[i % len(cluster_colors)]

            fig.add_trace(
                go.Scatter(
                    x=cluster_df[x_axis],
                    y=cluster_df[y_axis],
                    mode="markers",
                    marker=dict(
                        size=self._DEFAULT_MARKER_SIZE,
                        color=color,
                        line=dict(width=0.5, color="black"),
                        opacity=0.8,
                    ),
                    name=cluster.name or f"Cluster {i+1}",
                    hovertemplate=f"<b>{cluster.name or f'Cluster {i+1}'}</b><br>"
                    + f"{x_axis}: %{{x}}<br>"
                    + f"{y_axis}: %{{y}}<br>"
                    + "<extra></extra>",
                )
            )

        return fig

    def _cleanup_cache(self):
        """Clear the image cache when the program exits"""
        logging.info("Cleaning up image cache...")
        self.image_cache.clear()

    def _get_cluster_images(self):
        """Display images for all clusters in a grid layout"""
        if not self.cluster_manager.clusters:
            return html.Div("No clusters created yet")

        # Create cluster colors once
        cmap = plt.cm.get_cmap("tab20")
        cluster_colors = [
            f"rgb{tuple(int(x*255) for x in cmap(i % 20)[:3])}"
            for i in range(len(self.cluster_manager.clusters))
        ]

        # Get all channels from any dataset (they should be the same)
        channels_to_display = []
        for dataset_config in self.datasets.values():
            if hasattr(dataset_config, "channels_to_display"):
                channels_to_display = dataset_config.channels_to_display
                break

        # Create individual cluster panels
        cluster_panels = []
        for cluster_idx, cluster in enumerate(self.cluster_manager.clusters):
            # Create a single scrollable container for all channels
            all_channel_images = []
            for channel in channels_to_display:
                images = []
                for point in cluster.points:
                    cache_key = (point.dataset, point.fov_name, point.track_id, point.t)

                    if (
                        cache_key in self.image_cache
                        and channel in self.image_cache[cache_key]
                    ):
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
                                        f"Track {point.track_id}, t={point.t}",
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
                cluster_name = (
                    cluster.name if cluster.name else f"Cluster {cluster_idx + 1}"
                )
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
                                        f" ({len(cluster.points)} points)",
                                        style={
                                            "color": "#2c3e50",
                                            "fontSize": "14px",
                                        },
                                    ),
                                    html.Button(
                                        "✏️",
                                        id={
                                            "type": "edit-cluster-name",
                                            "index": cluster.id,
                                        },
                                        style={
                                            "marginLeft": "10px",
                                            "backgroundColor": "transparent",
                                            "border": "none",
                                            "cursor": "pointer",
                                            "fontSize": "12px",
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
                            f"({len(self.cluster_manager.clusters)} total)",
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
        import base64
        from io import BytesIO

        from PIL import Image

        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        # Use JPEG format with quality=85 for better compression
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode(
            "utf-8"
        )

    def preload_images(self):
        """Preload all images into memory for all datasets"""
        from viscy.data.triplet import TripletDataModule

        # Try to load from cache first
        if self.cache_path and self.load_cache():
            return

        logger.info("Preloading images into cache...")

        # Process each dataset
        for dataset_name, dataset_config in self.datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")

            if (
                not hasattr(dataset_config, "fov_tracks")
                or not dataset_config.fov_tracks
            ):
                logger.info(f"Skipping dataset {dataset_name} as it has no FOV tracks")
                continue

            # Get tracks for this dataset
            for fov_name, tracks in dataset_config.fov_tracks.items():
                if isinstance(tracks, list):
                    track_ids = tracks
                elif tracks == "all":
                    # Get all tracks for this FOV from features
                    if self.features_df is not None:
                        fov_tracks_series = self.features_df[
                            (self.features_df["dataset"] == dataset_name)
                            & (self.features_df["fov_name"] == fov_name)
                        ]["track_id"]
                        track_ids = fov_tracks_series.unique().tolist()

            logger.info(f"FOVs to process for {dataset_name}: {list(track_ids)}")

            # Process each FOV and its tracks
            for fov_name, track_ids in dataset_config.fov_tracks.items():
                if not track_ids:  # Skip FOVs with no tracks
                    logger.info(f"Skipping FOV {fov_name} as it has no tracks")
                    continue

                logger.info(f"Processing FOV {fov_name} with tracks {track_ids}")

                try:
                    data_module = TripletDataModule(
                        data_path=dataset_config.data_path,
                        tracks_path=dataset_config.tracks_path,
                        include_fov_names=[fov_name] * len(track_ids),
                        include_track_ids=track_ids,
                        source_channel=dataset_config.channels_to_display,
                        z_range=dataset_config.z_range,
                        initial_yx_patch_size=dataset_config.yx_patch_size,
                        final_yx_patch_size=dataset_config.yx_patch_size,
                        batch_size=1,
                        num_workers=self.num_loading_workers,
                        normalizations=[],  # Use empty list instead of None
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

                            cache_key = (dataset_name, fov_name, track_id, t)

                            logger.debug(f"Processing cache key: {cache_key}")

                            # Process each channel based on its type
                            processed_channels = {}
                            for idx, channel in enumerate(
                                dataset_config.channels_to_display
                            ):
                                try:
                                    if channel in ["Phase3D", "DIC", "BF"]:
                                        # For phase contrast, use the middle z-slice
                                        z_idx = (
                                            dataset_config.z_range[1]
                                            - dataset_config.z_range[0]
                                        ) // 2
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
                    logger.error(
                        f"Error setting up data module for FOV {fov_name}: {e}"
                    )
                    continue

        logger.info(f"Successfully cached {len(self.image_cache)} images")
        # Log some statistics about the cache
        cached_datasets = set(key[0] for key in self.image_cache.keys())
        cached_fovs = set((key[0], key[1]) for key in self.image_cache.keys())
        cached_tracks = set((key[0], key[1], key[2]) for key in self.image_cache.keys())
        logger.info(f"Cached datasets: {cached_datasets}")
        logger.info(f"Cached dataset-FOV combinations: {len(cached_fovs)}")
        logger.info(
            f"Number of unique dataset-FOV-track combinations: {len(cached_tracks)}"
        )

        # Save cache if path is specified
        if self.cache_path:
            self.save_cache()

    def save_cache(self, cache_path: Union[str, Path, None] = None):
        """Save the image cache to disk using pickle.

        Parameters
        ----------
        cache_path : Union[str, Path, None], optional
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
            "datasets": {
                name: {
                    "data_path": str(config.data_path),
                    "tracks_path": str(config.tracks_path),
                    "features_path": str(config.features_path),
                    "channels": config.channels_to_display,
                    "z_range": config.z_range,
                    "yx_patch_size": config.yx_patch_size,
                }
                for name, config in self.datasets.items()
            },
            "cache_size": len(self.image_cache),
        }

        try:
            logger.info(f"Saving image cache to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump((cache_metadata, self.image_cache), f)
            logger.info(f"Successfully saved cache with {len(self.image_cache)} images")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_cache(self, cache_path: Union[str, Path, None] = None) -> bool:
        """Load the image cache from disk using pickle.

        Parameters
        ----------
        cache_path : Union[str, Path, None], optional
            Path to load the cache from. If None, uses self.cache_path
        """
        import pickle

        if cache_path is None:
            if self.cache_path is None:
                logger.warning("No cache path specified, skipping cache load")
                return False
            cache_path = self.cache_path
        else:
            cache_path = Path(cache_path)

        try:
            logger.info(f"Loading image cache from {cache_path}")
            with open(cache_path, "rb") as f:
                cache_metadata, self.image_cache = pickle.load(f)
            logger.info(
                f"Successfully loaded cache with {len(self.image_cache)} images"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False

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
            if self.app is not None:
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
