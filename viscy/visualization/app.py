"""Main visualization application."""

import atexit
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.visualization.callbacks.selection import SelectionCallbacks
from viscy.visualization.components.figures import FigureCreator
from viscy.visualization.components.scatter_plot import ScatterPlot
from viscy.visualization.components.tabs import ViewTabs
from viscy.visualization.layouts.control_panel import ControlPanel
from viscy.visualization.styles.common import CommonStyles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingVisualizationApp:
    """Interactive visualization application for embeddings."""

    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        features_path: str,
        channels_to_display: Union[List[str], str],
        fov_tracks: Dict[str, Union[List[int], str]],
        z_range: Tuple[int, int] = (0, 1),
        yx_patch_size: Tuple[int, int] = (128, 128),
        num_PC_components: int = 3,
        cache_path: Optional[str] = None,
        num_loading_workers: int = 16,
    ) -> None:
        """Initialize the visualization app.

        Parameters
        ----------
        data_path : str
            Path to the data file.
        tracks_path : str
            Path to the tracks file.
        features_path : str
            Path to the features file.
        channels_to_display : Union[List[str], str]
            Channels to display.
        fov_tracks : Dict[str, Union[List[int], str]]
            Dictionary mapping FOVs to track IDs.
        z_range : Tuple[int, int], optional
            Z-range to use, by default (0, 1)
        yx_patch_size : Tuple[int, int], optional
            YX patch size, by default (128, 128)
        num_PC_components : int, optional
            Number of PC components, by default 3
        cache_path : Optional[str], optional
            Path to cache file, by default None
        num_loading_workers : int, optional
            Number of workers for loading data, by default 16
        """
        self.data_path = Path(data_path)
        self.tracks_path = Path(tracks_path)
        self.features_path = Path(features_path)
        self.fov_tracks = fov_tracks
        self.image_cache = {}
        self.cache_path = Path(cache_path) if cache_path else None
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
        # Initialize cluster storage
        self.clusters = []  # List to store all clusters
        self.cluster_points = set()  # Set to track all points in clusters
        # Initialize data
        self._prepare_data()
        self._create_figure()
        self._init_app()
        atexit.register(self._cleanup_cache)

    def _prepare_data(self):
        """Prepare the feature data and PCA transformation."""
        embedding_dataset = read_embedding_dataset(self.features_path)
        features = embedding_dataset["features"]
        self.features_df = features["sample"].to_dataframe().reset_index(drop=True)

        # PCA transformation
        scaled_features = StandardScaler().fit_transform(features.values)
        pca = PCA(n_components=self.num_PC_components)
        pca_coords = pca.fit_transform(scaled_features)

        # Add PCA coordinates to the features dataframe
        for i in range(self.num_PC_components):
            self.features_df[f"PCA{i+1}"] = pca_coords[:, i]

        # Store explained variance for each component
        self.explained_variance = [
            f"PC{i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        ]

        # Create PC selection options with explained variance
        self.pc_options = [
            {"label": pc_label, "value": f"PCA{i+1}"}
            for i, pc_label in enumerate(self.explained_variance)
        ]

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
        """Create the initial scatter plot figure."""
        self.fig = self._create_track_colored_figure()

    def _init_app(self):
        """Initialize the Dash application."""
        self.app = dash.Dash(__name__)

        # Create components
        control_panel = ControlPanel(self.pc_options)
        scatter_plot = ScatterPlot(self.fig)

        # Create main layout
        self.app.layout = dash.html.Div(
            style=CommonStyles.get_style("container"),
            children=[
                # Add data stores
                dcc.Store(id="selected-tracks-store", storage_type="memory"),
                dcc.Store(id="cluster-store", storage_type="memory"),
                # Main content
                dash.html.H1(
                    "Track Visualization",
                    style=CommonStyles.get_style("header"),
                ),
                control_panel.create_layout(),
                scatter_plot.create_layout(),
                ViewTabs().create_layout(),
            ],
        )

        # Register callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register all callbacks."""
        selection_callbacks = SelectionCallbacks(self)
        selection_callbacks.register()

        @self.app.callback(
            [
                Output("scatter-plot", "figure"),
                Output("trajectory-images", "children"),
                Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                Input("color-mode", "value"),
                Input("show-arrows", "value"),
                Input("x-axis", "value"),
                Input("y-axis", "value"),
                Input("trajectory-mode", "value"),
                Input("selection-mode", "value"),
                Input("scatter-plot", "relayoutData"),
                Input("scatter-plot", "selectedData"),
                Input("clear-selection", "n_clicks"),
            ],
            [
                State("scatter-plot", "figure"),
            ],
            prevent_initial_call=True,
        )
        def update_figure(
            color_mode,
            show_arrows,
            x_axis,
            y_axis,
            trajectory_mode,
            selection_mode,
            relayout_data,
            selected_data,
            clear_clicks,
            current_figure,
        ):
            """Update the figure and trajectory images."""
            show_arrows = len(show_arrows or []) > 0
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Handle clear selection
            if trigger_id == "clear-selection" and clear_clicks:
                if color_mode == "track":
                    fig = self._create_track_colored_figure(
                        show_arrows, x_axis, y_axis, trajectory_mode, selection_mode
                    )
                else:
                    fig = self._create_time_colored_figure(
                        show_arrows, x_axis, y_axis, trajectory_mode, selection_mode
                    )
                return fig, html.Div("Use the lasso tool to select points"), None

            # Create new figure when necessary
            if trigger_id in [
                "color-mode",
                "show-arrows",
                "x-axis",
                "y-axis",
                "selection-mode",
            ]:
                if color_mode == "track":
                    fig = self._create_track_colored_figure(
                        show_arrows, x_axis, y_axis, trajectory_mode, selection_mode
                    )
                else:
                    fig = self._create_time_colored_figure(
                        show_arrows, x_axis, y_axis, trajectory_mode, selection_mode
                    )

                # Update dragmode and selection settings
                fig.update_layout(
                    dragmode="lasso" if selection_mode == "lasso" else "pan",
                    clickmode="event+select",
                    uirevision="true",
                    selectdirection="any",
                )
            else:
                fig = current_figure

            # Get trajectory images based on selection mode and trigger
            if trigger_id == "scatter-plot":
                if selection_mode == "region" and relayout_data:
                    trajectory_images = self._get_trajectory_images_region(
                        x_axis, y_axis, trajectory_mode, relayout_data
                    )
                elif selection_mode == "lasso" and selected_data:
                    trajectory_images = self._get_trajectory_images_lasso(
                        x_axis, y_axis, selected_data
                    )
                else:
                    trajectory_images = html.Div(
                        "Use the {} tool to select points".format(
                            "lasso" if selection_mode == "lasso" else "shaded region"
                        )
                    )
            else:
                trajectory_images = dash.no_update

            return fig, trajectory_images, selected_data

        @self.app.callback(
            [
                Output("clusters-tab", "style"),
                Output("cluster-container", "children"),
                Output("view-tabs", "value"),
            ],
            [
                Input("cluster-button", "n_clicks"),
                Input("clear-clusters", "n_clicks"),
            ],
            [
                Input("scatter-plot", "selectedData"),
                Input("eps-slider", "value"),
                Input("min-samples-slider", "value"),
            ],
        )
        def update_clusters(
            cluster_clicks, clear_clicks, selected_data, eps, min_samples
        ):
            """Update cluster display."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return {"display": "none"}, None, "trajectory-tab"

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "cluster-button" and selected_data:
                # Create new cluster from selected points
                points = selected_data["points"]
                new_cluster = []
                for point in points:
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
                    self.clusters.append(new_cluster)
                    return (
                        {"display": "block"},
                        self._get_cluster_images(),
                        "clusters-tab",
                    )

            elif button_id == "clear-clusters":
                self.clusters = []
                self.cluster_points.clear()
                return {"display": "none"}, None, "trajectory-tab"

            return dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            Output("track-timeline", "children"),
            [Input("scatter-plot", "clickData")],
            prevent_initial_call=True,
        )
        def update_track_timeline(clickData):
            """Update the track timeline with images from the clicked track."""
            if not clickData:
                return html.Div("Click on a point to see its track timeline")

            # Extract track information from the clicked point
            point = clickData["points"][0]
            text = point["text"]
            lines = text.split("<br>")
            track_id = int(lines[0].split(": ")[1])
            fov = lines[2].split(": ")[1]

            # Get all points for this track
            track_data = self.filtered_features_df[
                (self.filtered_features_df["fov_name"] == fov)
                & (self.filtered_features_df["track_id"] == track_id)
            ].sort_values("t")

            if track_data.empty:
                return html.Div(f"No timeline data found for Track {track_id}")

            # Create a single scrollable container for all channels
            timeline_content = []
            for channel in self.channels_to_display:
                channel_row = []
                for _, row in track_data.iterrows():
                    cache_key = (row["fov_name"], row["track_id"], row["t"])
                    if (
                        cache_key in self.image_cache
                        and channel in self.image_cache[cache_key]
                    ):
                        channel_row.append(
                            html.Div(
                                [
                                    html.Img(
                                        src=self.image_cache[cache_key][channel],
                                        style={
                                            "width": "150px",
                                            "height": "150px",
                                            "border": "1px solid #ddd",
                                            "borderRadius": "4px",
                                        },
                                    ),
                                    html.Div(
                                        f"t={row['t']}",
                                        style={
                                            "textAlign": "center",
                                            "fontSize": "12px",
                                            "marginTop": "5px",
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

                if channel_row:
                    timeline_content.extend(
                        [
                            html.H4(
                                channel,
                                style={
                                    "marginTop": "10px",
                                    "marginBottom": "10px",
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                    "color": "#2c3e50",
                                    "paddingLeft": "10px",
                                },
                            ),
                            html.Div(
                                channel_row,
                                style={
                                    "whiteSpace": "nowrap",
                                    "marginBottom": "20px",
                                },
                            ),
                        ]
                    )

            if not timeline_content:
                return html.Div("No images found in cache for this track")

            return html.Div(
                [
                    html.H2(
                        f"Track {track_id} Timeline",
                        style={
                            "marginBottom": "20px",
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "color": "#2c3e50",
                            "borderBottom": "2px solid #007bff",
                            "paddingBottom": "10px",
                        },
                    ),
                    html.Div(
                        timeline_content,
                        style={
                            "overflowX": "auto",
                            "overflowY": "hidden",
                            "backgroundColor": "#ffffff",
                            "padding": "20px",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        },
                    ),
                ]
            )

    def run(self, debug=False, port=None):
        """Run the Dash server.

        Parameters
        ----------
        debug : bool, optional
            Whether to run in debug mode, by default False
        port : int, optional
            Port to run on, by default None
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
            self.app.run_server(
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

    def _cleanup_cache(self):
        """Clear the image cache when the program exits."""
        logging.info("Cleaning up image cache...")
        self.image_cache.clear()

    def _create_track_colored_figure(
        self,
        show_arrows=False,
        x_axis="PCA1",
        y_axis="PCA2",
        trajectory_mode="x",
        selection_mode="region",
    ) -> go.Figure:
        """Create scatter plot with track-based coloring."""
        return FigureCreator.create_track_colored_figure(
            self.features_df,
            self.filtered_features_df,
            self.clusters,
            self.cluster_points,
            show_arrows,
            x_axis,
            y_axis,
            trajectory_mode,
            selection_mode,
        )

    def _create_time_colored_figure(
        self,
        show_arrows=False,
        x_axis="PCA1",
        y_axis="PCA2",
        trajectory_mode="x",
        selection_mode="region",
    ) -> go.Figure:
        """Create scatter plot with time-based coloring."""
        return FigureCreator.create_time_colored_figure(
            self.features_df,
            self.filtered_features_df,
            show_arrows,
            x_axis,
            y_axis,
            trajectory_mode,
            selection_mode,
        )

    def _create_view_tabs(self):
        """Create the view tabs section."""
        # TODO: Implement tab creation
        return dash.html.Div()  # Placeholder

    def preload_images(self):
        """Preload all images into memory."""
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

    def save_cache(self, cache_path: Optional[str] = None):
        """Save the image cache to disk using pickle.

        Parameters
        ----------
        cache_path : Optional[str], optional
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

    def load_cache(self, cache_path: Optional[str] = None) -> bool:
        """Load the image cache from disk using pickle.

        Parameters
        ----------
        cache_path : Optional[str], optional
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

    @staticmethod
    def _normalize_image(img_array: np.ndarray) -> np.ndarray:
        """Normalize a single image array to [0, 255].

        Parameters
        ----------
        img_array : np.ndarray
            Input image array.

        Returns
        -------
        np.ndarray
            Normalized image array.
        """
        min_val = img_array.min()
        max_val = img_array.max()
        if min_val == max_val:
            return np.zeros_like(img_array, dtype=np.uint8)
        # Normalize in one step
        return ((img_array - min_val) * 255 / (max_val - min_val)).astype(np.uint8)

    @staticmethod
    def _numpy_to_base64(img_array: np.ndarray) -> str:
        """Convert numpy array to base64 string with compression.

        Parameters
        ----------
        img_array : np.ndarray
            Input image array.

        Returns
        -------
        str
            Base64 encoded image string.
        """
        if not isinstance(img_array, np.uint8):
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        # Use JPEG format with quality=85 for better compression
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode(
            "utf-8"
        )

    def _get_trajectory_images_region(
        self, x_axis: str, y_axis: str, trajectory_mode: str, relayout_data: Dict
    ) -> html.Div:
        """Get images of points within the shaded region."""
        if not relayout_data or not any(
            key.startswith("shapes") for key in relayout_data
        ):
            return html.Div("Drag the shaded region to see points along the trajectory")

        # Extract shaded region boundaries from relayout_data
        x0_key = next((k for k in relayout_data if k.endswith(".x0")), None)
        x1_key = next((k for k in relayout_data if k.endswith(".x1")), None)
        y0_key = next((k for k in relayout_data if k.endswith(".y0")), None)
        y1_key = next((k for k in relayout_data if k.endswith(".y1")), None)

        if not all([x0_key, x1_key, y0_key, y1_key]):
            return html.Div("Drag the shaded region to see points along the trajectory")

        # Get region boundaries
        x0 = relayout_data[x0_key]
        x1 = relayout_data[x1_key]
        y0 = relayout_data[y0_key]
        y1 = relayout_data[y1_key]

        # Find points within the rectangular region
        mask = (
            (self.filtered_features_df[x_axis] >= min(x0, x1))
            & (self.filtered_features_df[x_axis] <= max(x0, x1))
            & (self.filtered_features_df[y_axis] >= min(y0, y1))
            & (self.filtered_features_df[y_axis] <= max(y0, y1))
        )

        # Sort points based on trajectory mode
        sort_by = y_axis if trajectory_mode == "x" else x_axis
        nearby_points = self.filtered_features_df[mask].sort_values(sort_by)

        if len(nearby_points) == 0:
            return html.Div("No points found in the selected region")

        # Create a single scrollable container for all channels
        timeline_content = []
        for channel in self.channels_to_display:
            channel_row = []
            for _, row in nearby_points.iterrows():
                cache_key = (row["fov_name"], row["track_id"], row["t"])
                if (
                    cache_key in self.image_cache
                    and channel in self.image_cache[cache_key]
                ):
                    channel_row.append(
                        html.Div(
                            [
                                html.Img(
                                    src=self.image_cache[cache_key][channel],
                                    style={
                                        "width": "150px",
                                        "height": "150px",
                                        "border": "1px solid #ddd",
                                        "borderRadius": "4px",
                                    },
                                ),
                                html.Div(
                                    [
                                        f"Track {row['track_id']}, t={row['t']}",
                                        html.Br(),
                                        f"{x_axis}: {row[x_axis]:.2f}",
                                        html.Br(),
                                        f"{y_axis}: {row[y_axis]:.2f}",
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "fontSize": "12px",
                                        "marginTop": "5px",
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

            if channel_row:
                timeline_content.extend(
                    [
                        html.H4(
                            channel,
                            style={
                                "marginTop": "10px",
                                "marginBottom": "10px",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "color": "#2c3e50",
                                "paddingLeft": "10px",
                            },
                        ),
                        html.Div(
                            channel_row,
                            style={
                                "whiteSpace": "nowrap",
                                "marginBottom": "20px",
                            },
                        ),
                    ]
                )

        if not timeline_content:
            return html.Div("No images found in cache for the selected region")

        return html.Div(
            [
                html.H3(
                    "Trajectory Points",
                    style={
                        "marginBottom": "20px",
                        "fontSize": "24px",
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                    },
                ),
                html.Div(
                    timeline_content,
                    style={
                        "overflowX": "auto",
                        "overflowY": "hidden",
                        "padding": "15px",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
            ]
        )

    def _get_trajectory_images_lasso(
        self, x_axis: str, y_axis: str, selected_data: Optional[Dict]
    ) -> html.Div:
        """Get images of points selected by lasso."""
        if not selected_data or not selected_data.get("points"):
            return html.Div("Use the lasso tool to select points")

        # Dictionary to store points for each lasso selection
        lasso_clusters = {}
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

            if cache_key not in self.image_cache:
                continue

            curve_number = point.get("curveNumber", 0)
            if curve_number not in lasso_clusters:
                lasso_clusters[curve_number] = []

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
            timepoint_columns = []

            for _, row in cluster_df.iterrows():
                cache_key = (row["fov_name"], row["track_id"], row["t"])
                channel_images = []

                for channel in self.channels_to_display:
                    if channel in self.image_cache[cache_key]:
                        channel_images.append(
                            html.Div(
                                [
                                    html.Div(
                                        channel,
                                        style={
                                            "textAlign": "center",
                                            "fontSize": "12px",
                                            "fontWeight": "bold",
                                            "marginBottom": "5px",
                                        },
                                    ),
                                    html.Img(
                                        src=self.image_cache[cache_key][channel],
                                        style={
                                            "width": "150px",
                                            "height": "150px",
                                            "border": "1px solid #ddd",
                                            "borderRadius": "4px",
                                        },
                                    ),
                                ]
                            )
                        )

                if channel_images:
                    timepoint_columns.append(
                        html.Div(
                            [
                                html.Div(
                                    channel_images,
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "gap": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        f"Track {row['track_id']}, t={row['t']}",
                                        html.Br(),
                                        f"{x_axis}: {row[x_axis]:.2f}",
                                        html.Br(),
                                        f"{y_axis}: {row[y_axis]:.2f}",
                                    ],
                                    style={
                                        "textAlign": "center",
                                        "fontSize": "12px",
                                        "marginTop": "5px",
                                    },
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin": "10px",
                                "verticalAlign": "top",
                                "backgroundColor": "#f8f9fa",
                                "padding": "10px",
                                "borderRadius": "8px",
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                            },
                        )
                    )

            if timepoint_columns:
                cluster_sections.append(
                    html.Div(
                        [
                            html.H3(
                                f"Lasso Selection {cluster_idx + 1}",
                                style={
                                    "marginBottom": "15px",
                                    "fontSize": "20px",
                                    "fontWeight": "bold",
                                    "color": "#2c3e50",
                                    "borderBottom": "2px solid #007bff",
                                    "paddingBottom": "5px",
                                },
                            ),
                            html.Div(
                                timepoint_columns,
                                style={
                                    "overflowX": "auto",
                                    "whiteSpace": "nowrap",
                                    "padding": "15px",
                                    "backgroundColor": "#ffffff",
                                    "borderRadius": "8px",
                                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                    "marginBottom": "20px",
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
                        "fontSize": "24px",
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                    },
                ),
                html.Div(cluster_sections),
            ]
        )

    def _get_cluster_images(self) -> html.Div:
        """Display images for all clusters in a grid layout.

        Returns
        -------
        html.Div
            The cluster images component.
        """
        if not self.clusters:
            return html.Div("No clusters created yet")

        # Create cluster colors once
        cluster_colors = [
            f"rgb{tuple(int(x*255) for x in plt.cm.Set2(i % 8)[:3])}"
            for i in range(len(self.clusters))
        ]

        # Create individual cluster panels
        cluster_panels = []
        for cluster_idx, cluster_points in enumerate(self.clusters):
            # Create a single scrollable container for all channels
            all_channel_images = []
            for channel in self.channels_to_display:
                images = []
                for point in cluster_points:
                    cache_key = (point["fov_name"], point["track_id"], point["t"])
                    if channel in self.image_cache[cache_key]:
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
                                        f"Cluster {cluster_idx + 1}",
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
