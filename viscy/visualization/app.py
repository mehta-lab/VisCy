"""Main visualization application."""

import atexit
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.visualization.callbacks.selection import SelectionCallbacks
from viscy.visualization.components.figures import FigureCreator
from viscy.visualization.components.image_grid import ImageGrid
from viscy.visualization.components.scatter_plot import ScatterPlot
from viscy.visualization.components.scrollable_container import ScrollableContainer
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
        # Initialize cluster storage with annotations
        self.clusters: List[List[Dict[str, Any]]] = []
        self.cluster_annotations: Dict[int, Dict[str, str]] = {}
        self.cluster_points: Set[Tuple[str, int, int]] = set()
        # Initialize cluster colors
        self.cluster_colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#17becf",  # Cyan
        ]
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
                dcc.Store(id="image-click-store", storage_type="memory"),
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
        # Register selection callbacks
        from viscy.visualization.callbacks.selection import SelectionCallbacks

        selection_callbacks = SelectionCallbacks(self)
        selection_callbacks.register()

        # Register image callbacks
        from viscy.visualization.callbacks.image import ImageCallbacks

        image_callbacks = ImageCallbacks(self)
        image_callbacks.register()

        # Register cluster callbacks
        from viscy.visualization.callbacks.cluster import ClusterCallbacks

        cluster_callbacks = ClusterCallbacks(self)
        cluster_callbacks.register()

        # Register other callbacks
        self._register_figure_update_callbacks()
        self._register_timeline_callbacks()

    def _register_figure_update_callbacks(self):
        """Register callbacks for figure updates."""

        @self.app.callback(
            [
                Output("scatter-plot", "figure", allow_duplicate=True),
                Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                Input("color-mode", "value"),
                Input("show-arrows", "value"),
                Input("x-axis", "value"),
                Input("y-axis", "value"),
                Input("scatter-plot", "relayoutData"),
                Input("scatter-plot", "selectedData"),
                Input("clear-selection", "n_clicks"),
                Input("image-click-store", "data"),
            ],
            [State("scatter-plot", "figure")],
            prevent_initial_call=True,
        )
        def update_figure(
            color_mode,
            show_arrows,
            x_axis,
            y_axis,
            relayout_data,
            selected_data,
            clear_clicks,
            image_click_data,
            current_figure,
        ):
            """Update the figure."""
            show_arrows = len(show_arrows or []) > 0
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            trigger_prop = ctx.triggered[0]["prop_id"].split(".")[1]

            # Skip ALL updates during lasso selection
            if trigger_id == "scatter-plot":
                if current_figure.get("layout", {}).get("dragmode") == "lasso":
                    return dash.no_update, dash.no_update

            # Handle image click
            if trigger_id == "image-click-store" and image_click_data:
                # Find the point in the filtered_features_df that matches the clicked image
                point_data = self.filtered_features_df[
                    (
                        self.filtered_features_df["fov_name"]
                        == image_click_data["fov_name"]
                    )
                    & (
                        self.filtered_features_df["track_id"]
                        == image_click_data["track_id"]
                    )
                    & (self.filtered_features_df["t"] == image_click_data["t"])
                ]

                if point_data.empty:
                    return dash.no_update, dash.no_update

                # Create selectedData format with the actual x, y coordinates
                selected_data = {
                    "points": [
                        {
                            "x": float(point_data[x_axis].iloc[0]),
                            "y": float(point_data[y_axis].iloc[0]),
                            "text": f"Track: {image_click_data['track_id']}<br>Time: {image_click_data['t']}<br>FOV: {image_click_data['fov_name']}",
                        }
                    ]
                }

                # Create a new figure with updated opacities
                if color_mode == "track":
                    fig = self._create_track_colored_figure(
                        show_arrows,
                        x_axis,
                        y_axis,
                        highlight_point=(
                            image_click_data["fov_name"],
                            image_click_data["track_id"],
                            image_click_data["t"],
                        ),
                    )
                else:
                    fig = self._create_time_colored_figure(
                        show_arrows,
                        x_axis,
                        y_axis,
                        highlight_point=(
                            image_click_data["fov_name"],
                            image_click_data["track_id"],
                            image_click_data["t"],
                        ),
                    )

                return fig, selected_data

            # Handle clear selection
            if trigger_id == "clear-selection" and clear_clicks:
                if color_mode == "track":
                    fig = self._create_track_colored_figure(show_arrows, x_axis, y_axis)
                else:
                    fig = self._create_time_colored_figure(show_arrows, x_axis, y_axis)
                # Reset selection and UI state
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="event+select",
                    uirevision=True,  # Keep UI state consistent
                    selectdirection="any",
                )
                return fig, None

            # Create new figure only when necessary
            if trigger_id in [
                "color-mode",
                "show-arrows",
                "x-axis",
                "y-axis",
            ]:
                if color_mode == "track":
                    fig = self._create_track_colored_figure(show_arrows, x_axis, y_axis)
                else:
                    fig = self._create_time_colored_figure(show_arrows, x_axis, y_axis)
                # Maintain the lasso mode and selection state
                fig.update_layout(
                    dragmode="lasso",
                    clickmode="event+select",
                    uirevision=True,
                    selectdirection="any",
                )
                return fig, selected_data

            # For all other cases
            return dash.no_update, dash.no_update

    def _register_timeline_callbacks(self):
        """Register callbacks for timeline updates."""

        @self.app.callback(
            [
                Output("track-timeline", "children"),
                Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                Input("scatter-plot", "clickData"),
                Input("scatter-plot", "selectedData"),
                Input("image-click-store", "data"),
                Input("cluster-button", "n_clicks"),
            ],
            [
                State("scatter-plot", "selectedData"),
            ],
            prevent_initial_call=True,
        )
        def update_track_timeline(
            clickData, selectedData, image_click_data, cluster_clicks, current_selection
        ):
            """Update the track timeline with images from the clicked/selected track."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return (
                    html.Div("Click on a point or use lasso to see track timeline"),
                    dash.no_update,
                )

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            trigger_prop = ctx.triggered[0]["prop_id"].split(".")[1]

            # Clear selection only when creating cluster
            if trigger_id == "cluster-button" and cluster_clicks:
                return dash.no_update, None

            # Handle lasso selection
            if (
                trigger_id == "scatter-plot"
                and trigger_prop == "selectedData"
                and selectedData
            ):
                points = selectedData.get("points", [])
                if not points:
                    return html.Div("No points selected"), selectedData

                # Create a grid of all selected points
                all_track_data = []
                for point in points:
                    text = point["text"]
                    lines = text.split("<br>")
                    track_id = int(lines[0].split(": ")[1])
                    t = int(lines[1].split(": ")[1])
                    fov = lines[2].split(": ")[1]
                    all_track_data.append((track_id, t, fov))

                if not all_track_data:
                    return (
                        html.Div("No timeline data found for selected points"),
                        selectedData,
                    )

                # Create image grids for each selected point
                channel_grids = []
                for channel in self.channels_to_display:
                    images = []
                    for track_id, t, fov in all_track_data:
                        cache_key = (fov, track_id, t)
                        if (
                            cache_key in self.image_cache
                            and channel in self.image_cache[cache_key]
                        ):
                            images.append(
                                {
                                    "src": self.image_cache[cache_key][channel],
                                    "track_id": track_id,
                                    "t": t,
                                    "fov": fov,
                                }
                            )

                    if images:
                        channel_grids.append(
                            ImageGrid(
                                images=images,
                                channel_name=channel,
                                highlight_key=None,  # No highlight for multiple selection
                            ).create_layout()
                        )

                if not channel_grids:
                    return (
                        html.Div("No images found in cache for selected points"),
                        selectedData,
                    )

                return (
                    ScrollableContainer(
                        title=f"Selected Points ({len(all_track_data)} points)",
                        content=channel_grids,
                        max_height="80vh",
                        direction="horizontal",
                    ).create_layout(),
                    selectedData,
                )

            # Handle single point click
            elif trigger_id == "scatter-plot" and clickData:
                point = clickData["points"][0]
                text = point["text"]
                lines = text.split("<br>")
                track_id = int(lines[0].split(": ")[1])
                clicked_t = int(lines[1].split(": ")[1])
                fov = lines[2].split(": ")[1]
            elif trigger_id == "image-click-store" and image_click_data:
                track_id = image_click_data["track_id"]
                clicked_t = image_click_data["t"]
                fov = image_click_data["fov_name"]
            else:
                return dash.no_update, dash.no_update

            # Get all timepoints for this track
            track_data = self.filtered_features_df[
                (self.filtered_features_df["fov_name"] == fov)
                & (self.filtered_features_df["track_id"] == track_id)
            ].sort_values("t")

            if track_data.empty:
                return (
                    html.Div(f"No timeline data found for Track {track_id}"),
                    dash.no_update,
                )

            # Create image grids for each channel
            channel_grids = []
            for channel in self.channels_to_display:
                images = []
                for _, row in track_data.iterrows():
                    cache_key = (row["fov_name"], row["track_id"], row["t"])
                    if (
                        cache_key in self.image_cache
                        and channel in self.image_cache[cache_key]
                    ):
                        images.append(
                            {
                                "src": self.image_cache[cache_key][channel],
                                "track_id": row["track_id"],
                                "t": row["t"],
                                "fov": row["fov_name"],
                            }
                        )

                if images:
                    channel_grids.append(
                        ImageGrid(
                            images=images,
                            channel_name=channel,
                            highlight_key=(
                                fov,
                                track_id,
                                clicked_t,
                            ),  # Highlight clicked timepoint
                        ).create_layout()
                    )

            if not channel_grids:
                return (
                    html.Div("No images found in cache for this track"),
                    dash.no_update,
                )

            # Create scrollable container with all channel grids
            return (
                ScrollableContainer(
                    title=f"Track {track_id} Timeline (t={clicked_t} highlighted)",
                    content=channel_grids,
                    max_height="80vh",
                    direction="horizontal",
                ).create_layout(),
                dash.no_update,
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
        highlight_point=None,
    ) -> go.Figure:
        """Create scatter plot with track-based coloring."""
        fig = FigureCreator.create_track_colored_figure(
            self.features_df,
            self.filtered_features_df,
            self.clusters,
            self.cluster_points,
            show_arrows,
            x_axis,
            y_axis,
            highlight_point,
        )

        # Add cluster points with their respective colors
        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster:
                cluster_df = pd.DataFrame(cluster)
                color = self.cluster_colors[cluster_idx % len(self.cluster_colors)]
                opacity = 0.3

                # Create trace for cluster points
                cluster_trace = go.Scattergl(
                    x=cluster_df["x"],
                    y=cluster_df["y"],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=12,
                        symbol="circle",
                        opacity=opacity,
                    ),
                    name=self.cluster_annotations.get(cluster_idx, {}).get(
                        "label", f"Cluster {cluster_idx + 1}"
                    ),
                    hoverinfo="text",
                    text=[
                        f"Cluster: {cluster_idx + 1}<br>"
                        f"Track: {row['track_id']}<br>"
                        f"Time: {row['t']}<br>"
                        f"FOV: {row['fov_name']}"
                        for _, row in cluster_df.iterrows()
                    ],
                    showlegend=True,
                    selected=dict(
                        marker=dict(
                            color=color,
                            size=12,
                            opacity=1.0,
                        )
                    ),
                    unselected=dict(
                        marker=dict(
                            color=color,
                            opacity=opacity,
                        )
                    ),
                )
                fig.add_trace(cluster_trace)

        # Update layout
        fig.update_layout(
            dragmode="lasso",
            clickmode="event+select",
            uirevision=True,
            selectdirection="any",
        )

        return fig

    def _create_time_colored_figure(
        self,
        show_arrows=False,
        x_axis="PCA1",
        y_axis="PCA2",
        highlight_point=None,
    ) -> go.Figure:
        """Create scatter plot with time-based coloring."""
        return FigureCreator.create_time_colored_figure(
            self.features_df,
            self.filtered_features_df,
            show_arrows,
            x_axis,
            y_axis,
            highlight_point=highlight_point,
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
        self, x_axis: str, y_axis: str, relayout_data: Dict
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
        sort_by = y_axis if x_axis == "x" else x_axis
        nearby_points = self.filtered_features_df[mask].sort_values(sort_by)

        if len(nearby_points) == 0:
            return html.Div("No points found in the selected region")

        # Create cluster color mapping
        cluster_colors = [
            f"rgb{tuple(int(x*255) for x in plt.cm.Set2(i % 8)[:3])}"
            for i in range(len(self.clusters))
        ]
        point_to_cluster = {}
        for cluster_idx, cluster in enumerate(self.clusters):
            for point in cluster:
                point_key = (point["fov_name"], point["track_id"], point["t"])
                point_to_cluster[point_key] = cluster_idx

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
                    # Determine border color based on cluster membership
                    border_style = "1px solid #ddd"
                    if cache_key in point_to_cluster:
                        cluster_idx = point_to_cluster[cache_key]
                        border_style = f"2px solid {cluster_colors[cluster_idx]}"

                    channel_row.append(
                        html.Div(
                            [
                                html.Img(
                                    src=self.image_cache[cache_key][channel],
                                    style={
                                        "width": "150px",
                                        "height": "150px",
                                        "border": border_style,
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                    id={
                                        "type": "image",
                                        "track_id": row["track_id"],
                                        "t": row["t"],
                                        "fov": row["fov_name"],
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
                                        "color": (
                                            cluster_colors[point_to_cluster[cache_key]]
                                            if cache_key in point_to_cluster
                                            else "#2c3e50"
                                        ),
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
                                            "cursor": "pointer",
                                        },
                                        id={
                                            "type": "image",
                                            "track_id": row["track_id"],
                                            "t": row["t"],
                                            "fov": row["fov_name"],
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
        """Get the cluster images layout.

        Returns
        -------
        html.Div
            The cluster images layout.
        """
        from viscy.visualization.components.cluster_manager import ClusterManager
        from viscy.visualization.components.image_grid import ImageGrid
        from viscy.visualization.components.scrollable_container import (
            ScrollableContainer,
        )

        if not self.clusters:
            return None

        cluster_manager = ClusterManager()
        cluster_cards = []

        for cluster_idx, cluster in enumerate(self.clusters):
            # Get cluster color
            color = self.cluster_colors[cluster_idx % len(self.cluster_colors)]

            # Filter points to only include those from selected track IDs
            filtered_points = [
                point
                for point in cluster
                if any(
                    point["track_id"] in track_ids
                    for fov, track_ids in self.filtered_tracks_by_fov.items()
                    if point["fov_name"] == fov
                )
            ]

            # Create image grids for each channel
            channel_grids = []
            for channel in self.channels_to_display:
                images = []
                for point in filtered_points:
                    cache_key = (point["fov_name"], point["track_id"], point["t"])
                    if (
                        cache_key in self.image_cache
                        and channel in self.image_cache[cache_key]
                    ):
                        images.append(
                            {
                                "src": self.image_cache[cache_key][channel],
                                "track_id": point["track_id"],
                                "t": point["t"],
                                "fov": point["fov_name"],
                            }
                        )

                if images:
                    channel_grids.append(
                        html.Div(
                            [
                                html.H4(
                                    channel,
                                    style={
                                        "marginTop": "5px",
                                        "marginBottom": "5px",
                                        "fontSize": "14px",
                                        "fontWeight": "bold",
                                        "color": color,
                                    },
                                ),
                                ImageGrid(
                                    images=images,
                                    channel_name=channel,
                                    highlight_key=None,  # No highlight for cluster view
                                ).create_layout(),
                            ]
                        )
                    )

            if channel_grids:
                # Create cluster card
                cluster_card = html.Div(
                    [
                        # Header section with title and controls
                        html.Div(
                            [
                                # Left side - Title and delete button
                                html.Div(
                                    [
                                        html.H3(
                                            self.cluster_annotations.get(
                                                cluster_idx, {}
                                            ).get(
                                                "label", f"Cluster {cluster_idx + 1}"
                                            ),
                                            style={
                                                "margin": "0",
                                                "fontSize": "16px",
                                                "fontWeight": "bold",
                                                "color": color,
                                                "marginRight": "10px",
                                            },
                                        ),
                                        html.Button(
                                            "×",
                                            id={
                                                "type": "delete-cluster",
                                                "index": cluster_idx,
                                            },
                                            style={
                                                "border": "none",
                                                "background": "none",
                                                "fontSize": "20px",
                                                "cursor": "pointer",
                                                "color": "#ff0000",
                                                "padding": "0 5px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                    },
                                ),
                                # Right side - Point count
                                html.P(
                                    f"Points: {len(filtered_points)} (from selected tracks)",
                                    style={
                                        "margin": "0",
                                        "color": color,
                                        "fontSize": "14px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "10px",
                                "padding": "5px 0",
                                "borderBottom": f"2px solid {color}",
                            },
                        ),
                        # Controls section
                        html.Div(
                            [
                                dcc.Input(
                                    id={"type": "cluster-label", "index": cluster_idx},
                                    type="text",
                                    placeholder="Enter cluster label",
                                    value=self.cluster_annotations.get(
                                        cluster_idx, {}
                                    ).get("label", ""),
                                    style={
                                        "width": "200px",
                                        "marginRight": "10px",
                                        "padding": "5px",
                                    },
                                ),
                                dcc.Textarea(
                                    id={
                                        "type": "cluster-description",
                                        "index": cluster_idx,
                                    },
                                    placeholder="Enter cluster description",
                                    value=self.cluster_annotations.get(
                                        cluster_idx, {}
                                    ).get("description", ""),
                                    style={
                                        "width": "calc(100% - 220px)",  # Account for the input width and margin
                                        "height": "32px",
                                        "resize": "none",
                                        "padding": "5px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "marginBottom": "10px",
                            },
                        ),
                        # Channel grids section with auto height
                        html.Div(
                            channel_grids,
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "5px",
                                "overflowY": "auto",
                                "paddingRight": "5px",
                            },
                        ),
                    ],
                    style={
                        "border": f"2px solid {color}",
                        "borderRadius": "8px",
                        "padding": "10px",
                        "backgroundColor": "#ffffff",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "display": "flex",
                        "flexDirection": "column",
                        "width": "calc(25% - 15px)",  # 4 columns with gap
                        "marginBottom": "20px",  # Space between rows
                        "marginRight": "20px",  # Space between cards in a row
                        "height": "fit-content",  # Adjust height to content
                        "maxHeight": "90vh",  # Maximum height constraint
                    },
                )
                cluster_cards.append(cluster_card)

        # Create rows with maximum 4 cards each
        rows = []
        for i in range(0, len(cluster_cards), 4):
            row = cluster_cards[i : i + 4]
            rows.append(
                html.Div(
                    row,
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "width": "100%",
                        "marginBottom": "20px",
                    },
                )
            )

        return html.Div(
            [
                html.H2(
                    f"Clusters ({len(self.clusters)})",
                    style={
                        "marginBottom": "20px",
                        "fontSize": "24px",
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                    },
                ),
                # Vertical scrollable container for all rows
                html.Div(
                    rows,
                    style={
                        "overflowY": "auto",
                        "maxHeight": "calc(100vh - 200px)",  # Leave space for header
                        "width": "100%",
                        "paddingRight": "10px",  # Space for scrollbar
                    },
                ),
            ]
        )

    def export_clusters(self, output_path: Optional[str] = None) -> Dict:
        """Export cluster annotations and points to a JSON file.

        Parameters
        ----------
        output_path : Optional[str]
            Path to save the JSON file. If None, returns the dictionary without saving.

        Returns
        -------
        Dict
            Dictionary containing cluster data and annotations
        """
        export_data = {
            "clusters": [
                {
                    "points": cluster,
                    "annotation": self.cluster_annotations.get(
                        i, {"label": "", "description": ""}
                    ),
                }
                for i, cluster in enumerate(self.clusters)
            ]
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        return export_data

    def import_clusters(self, input_path: str) -> None:
        """Import cluster annotations and points from a JSON file.

        Parameters
        ----------
        input_path : str
            Path to the JSON file containing cluster data
        """
        with open(input_path, "r") as f:
            import_data = json.load(f)

        self.clusters = []
        self.cluster_annotations = {}
        self.cluster_points.clear()

        for i, cluster_data in enumerate(import_data["clusters"]):
            self.clusters.append(cluster_data["points"])
            self.cluster_annotations[i] = cluster_data["annotation"]
            # Rebuild cluster points set
            for point in cluster_data["points"]:
                self.cluster_points.add(
                    (point["fov_name"], point["track_id"], point["t"])
                )
