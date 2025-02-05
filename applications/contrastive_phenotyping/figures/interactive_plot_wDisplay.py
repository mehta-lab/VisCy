# %%
# This is a simple example of an interactive plot using Dash.
import atexit
import base64
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


class ImageDisplayApp:
    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        features_path: str,
        channels_to_display: list[str] | str,
        fov_tracks: dict[str, list[int] | str],
        z_range: tuple[int, int] | list[int] = (0, 1),
        yx_patch_size: tuple[int, int] | list[int] = (128, 128),
    ) -> None:
        self.data_path = Path(data_path)
        self.tracks_path = Path(tracks_path)
        self.features_path = Path(features_path)
        self.fov_tracks = fov_tracks
        self.image_cache = {}
        self.app = None
        self.features_df = None
        self.fig = None
        self.channels_to_display = channels_to_display
        self.z_range = z_range
        self.yx_patch_size = yx_patch_size
        self.filtered_tracks_by_fov = {}
        self._z_idx = (self.z_range[1] - self.z_range[0]) // 2
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

        # PCA transformation
        scaled_features = StandardScaler().fit_transform(features.values)
        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(scaled_features)

        # Add PCA coordinates to the features dataframe
        self.features_df["PCA1"] = pca_coords[:, 0]
        self.features_df["PCA2"] = pca_coords[:, 1]
        self.features_df["PCA3"] = pca_coords[:, 2]

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
                                "width": "60%",
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
                                    ],
                                ),
                                dcc.Loading(
                                    id="loading",
                                    children=[
                                        dcc.Graph(
                                            id="scatter-plot",
                                            figure=self.fig,
                                            config={"displayModeBar": False},
                                            style={"height": "50vh"},
                                        ),
                                    ],
                                    type="default",
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "width": "40%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                                "paddingLeft": "20px",
                            },
                            children=[
                                html.Div(
                                    id="track-timeline",
                                    style={
                                        "height": "80vh",
                                        "overflowY": "auto",
                                    },
                                ),
                            ],
                        ),
                    ]
                ),
            ],
        )

        @self.app.callback(
            dd.Output("scatter-plot", "figure"),
            [dd.Input("color-mode", "value"), dd.Input("show-arrows", "value")],
        )
        def update_figure(color_mode, show_arrows):
            show_arrows = "show" in (show_arrows or [])
            if color_mode == "track":
                return self._create_track_colored_figure(show_arrows)
            else:
                return self._create_time_colored_figure(show_arrows)

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

            # Create rows for each channel
            channel_rows = []
            for channel in self.channels_to_display:
                images = []
                for t in track_data["t"].unique():
                    cache_key = (fov_name, track_id, t)
                    if cache_key in self.image_cache:
                        # Define styles based on whether this is the clicked timepoint
                        is_clicked = t == clicked_time
                        image_style = {
                            "width": "100px",
                            "height": "100px",
                            "margin": "2px",
                            "display": "inline-block",
                            "border": (
                                "3px solid #007bff" if is_clicked else "none"
                            ),  # Blue border for clicked image
                        }
                        time_style = {
                            "textAlign": "center",
                            "fontSize": "24px" if is_clicked else "18px",
                            "fontWeight": "bold" if is_clicked else "normal",
                            "color": "#007bff" if is_clicked else "black",
                        }

                        images.append(
                            html.Div(
                                [
                                    html.Img(
                                        src=self.image_cache[cache_key][channel],
                                        style=image_style,
                                    ),
                                    html.Div(
                                        f"t={t}",
                                        style=time_style,
                                    ),
                                ],
                                style={"display": "inline-block"},
                            )
                        )

                if images:  # Only add row if there are images
                    channel_rows.extend(
                        [
                            html.H5(f"{channel}", style={"margin": "5px"}),
                            html.Div(
                                images,
                                style={
                                    "overflowX": "auto",
                                    "whiteSpace": "nowrap",
                                    "padding": "10px",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "5px",
                                    "marginBottom": "10px",
                                },
                            ),
                        ]
                    )

            return html.Div(
                [
                    html.H4(f"Track {track_id} (FOV: {fov_name})"),
                    html.Div(channel_rows),
                ]
            )

    def _create_track_colored_figure(self, show_arrows=False):
        """Create scatter plot with track-based coloring"""
        unique_tracks = self.filtered_features_df["track_id"].unique()
        cmap = plt.cm.tab20
        track_colors = {
            track_id: f"rgb{tuple(int(x*255) for x in cmap(i % 20)[:3])}"
            for i, track_id in enumerate(unique_tracks)
        }

        fig = go.Figure()

        # Add background points with hover info
        all_tracks_df = self.features_df[
            self.features_df["fov_name"].isin(self.fov_tracks.keys())
        ]
        fig.add_trace(
            go.Scatter(
                x=all_tracks_df["PCA1"],
                y=all_tracks_df["PCA2"],
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
            )
        )

        for track_id in unique_tracks:
            track_data = self.filtered_features_df[
                self.filtered_features_df["track_id"] == track_id
            ].sort_values("t")

            # Add points
            fig.add_trace(
                go.Scatter(
                    x=track_data["PCA1"],
                    y=track_data["PCA2"],
                    mode="markers",
                    marker=dict(size=15, color=track_colors[track_id]),
                    name=f"Track {track_id}",
                    text=[
                        f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for t, fov in zip(track_data["t"], track_data["fov_name"])
                    ],
                    hoverinfo="text",
                )
            )

            # Add arrows if requested
            if show_arrows and len(track_data) > 1:
                for i in range(len(track_data) - 1):
                    fig.add_annotation(
                        x=track_data["PCA1"].iloc[i + 1],
                        y=track_data["PCA2"].iloc[i + 1],
                        ax=track_data["PCA1"].iloc[i],
                        ay=track_data["PCA2"].iloc[i],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=2,
                        arrowwidth=2,
                        arrowcolor=track_colors[track_id],
                    )

        self._update_figure_layout(fig)
        return fig

    def _create_time_colored_figure(self, show_arrows=False):
        """Create scatter plot with time-based coloring"""
        fig = go.Figure()

        # Add background points with hover info
        all_tracks_df = self.features_df[
            self.features_df["fov_name"].isin(self.fov_tracks.keys())
        ]
        fig.add_trace(
            go.Scatter(
                x=all_tracks_df["PCA1"],
                y=all_tracks_df["PCA2"],
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
            )
        )

        # Add time-colored points
        fig.add_trace(
            go.Scatter(
                x=self.filtered_features_df["PCA1"],
                y=self.filtered_features_df["PCA2"],
                mode="markers",
                marker=dict(
                    size=15,
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
            )
        )

        # Add arrows if requested
        if show_arrows:
            for track_id in self.filtered_features_df["track_id"].unique():
                track_data = self.filtered_features_df[
                    self.filtered_features_df["track_id"] == track_id
                ].sort_values("t")

                if len(track_data) > 1:
                    for i in range(len(track_data) - 1):
                        fig.add_annotation(
                            x=track_data["PCA1"].iloc[i + 1],
                            y=track_data["PCA2"].iloc[i + 1],
                            ax=track_data["PCA1"].iloc[i],
                            ay=track_data["PCA2"].iloc[i],
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=2,
                            arrowwidth=2,
                            arrowcolor="gray",
                        )

        self._update_figure_layout(fig)
        return fig

    def _update_figure_layout(self, fig):
        """Update the layout for a figure"""
        fig.update_layout(
            plot_bgcolor="white",
            title="PCA visualization of Selected Tracks",
            xaxis_title="PC1",
            yaxis_title="PC2",
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

    def preload_images(self):
        """Preload all images into memory"""
        logger.info("Preloading images into cache...")

        # Process each FOV and its tracks
        for fov_name, track_ids in self.filtered_tracks_by_fov.items():
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
                num_workers=16,
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

                    # Process each channel based on its type
                    processed_channels = {}
                    for idx, channel in enumerate(self.channels_to_display):
                        if channel in ["Phase3D", "DIC", "BF"]:
                            # FIXME: the z is hardcorded.
                            processed = self._normalize_image(img[0, idx, self._z_idx])
                        else:
                            processed = self._normalize_image(
                                np.max(img[0, idx], axis=0)
                            )

                        processed_channels[channel] = self._numpy_to_base64(processed)

                    self.image_cache[cache_key] = processed_channels

                except Exception as e:
                    print(f"Error processing images for {fov_name}, {track_id}: {e}")
                    continue

        logging.info(f"Cached {len(self.image_cache)} images")

    def _cleanup_cache(self):
        """Clear the image cache when the program exits"""
        logging.info("Cleaning up image cache...")
        self.image_cache.clear()

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


# %%
if __name__ == "__main__":
    # Example of using multiple FOVs with specific track IDs for each
    fov_tracks_dict = {
        "/0/6/000000": [1, 5, 6, 7, 9, 14, 15],
        # "/0/3/002000": "all",  # Use all tracks for this FOV
    }

    try:
        app = ImageDisplayApp(
            data_path="/hpc/projects/organelle_phenotyping/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/registered_chunked.zarr",
            tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.2-tracking/track.zarr",
            features_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/jun_time_interval_1_epoch_178.zarr",
            channels_to_display=[
                "Phase3D",
                "MultiCam_GFP_mCherry_BF-Prime BSI Express",
            ],
            fov_tracks=fov_tracks_dict,
            z_range=(31, 36),
            yx_patch_size=(128, 128),
        )
        app.preload_images()
        app.run(debug=True)  # Debug mode without reloader
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application shutdown complete")

# %%
