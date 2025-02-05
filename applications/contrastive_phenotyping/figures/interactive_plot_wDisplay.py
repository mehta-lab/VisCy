# %%
# This is a simple example of an interactive plot using Dash.
import atexit
import base64
from io import BytesIO
from pathlib import Path

import dash
import dash.dependencies as dd
import numpy as np
import plotly.express as px
from dash import dcc, html
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from viscy.representation.evaluation import dataset_of_tracks
from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset


class ImageDisplayApp:
    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        features_path: str,
        fov_name: str,
        channels_to_display: list[str],
        z_range: tuple[int, int] | list[int] = (0, 1),
        yx_patch_size: tuple[int, int] | list[int] = (128, 128),
    ) -> None:
        self.data_path = Path(data_path)
        self.tracks_path = Path(tracks_path)
        self.features_path = Path(features_path)
        self.fov_name = fov_name
        self.image_cache = {}
        self.app = None
        self.features_df = None
        self.fig = None
        self.channels_to_display = channels_to_display
        self.z_range = z_range
        self.yx_patch_size = yx_patch_size

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

        # Filter data for the specific fov_name
        self.filtered_tracks = (
            self.features_df[self.features_df["fov_name"] == self.fov_name]["track_id"]
            .unique()
            .tolist()
        )
        self.filtered_features_df = self.features_df[
            self.features_df["fov_name"] == self.fov_name
        ]

    def _create_figure(self):
        """Create the scatter plot figure"""
        self.fig = px.scatter(
            self.filtered_features_df,  # Use filtered dataframe
            x="PCA1",
            y="PCA2",
            color="t",  # Color by time
            hover_name="fov_name",
            hover_data=["id", "t", "track_id"],
            title=f"PCA visualization for FOV {self.fov_name}",
            labels={
                "PCA1": "First Principal Component",
                "PCA2": "Second Principal Component",
                "t": "Time",
            },
        )
        # Update layout for better visualization
        self.fig.update_layout(
            plot_bgcolor="white",
            width=800,
            height=600,
        )
        self.fig.update_traces(
            marker=dict(size=8),
            selector=dict(mode="markers"),
        )

    def _init_app(self):
        """Initialize the Dash application"""
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div(
            [
                dcc.Graph(id="scatter-plot", figure=self.fig),
                html.Div(
                    [
                        html.Div(id="track-timeline", style={"margin": "20px"}),
                    ]
                ),
            ]
        )

        # Register callback for track timeline display
        @self.app.callback(
            dd.Output("track-timeline", "children"),
            [dd.Input("scatter-plot", "clickData")],
        )
        def update_track_timeline(clickData):
            if clickData is None:
                return html.Div("Click on a point to see the track timeline")

            # Get track information from clicked point
            fov_name = clickData["points"][0]["hovertext"]
            track_id = clickData["points"][0]["customdata"][2]

            # Get all timepoints for this track
            track_data = self.features_df[
                (self.features_df["fov_name"] == fov_name)
                & (self.features_df["track_id"] == track_id)
            ].sort_values("t")

            if track_data.empty:
                return html.Div(f"No data found for track {track_id}")

            # Create rows for each channel
            channel_images = {channel: [] for channel in self.channels_to_display}

            for t in track_data["t"].unique():
                cache_key = (fov_name, track_id, t)
                if cache_key in self.image_cache:
                    # Add images for each channel
                    for channel in self.channels_to_display:
                        channel_images[channel].append(
                            html.Div(
                                [
                                    html.Img(
                                        src=self.image_cache[cache_key][channel],
                                        style={
                                            "width": "150px",
                                            "height": "150px",
                                            "margin": "2px",
                                            "display": "inline-block",
                                        },
                                    ),
                                    html.Div(
                                        f"t={t}",
                                        style={
                                            "textAlign": "center",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ],
                                style={"display": "inline-block"},
                            )
                        )

            # Create timeline display with a row for each channel
            channel_rows = []
            for channel in self.channels_to_display:
                channel_rows.extend(
                    [
                        html.H5(f"{channel} Channel", style={"margin": "5px"}),
                        html.Div(
                            channel_images[channel],
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
                [html.H4(f"Track {track_id} Timeline"), html.Div(channel_rows)]
            )

    @staticmethod
    def _normalize_image(img_array):
        """Normalize a single image array to [0, 255]"""
        img_array = np.clip(img_array, img_array.min(), img_array.max())
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        return (img_array * 255).astype(np.uint8)

    @staticmethod
    def _numpy_to_base64(img_array):
        """Convert numpy array to base64 string"""
        if not isinstance(img_array, np.uint8):
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode(
            "utf-8"
        )

    def preload_images(self):
        """Preload all images into memory"""
        print("Preloading images into cache...")
        data_module = TripletDataModule(
            data_path=self.data_path,
            tracks_path=self.tracks_path,
            include_fov_names=[self.fov_name] * len(self.filtered_tracks),
            include_track_ids=self.filtered_tracks,
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
                cache_key = (self.fov_name, track_id[0], t[0])

                # Process each channel based on its type
                processed_channels = {}
                for idx, channel in enumerate(self.channels_to_display):
                    if channel == "Phase3D":
                        # For Phase3D, take the middle z-slice
                        processed = self._normalize_image(img[0, idx, 2])
                    else:
                        # For other channels, do max projection
                        processed = self._normalize_image(np.max(img[0, idx], axis=0))

                    processed_channels[channel] = self._numpy_to_base64(processed)

                # Store processed channels in cache
                self.image_cache[cache_key] = processed_channels

            except Exception as e:
                print(f"Error processing images for {self.fov_name}, {track_id}: {e}")
                continue

        print(f"Cached {len(self.image_cache)} images")

    def _cleanup_cache(self):
        """Clear the image cache when the program exits"""
        print("Cleaning up image cache...")
        self.image_cache.clear()

    def run(self, debug=True):
        """Run the Dash server"""
        self.app.run_server(debug=debug)


# %%
if __name__ == "__main__":
    app = ImageDisplayApp(
        data_path="/hpc/projects/organelle_phenotyping/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/registered_chunked.zarr",
        tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.2-tracking/track.zarr",
        features_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/jun_time_interval_1_epoch_178.zarr",
        fov_name="/0/6/000000",
        channels_to_display=["Phase3D", "MultiCam_GFP_mCherry_BF-Prime BSI Express"],
        z_range=(31, 36),
        yx_patch_size=(128, 128),
    )
    app.preload_images()
    app.run()

# %%
