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


# %%
class EmbeddingVisualizationApp:
    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        features_path: str,
        channels_to_display: list[str] | str,
        fov_tracks: dict[str, list[int] | str],
        z_range: tuple[int, int] | list[int] = (0, 1),
        yx_patch_size: tuple[int, int] | list[int] = (128, 128),
        num_PC_components: int = 3,
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
        self.num_PC_components = num_PC_components
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
                                                    options=self.pc_options,
                                                    value="PCA1",
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
                                                    options=self.pc_options,
                                                    value="PCA2",
                                                    style={"width": "200px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Selection mode:",
                                                    style={"marginRight": "10px"},
                                                ),
                                                dcc.RadioItems(
                                                    id="selection-mode",
                                                    options=[
                                                        {
                                                            "label": "Shaded region",
                                                            "value": "region",
                                                        },
                                                        {
                                                            "label": "Lasso",
                                                            "value": "lasso",
                                                        },
                                                    ],
                                                    value="region",
                                                    inline=True,
                                                ),
                                                html.Button(
                                                    "Clear Selection",
                                                    id="clear-selection",
                                                    style={
                                                        "marginLeft": "10px",
                                                        "backgroundColor": "#dc3545",
                                                        "color": "white",
                                                        "border": "none",
                                                        "padding": "5px 10px",
                                                        "borderRadius": "4px",
                                                        "cursor": "pointer",
                                                    },
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Trajectory:",
                                                    style={"marginRight": "10px"},
                                                ),
                                                dcc.RadioItems(
                                                    id="trajectory-mode",
                                                    options=[
                                                        {
                                                            "label": "X-axis",
                                                            "value": "x",
                                                        },
                                                        {
                                                            "label": "Y-axis",
                                                            "value": "y",
                                                        },
                                                    ],
                                                    value="x",
                                                    inline=True,
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
                            ],
                        ),
                        # Trajectory Images Section
                        html.Div(
                            [
                                html.H4(
                                    "Points along trajectory",
                                    style={
                                        "marginTop": "20px",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Div(
                                    id="trajectory-images",
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
                        ),
                        # Track Timeline Section (existing)
                        html.Div(
                            style={
                                "width": "100%",
                                "display": "block",
                                "marginTop": "20px",
                            },
                            children=[
                                html.Div(
                                    id="track-timeline",
                                    style={
                                        "height": "auto",
                                        "overflowY": "auto",
                                        "maxHeight": "80vh",
                                    },
                                ),
                            ],
                        ),
                    ]
                ),
            ],
        )

        @self.app.callback(
            [
                dd.Output("scatter-plot", "figure", allow_duplicate=True),
                dd.Output("trajectory-images", "children", allow_duplicate=True),
                dd.Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [dd.Input("clear-selection", "n_clicks")],
            [
                dd.State("scatter-plot", "figure"),
                dd.State("color-mode", "value"),
                dd.State("show-arrows", "value"),
                dd.State("x-axis", "value"),
                dd.State("y-axis", "value"),
                dd.State("trajectory-mode", "value"),
                dd.State("selection-mode", "value"),
            ],
            prevent_initial_call=True,
        )
        def clear_selection(
            n_clicks,
            current_figure,
            color_mode,
            show_arrows,
            x_axis,
            y_axis,
            trajectory_mode,
            selection_mode,
        ):
            if n_clicks:
                # Create a new figure with no selections
                if color_mode == "track":
                    fig = self._create_track_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                        trajectory_mode,
                        selection_mode,
                    )
                else:
                    fig = self._create_time_colored_figure(
                        len(show_arrows or []) > 0,
                        x_axis,
                        y_axis,
                        trajectory_mode,
                        selection_mode,
                    )

                fig.update_layout(
                    dragmode="lasso" if selection_mode == "lasso" else "pan",
                    clickmode="event+select",
                    uirevision=None,  # Reset UI state
                    selectdirection="any",
                )

                return fig, html.Div("Use the lasso tool to select points"), None
            return dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            [
                dd.Output("scatter-plot", "figure", allow_duplicate=True),
                dd.Output("trajectory-images", "children", allow_duplicate=True),
                dd.Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                dd.Input("color-mode", "value"),
                dd.Input("show-arrows", "value"),
                dd.Input("x-axis", "value"),
                dd.Input("y-axis", "value"),
                dd.Input("trajectory-mode", "value"),
                dd.Input("selection-mode", "value"),
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
            trajectory_mode,
            selection_mode,
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
                fig = dash.no_update

            # Handle selection mode changes
            if triggered_id == "selection-mode":
                selected_data = None
                if color_mode == "track":
                    fig = self._create_track_colored_figure(
                        show_arrows, x_axis, y_axis, trajectory_mode, selection_mode
                    )
                else:
                    fig = self._create_time_colored_figure(
                        show_arrows, x_axis, y_axis, trajectory_mode, selection_mode
                    )
                fig.update_layout(
                    dragmode="lasso" if selection_mode == "lasso" else "pan",
                    clickmode="event+select",
                    uirevision=None,
                    selectdirection="any",
                )

            # Get trajectory images based on selection mode
            if selection_mode == "region":
                trajectory_images = self._get_trajectory_images_region(
                    x_axis, y_axis, trajectory_mode, relayout_data
                )
            else:
                trajectory_images = self._get_trajectory_images_lasso(
                    x_axis, y_axis, selected_data
                )

            return fig, trajectory_images, selected_data

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

    def _create_track_colored_figure(
        self,
        show_arrows=False,
        x_axis="PCA1",
        y_axis="PCA2",
        trajectory_mode="x",
        selection_mode="region",
    ):
        """Create scatter plot with track-based coloring"""
        unique_tracks = self.filtered_features_df["track_id"].unique()
        cmap = plt.cm.tab20
        track_colors = {
            track_id: f"rgb{tuple(int(x*255) for x in cmap(i % 20)[:3])}"
            for i, track_id in enumerate(unique_tracks)
        }

        fig = go.Figure()

        # Add background points with hover info (excluding the colored tracks)
        background_df = self.features_df[
            (self.features_df["fov_name"].isin(self.fov_tracks.keys()))
            & (~self.features_df["track_id"].isin(unique_tracks))
        ]

        if not background_df.empty:
            fig.add_trace(
                go.Scatter(
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
                )
            )

        # Add points for each selected track
        for track_id in unique_tracks:
            track_data = self.filtered_features_df[
                self.filtered_features_df["track_id"] == track_id
            ].sort_values("t")

            # Add points
            fig.add_trace(
                go.Scatter(
                    x=track_data[x_axis],
                    y=track_data[y_axis],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color=track_colors[track_id],
                        line=dict(width=1, color="black"),
                        opacity=1.0,
                    ),
                    name=f"Track {track_id}",
                    text=[
                        f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}"
                        for t, fov in zip(track_data["t"], track_data["fov_name"])
                    ],
                    hoverinfo="text",
                    unselected=dict(marker=dict(opacity=0.3, size=15)),
                    selected=dict(marker=dict(size=20, opacity=1.0)),
                )
            )

            # Add trajectory lines and arrows if requested
            if show_arrows and len(track_data) > 1:
                x_coords = track_data[x_axis].values
                y_coords = track_data[y_axis].values
                t_coords = track_data["t"].values

                # Add dashed lines for the trajectory
                fig.add_trace(
                    go.Scatter(
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
                arrow_interval = max(
                    1, len(track_data) // 5
                )  # Show ~5 arrows per track
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
                            arrowsize=2,
                            arrowwidth=2,
                            arrowcolor=track_colors[track_id],
                            opacity=0.8,
                        )

        # Add draggable shaded region for trajectory only if in region selection mode
        if selection_mode == "region":
            x_range = [
                self.filtered_features_df[x_axis].min(),
                self.filtered_features_df[x_axis].max(),
            ]
            y_range = [
                self.filtered_features_df[y_axis].min(),
                self.filtered_features_df[y_axis].max(),
            ]

            # Add padding to ranges
            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
            y_range = [y_range[0] - y_padding, y_range[1] + y_padding]

            if trajectory_mode == "x":
                # Vertical shaded region
                x_mid = (x_range[0] + x_range[1]) / 2
                tolerance = (x_range[1] - x_range[0]) * 0.05  # 5% tolerance

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
                # Horizontal shaded region
                y_mid = (y_range[0] + y_range[1]) / 2
                tolerance = (y_range[1] - y_range[0]) * 0.05  # 5% tolerance

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

            # Update axes ranges to show the full region
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)

        self._update_figure_layout(fig, x_axis, y_axis)
        return fig

    def _create_time_colored_figure(
        self,
        show_arrows=False,
        x_axis="PCA1",
        y_axis="PCA2",
        trajectory_mode="x",
        selection_mode="region",
    ):
        """Create scatter plot with time-based coloring"""
        fig = go.Figure()

        # Add background points with hover info
        all_tracks_df = self.features_df[
            self.features_df["fov_name"].isin(self.fov_tracks.keys())
        ]
        fig.add_trace(
            go.Scatter(
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
            )
        )

        # Add time-colored points
        fig.add_trace(
            go.Scatter(
                x=self.filtered_features_df[x_axis],
                y=self.filtered_features_df[y_axis],
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

        # Add draggable shaded region for trajectory only if in region selection mode
        if selection_mode == "region":
            x_range = [
                self.filtered_features_df[x_axis].min(),
                self.filtered_features_df[x_axis].max(),
            ]
            y_range = [
                self.filtered_features_df[y_axis].min(),
                self.filtered_features_df[y_axis].max(),
            ]

            # Add padding to ranges
            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
            y_range = [y_range[0] - y_padding, y_range[1] + y_padding]

            if trajectory_mode == "x":
                # Vertical shaded region
                x_mid = (x_range[0] + x_range[1]) / 2
                tolerance = (x_range[1] - x_range[0]) * 0.05  # 5% tolerance

                # Add draggable shaded region
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
                # Horizontal shaded region
                y_mid = (y_range[0] + y_range[1]) / 2
                tolerance = (y_range[1] - y_range[0]) * 0.05  # 5% tolerance

                # Add draggable shaded region
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

            # Update axes ranges to show the full region
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)

        self._update_figure_layout(fig, x_axis, y_axis)
        return fig

    def _update_figure_layout(self, fig, x_axis="PCA1", y_axis="PCA2"):
        """Update the layout for a figure"""
        # Get the axis labels with explained variance
        x_label = next(
            (opt["label"] for opt in self.pc_options if opt["value"] == x_axis),
            x_axis,
        )
        y_label = next(
            (opt["label"] for opt in self.pc_options if opt["value"] == y_axis),
            y_axis,
        )

        fig.update_layout(
            plot_bgcolor="white",
            title="PCA visualization of Selected Tracks",
            xaxis_title=x_label,
            yaxis_title=y_label,
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

    def _cleanup_cache(self):
        """Clear the image cache when the program exits"""
        logging.info("Cleaning up image cache...")
        self.image_cache.clear()

    def _get_trajectory_images_region(
        self, x_axis, y_axis, trajectory_mode, relayout_data
    ):
        """Get images of points within the shaded region"""
        if not relayout_data or not any(
            key.startswith("shapes") for key in relayout_data
        ):
            return html.Div("Drag the shaded region to see points along the trajectory")

        # Extract shaded region position from relayout_data
        region_center = None
        if trajectory_mode == "x":
            # For x-mode, look for x0 and x1 to get center
            x0_key = next((k for k in relayout_data if k.endswith(".x0")), None)
            x1_key = next((k for k in relayout_data if k.endswith(".x1")), None)
            if x0_key and x1_key:
                x0 = relayout_data[x0_key]
                x1 = relayout_data[x1_key]
                region_center = (x0 + x1) / 2
        else:
            # For y-mode, look for y0 and y1 to get center
            y0_key = next((k for k in relayout_data if k.endswith(".y0")), None)
            y1_key = next((k for k in relayout_data if k.endswith(".y1")), None)
            if y0_key and y1_key:
                y0 = relayout_data[y0_key]
                y1 = relayout_data[y1_key]
                region_center = (y0 + y1) / 2

        if region_center is None:
            return html.Div("Drag the shaded region to see points along the trajectory")

        # Calculate tolerance and find nearby points
        if trajectory_mode == "x":
            data_range = (
                self.filtered_features_df[x_axis].max()
                - self.filtered_features_df[x_axis].min()
            )
            tolerance = data_range * 0.05
            distances = abs(self.filtered_features_df[x_axis] - region_center)
            sort_by = y_axis
        else:
            data_range = (
                self.filtered_features_df[y_axis].max()
                - self.filtered_features_df[y_axis].min()
            )
            tolerance = data_range * 0.05
            distances = abs(self.filtered_features_df[y_axis] - region_center)
            sort_by = x_axis

        # Get points within tolerance
        nearby_points = self.filtered_features_df[distances <= tolerance].sort_values(
            sort_by
        )

        if len(nearby_points) == 0:
            return html.Div("No points found in the selected region")

        # Create channel rows
        channel_rows = []
        for channel in self.channels_to_display:
            images = []
            for _, row in nearby_points.iterrows():
                cache_key = (row["fov_name"], row["track_id"], row["t"])
                if cache_key in self.image_cache:
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
                                html.Div(
                                    f"{x_axis}: {row[x_axis]:.2f}, {y_axis}: {row[y_axis]:.2f}",
                                    style={
                                        "textAlign": "center",
                                        "fontSize": "10px",
                                        "color": "#666",
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

        return html.Div(channel_rows)

    def _get_trajectory_images_lasso(self, x_axis, y_axis, selected_data):
        """Get images of points selected by lasso"""
        if not selected_data or not selected_data.get("points"):
            return html.Div("Use the lasso tool to select points")

        selected_info = []
        seen_points = set()

        for point in selected_data["points"]:
            text = point["text"]
            lines = text.split("<br>")
            track_id = int(lines[0].split(": ")[1])
            t = int(lines[1].split(": ")[1])
            fov = lines[2].split(": ")[1]

            point_id = (track_id, t, fov)
            if point_id not in seen_points:
                seen_points.add(point_id)
                # Check if the point exists in the cache before adding
                cache_key = (fov, track_id, t)
                if cache_key in self.image_cache:
                    selected_info.append(
                        {
                            "track_id": track_id,
                            "t": t,
                            "fov_name": fov,
                            x_axis: point["x"],
                            y_axis: point["y"],
                            "cluster": len(selected_info) // 10 + 1,
                        }
                    )
                else:
                    logger.debug(f"Skipping point {point_id} as it's not in the cache")

        if not selected_info:
            return html.Div("No cached images found for the selected points")

        selected_points = pd.DataFrame(selected_info)

        # Get the corresponding points from filtered_features_df using merge
        nearby_points = pd.merge(
            self.filtered_features_df,
            selected_points[["track_id", "t", "fov_name", "cluster"]],
            on=["track_id", "t", "fov_name"],
            how="inner",
        ).sort_values(x_axis)

        if len(nearby_points) == 0:
            return html.Div("No points selected")

        # Create sections for each cluster
        cluster_sections = []
        for cluster_id in sorted(selected_points["cluster"].unique()):
            cluster_points = nearby_points[nearby_points["cluster"] == cluster_id]

            # Create channel rows for this cluster
            channel_rows = []
            for channel in self.channels_to_display:
                images = []
                for _, row in cluster_points.iterrows():
                    cache_key = (row["fov_name"], row["track_id"], row["t"])
                    if cache_key in self.image_cache:
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
                                    html.Div(
                                        f"{x_axis}: {row[x_axis]:.2f}, {y_axis}: {row[y_axis]:.2f}",
                                        style={
                                            "textAlign": "center",
                                            "fontSize": "10px",
                                            "color": "#666",
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
                                f"Cluster {cluster_id}",
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
                    f"Selected Points ({len(cluster_sections)} clusters)",
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
