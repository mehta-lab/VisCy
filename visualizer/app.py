"""
Dash application setup, layout, and callbacks.

This module provides functions for creating and configuring the Dash web
application, including layout definition and interactive callbacks.

Functions
---------
create_app : Create and configure Dash application
run_app : Run the Dash application server
"""

import logging

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

from .config import MultiDatasetConfig
from .data_loading import load_multiple_datasets
from .image_cache import MultiDatasetImageCache
from .timeline import create_track_timeline
from .visualization import create_phate_figure

logger = logging.getLogger(__name__)


def create_app(config: MultiDatasetConfig) -> Dash:
    """
    Create and configure Dash application.

    Parameters
    ----------
    config : MultiDatasetConfig
        Configuration object specifying datasets and application parameters.

    Returns
    -------
    Dash
        Configured Dash application instance.
    """
    logger.info("Loading data...")

    if len(config.datasets) == 1:
        logger.info("=== SINGLE-DATASET MODE ===")
    else:
        logger.info(f"=== MULTI-DATASET MODE ({len(config.datasets)} datasets) ===")

    adata, plot_df, track_options, has_annotations = load_multiple_datasets(config)

    logger.info("Initializing image cache...")
    image_cache = MultiDatasetImageCache(config.datasets, config.yx_patch_size)

    if has_annotations and "annotation" in plot_df.columns:
        annotation_values = sorted(
            [s for s in plot_df["annotation"].unique() if pd.notna(s)]
        )
    else:
        annotation_values = []

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H1(
                "PHATE Track Viewer with Infection Annotations",
                style={"textAlign": "center", "marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Color points by:", style={"fontWeight": "bold"}
                            ),
                            dcc.Dropdown(
                                id="color-mode",
                                options=[
                                    opt
                                    for opt in [
                                        (
                                            {
                                                "label": "Annotation",
                                                "value": "annotation",
                                            }
                                            if has_annotations
                                            else None
                                        ),
                                        {"label": "Time", "value": "time"},
                                        {"label": "Track ID", "value": "track_id"},
                                        {"label": "Dataset", "value": "dataset"},
                                    ]
                                    if opt is not None
                                ],
                                value=(
                                    config.default_color_mode
                                    if has_annotations
                                    else "time"
                                ),
                                clearable=False,
                                style={"width": "200px", "marginLeft": "10px"},
                            ),
                        ],
                        style={
                            "marginBottom": "15px",
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Filter by annotation:", style={"fontWeight": "bold"}
                            ),
                            dcc.Checklist(
                                id="annotation-filter",
                                options=[
                                    {
                                        "label": status.replace("_", " ").title(),
                                        "value": status,
                                    }
                                    for status in annotation_values
                                ],
                                value=annotation_values,
                                inline=True,
                                style={"marginLeft": "10px"},
                            ),
                        ],
                        style={
                            "marginBottom": "15px",
                            "display": "none" if not has_annotations else "block",
                        },
                        id="annotation-filter-container",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Selected Tracks:", style={"fontWeight": "bold"}
                            ),
                            dcc.Dropdown(
                                id="track-selector",
                                options=[
                                    {"label": track, "value": track}
                                    for track in track_options
                                ],
                                multi=True,
                                placeholder="Click points on plot to select tracks, or choose from dropdown...",
                                value=[],
                                style={"width": "100%"},
                            ),
                        ],
                        style={"marginBottom": "15px"},
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="show-trajectories",
                                options=[
                                    {
                                        "label": " Show track trajectories (lines connecting points in temporal order)",
                                        "value": "show",
                                    }
                                ],
                                value=[],
                                inline=True,
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Highlight timepoint (for selected tracks):",
                                style={"fontWeight": "bold", "marginRight": "10px"},
                            ),
                            dcc.Input(
                                id="highlight-timepoint",
                                type="number",
                                placeholder="Enter timepoint (e.g., 0, 1, 2...)",
                                min=0,
                                step=1,
                                style={"width": "200px"},
                            ),
                            html.Span(
                                " ⭐ Highlights as yellow star on plot",
                                style={
                                    "marginLeft": "10px",
                                    "color": "#666",
                                    "fontSize": "14px",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "10px",
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                ],
                style={
                    "padding": "20px",
                    "backgroundColor": "#f9f9f9",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
            ),
            dcc.Loading(
                dcc.Graph(
                    id="phate-scatter",
                    config={
                        "displayModeBar": True,
                        "scrollZoom": True,
                        "displaylogo": False,
                    },
                ),
                type="default",
            ),
            html.Div(
                id="track-timelines",
                style={
                    "marginTop": "30px",
                    "padding": "20px",
                },
            ),
        ],
        style={
            "maxWidth": "1800px",
            "margin": "0 auto",
            "padding": "20px",
            "fontFamily": "Arial, sans-serif",
        },
    )

    @app.callback(
        Output("phate-scatter", "figure"),
        [
            Input("color-mode", "value"),
            Input("annotation-filter", "value"),
            Input("track-selector", "value"),
            Input("show-trajectories", "value"),
            Input("highlight-timepoint", "value"),
        ],
    )
    def update_phate_plot(
        color_mode,
        selected_annotations,
        selected_tracks,
        show_trajectories,
        highlight_timepoint,
    ):
        """Update PHATE scatter plot based on filters and selections."""
        show_traj = "show" in show_trajectories if show_trajectories else False
        return create_phate_figure(
            plot_df,
            color_mode,
            selected_annotations,
            selected_tracks,
            show_traj,
            highlight_timepoint,
        )

    @app.callback(
        Output("annotation-filter-container", "style"), Input("color-mode", "value")
    )
    def toggle_annotation_filter(color_mode):
        """Show annotation filter only when coloring by annotation."""
        if color_mode == "annotation" and has_annotations:
            return {"marginBottom": "15px", "display": "block"}
        return {"marginBottom": "15px", "display": "none"}

    @app.callback(
        Output("track-selector", "value"),
        Input("phate-scatter", "clickData"),
        State("track-selector", "value"),
    )
    def add_track_on_click(click_data, current_tracks):
        """Add clicked track to selection."""
        if click_data is None:
            return current_tracks

        point = click_data["points"][0]
        track_key = point["customdata"][0]

        if current_tracks is None:
            current_tracks = []

        if track_key not in current_tracks:
            return current_tracks + [track_key]

        return current_tracks

    @app.callback(
        Output("track-timelines", "children"), Input("track-selector", "value")
    )
    def update_timelines(selected_tracks):
        """Update track timeline displays."""
        if not selected_tracks:
            return html.Div(
                "Select tracks by clicking points on the PHATE plot or using the dropdown above.",
                style={
                    "textAlign": "center",
                    "color": "#999",
                    "padding": "50px",
                    "fontSize": "18px",
                },
            )

        # Get channels from the first dataset as a default (not used for multi-dataset)
        default_channels = list(config.datasets[0].channels) if config.datasets else []
        return create_track_timeline(
            selected_tracks, adata, plot_df, image_cache, default_channels
        )

    logger.info(f"✓ Dash app created with {len(track_options)} tracks")

    return app


def run_app(app: Dash, debug: bool = False, port: int = 8050):
    """
    Run the Dash application server.

    Parameters
    ----------
    app : Dash
        Configured Dash application instance.
    debug : bool, optional
        Enable debug mode (default: False).
    port : int, optional
        Server port (default: 8050).
    """
    logger.info(f"Starting PHATE Track Viewer on http://localhost:{port}")
    app.run(debug=debug, port=port, host="0.0.0.0")
