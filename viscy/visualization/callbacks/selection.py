"""Selection-related callbacks for the visualization app."""

import json
from typing import Dict, Any, List, Optional

import dash
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from viscy.visualization.components.scrollable_container import ScrollableContainer


class SelectionCallbacks:
    """Class to handle selection-related callbacks."""

    def __init__(self, app):
        """Initialize the selection callbacks.

        Args:
            app: The main visualization app instance.
        """
        self.app = app

    def register(self):
        """Register all selection-related callbacks."""
        self._register_selection_callback()
        self._register_clear_selection_callback()

    def _register_selection_callback(self):
        """Register callback for point selection."""

        @self.app.app.callback(
            [
                Output("track-timeline", "children", allow_duplicate=True),
                Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [Input("scatter-plot", "clickData")],
            [
                State("scatter-plot", "selectedData"),
                State("view-tabs", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_selection(
            click_data: Optional[Dict[str, Any]],
            selected_data: Optional[Dict[str, Any]],
            active_tab: str,
        ):
            """Update selection based on click or lasso."""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            # If we're in clusters tab and have lasso selection, keep it
            if active_tab == "clusters-tab" and selected_data:
                return dash.no_update, selected_data

            # Handle single point click for timeline view
            if click_data and click_data.get("points"):
                point = click_data["points"][0]
                text = point["text"]
                lines = text.split("<br>")

                # Extract track information
                if "Cluster:" in lines[0]:
                    # Clicked on a cluster point
                    track_id = int(lines[1].split(": ")[1])
                    t = int(lines[2].split(": ")[1])
                    fov = lines[3].split(": ")[1]
                else:
                    # Clicked on a regular point
                    track_id = int(lines[0].split(": ")[1])
                    t = int(lines[1].split(": ")[1])
                    fov = lines[2].split(": ")[1]

                # Get all timepoints for this track
                track_data = self.app.filtered_features_df[
                    (self.app.filtered_features_df["fov_name"] == fov)
                    & (self.app.filtered_features_df["track_id"] == track_id)
                ].sort_values("t")

                if track_data.empty:
                    return (
                        html.Div(f"No timeline data found for Track {track_id}"),
                        None,  # Clear selection
                    )

                # Create image grids for each channel
                channel_grids = []
                for channel in self.app.channels_to_display:
                    images = []
                    for _, row in track_data.iterrows():
                        cache_key = (row["fov_name"], row["track_id"], row["t"])
                        if (
                            cache_key in self.app.image_cache
                            and channel in self.app.image_cache[cache_key]
                        ):
                            images.append(
                                html.Div(
                                    [
                                        html.Img(
                                            src=self.app.image_cache[cache_key][
                                                channel
                                            ],
                                            style={
                                                "width": "150px",
                                                "height": "150px",
                                                "border": (
                                                    "2px solid #ffd700"  # Gold border for clicked timepoint
                                                    if row["t"] == t
                                                    else "1px solid #ddd"
                                                ),
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
                                            f"Time: {row['t']}",
                                            style={
                                                "fontSize": "12px",
                                                "textAlign": "center",
                                                "marginTop": "5px",
                                                "fontWeight": (
                                                    "bold"
                                                    if row["t"] == t
                                                    else "normal"
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

                    if images:
                        channel_grids.append(
                            html.Div(
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
                                        images,
                                        style={
                                            "whiteSpace": "nowrap",
                                            "marginBottom": "20px",
                                        },
                                    ),
                                ]
                            )
                        )

                if not channel_grids:
                    return (
                        html.Div("No images found in cache for this track."),
                        None,  # Clear selection
                    )

                return (
                    ScrollableContainer(
                        title=f"Track {track_id} Timeline (FOV: {fov})",
                        content=html.Div(
                            channel_grids,
                            style={"padding": "20px"},
                        ),
                        max_height="80vh",
                        direction="horizontal",
                    ).create_layout(),
                    None,  # Clear selection after showing timeline
                )

            return dash.no_update, dash.no_update

    def _register_clear_selection_callback(self):
        """Register callback for clearing selection."""

        @self.app.app.callback(
            Output("scatter-plot", "selectedData"),
            [Input("clear-selection", "n_clicks")],
            prevent_initial_call=True,
        )
        def clear_selection(n_clicks):
            """Clear the current selection."""
            if n_clicks:
                return None
            raise PreventUpdate
